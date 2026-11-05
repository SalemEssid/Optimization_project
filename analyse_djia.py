"""
Projet d'optimisation - Analyse de données financières (DJIA, 2015-2025)

Partie I  : régression linéaire (équation normale, vérifications numériques)
Partie II : SVM linéaire (hyperplan séparant hausse / baisse)

Remarque : dans le fichier fourni, la colonne « Price » correspond au cours de
clôture (Close) ; elle est renommée « Close » dès le chargement.

Sorties : figures/*.svg|png et resultats.json (lus par le générateur du rapport).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from scipy import optimize, stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent
CSV = BASE / "Dow Jones Industrial.csv"
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# Style des graphiques (palette de référence, encre neutre, grille discrète)
# ----------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, PLANE = "#e1e0d9", "#c3c2b7", "#f0efec"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "figure.dpi": 110,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "svg.fonttype": "path",
    "axes.unicode_minus": True,
})


def fr_num(x, d=2):
    """Nombre au format français (virgule décimale, signe moins typographique)."""
    s = f"{x:.{d}f}".replace(".", ",")
    return s.replace("-", "−")


pct = lambda d: FuncFormatter(lambda v, _: fr_num(100 * v, d) + " %")
plain = lambda d: FuncFormatter(lambda v, _: fr_num(v, d))


def save(fig, name):
    fig.savefig(FIG / f"{name}.svg")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)


res = {}

# ----------------------------------------------------------------------------
# 1. Chargement, nettoyage, tri chronologique
# ----------------------------------------------------------------------------
raw = pd.read_csv(CSV, encoding="utf-8-sig", dtype=str)
df = raw.rename(columns={"Price": "Close"}).copy()
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
for c in ["Close", "Open", "High", "Low"]:
    df[c] = df[c].str.replace(",", "").astype(float)
df = df.sort_values("Date").reset_index(drop=True)
T_raw = len(df)

# Le NYSE est fermé le 25 décembre : ces lignes recopient la séance précédente
# (rendement nul artificiel) et ne sont pas des séances boursières.
noel = (df.Date.dt.month == 12) & (df.Date.dt.day == 25)
removed = df.loc[noel, "Date"].dt.strftime("%d/%m/%Y").tolist()
df = df.loc[~noel].reset_index(drop=True)
T = len(df)

close, high, low = df.Close.to_numpy(), df.High.to_numpy(), df.Low.to_numpy()
dates = df.Date.to_numpy()

# Indices 0-based : la séance t (1-based) correspond à i = t - 1.
# R_t = ln(Close_t / Close_{t-1}) défini pour t >= 2 ; on garde t = 2..T-1.
R = np.full(T, np.nan)
R[1:] = np.log(close[1:] / close[:-1])
idx = np.arange(1, T - 1)
x1 = R[idx]
x2 = (high[idx] - low[idx]) / close[idx]
y = R[idx + 1]
d_obs = dates[idx]
n = len(idx)
assert n == T - 2

n_tr = int(0.8 * n)
n_te = n - n_tr
tr, te = slice(0, n_tr), slice(n_tr, n)

res["data"] = dict(
    T_raw=T_raw, T=T, n=n, n_tr=n_tr, n_te=n_te, removed=removed,
    first_date=pd.Timestamp(dates[0]).strftime("%d/%m/%Y"),
    last_date=pd.Timestamp(dates[-1]).strftime("%d/%m/%Y"),
    tr_start=pd.Timestamp(d_obs[0]).strftime("%d/%m/%Y"),
    tr_end=pd.Timestamp(d_obs[n_tr - 1]).strftime("%d/%m/%Y"),
    te_start=pd.Timestamp(d_obs[n_tr]).strftime("%d/%m/%Y"),
    te_end=pd.Timestamp(d_obs[-1]).strftime("%d/%m/%Y"),
    close_first=close[0], close_last=close[-1],
    close_min=close.min(), close_min_date=pd.Timestamp(dates[close.argmin()]).strftime("%d/%m/%Y"),
    close_max=close.max(), close_max_date=pd.Timestamp(dates[close.argmax()]).strftime("%d/%m/%Y"),
)


def describe(v):
    return dict(mean=v.mean(), std=v.std(ddof=1), min=v.min(), max=v.max(),
                skew=stats.skew(v), kurt=stats.kurtosis(v))  # kurtosis en excès


res["desc"] = {
    "x1_tr": describe(x1[tr]), "x2_tr": describe(x2[tr]), "y_tr": describe(y[tr]),
    "x1_te": describe(x1[te]), "x2_te": describe(x2[te]), "y_te": describe(y[te]),
}
res["sessions_per_year"] = df.groupby(df.Date.dt.year).size().to_dict()
res["corr_tr"] = dict(
    x1_x2=np.corrcoef(x1[tr], x2[tr])[0, 1],
    y_x1=np.corrcoef(y[tr], x1[tr])[0, 1],
    y_x2=np.corrcoef(y[tr], x2[tr])[0, 1],
)
res["corr_te"] = dict(
    x1_x2=np.corrcoef(x1[te], x2[te])[0, 1],
    y_x1=np.corrcoef(y[te], x1[te])[0, 1],
    y_x2=np.corrcoef(y[te], x2[te])[0, 1],
)
# Aperçu des premières lignes de X et y (dates des séances t)
res["X_head"] = [dict(date=pd.Timestamp(d_obs[i]).strftime("%d/%m/%Y"), x1=x1[i], x2=x2[i], y=y[i])
                 for i in (0, 1, 2, n_tr - 1, n - 1)]
res["X_head_idx"] = [0, 1, 2, n_tr - 1, n - 1]
i_min = np.argmin(x1)
res["extremes"] = dict(
    r_min=x1[i_min], r_min_date=pd.Timestamp(d_obs[i_min]).strftime("%d/%m/%Y"),
    r_max=x1.max(), r_max_date=pd.Timestamp(d_obs[np.argmax(x1)]).strftime("%d/%m/%Y"),
    x2_max=x2.max(), x2_max_date=pd.Timestamp(d_obs[np.argmax(x2)]).strftime("%d/%m/%Y"),
)
split_date = d_obs[n_tr]

# --- Figure 1 : cours de clôture
fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.axvspan(split_date, dates[-1], color=PLANE, lw=0)
ax.plot(dates, close, color=BLUE, lw=1.3)
ax.axvline(split_date, color=MUTED, lw=0.9)
ax.set_title("Cours de clôture quotidien du Dow Jones (Close = colonne Price)")
ax.set_xlabel("Date")
ax.set_ylabel("Points d'indice")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f} k"))
ax.set_xlim(dates[0], dates[-1])
yl = ax.get_ylim()
ax.text(pd.Timestamp(d_obs[n_tr // 2]), yl[1] * 0.97, "Entraînement (80 %)", ha="center", va="top", color=INK2)
ax.text(pd.Timestamp(d_obs[n_tr + n_te // 2]), yl[1] * 0.97, "Test (20 %)", ha="center", va="top", color=INK2)
save(fig, "fig1_close")

# --- Figure 2 : rendements et amplitude relative (deux panneaux, un axe chacun)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 4.4), sharex=True)
for a in (a1, a2):
    a.axvspan(split_date, dates[-1], color=PLANE, lw=0)
    a.axvline(split_date, color=MUTED, lw=0.9)
a1.plot(d_obs, x1, color=BLUE, lw=0.6)
a1.set_title(r"Rendement logarithmique quotidien $R_t = \ln(\mathrm{Close}_t/\mathrm{Close}_{t-1})$")
a1.set_ylabel("$R_t$")
a1.yaxis.set_major_formatter(pct(0))
a2.plot(d_obs, x2, color=ORANGE, lw=0.6)
a2.set_title(r"Amplitude relative de séance $x_{2,t} = (\mathrm{High}_t-\mathrm{Low}_t)/\mathrm{Close}_t$")
a2.set_ylabel("$x_{2,t}$")
a2.set_xlabel("Date (zone grisée : période de test)")
a2.yaxis.set_major_formatter(pct(0))
a2.set_xlim(d_obs[0], d_obs[-1])
fig.tight_layout()
save(fig, "fig2_rendements")

# ----------------------------------------------------------------------------
# Partie I - Régression linéaire
# ----------------------------------------------------------------------------
X = np.column_stack([np.ones(n), x1, x2])
Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]


def J(beta, X=Xtr, y=ytr):
    r = X @ beta - y
    return r @ r / (2 * len(y))


def gradJ(beta, X=Xtr, y=ytr):
    return X.T @ (X @ beta - y) / len(y)


XtX = Xtr.T @ Xtr
Xty = Xtr.T @ ytr
H = XtX / n_tr                       # hessienne de J (constante)
eig = np.linalg.eigvalsh(H)
beta_ne = np.linalg.solve(XtX, Xty)  # équation normale X^T X b = X^T y

# Forme centrée équivalente : (b1, b2) = S_xx^{-1} S_xy, b0 = ybar - b1 x1bar - b2 x2bar
Z = Xtr[:, 1:] - Xtr[:, 1:].mean(0)
Sxx = Z.T @ Z / n_tr
Sxy = Z.T @ (ytr - ytr.mean()) / n_tr
b12 = np.linalg.solve(Sxx, Sxy)
b0c = ytr.mean() - b12 @ Xtr[:, 1:].mean(0)

res["ne"] = dict(
    XtX=XtX.tolist(), Xty=Xty.tolist(), H=H.tolist(), eig=eig.tolist(),
    cond=eig[-1] / eig[0], beta=beta_ne.tolist(),
    grad_norm=float(np.linalg.norm(gradJ(beta_ne))),
    J_star=float(J(beta_ne)), J_zero=float(J(np.zeros(3))),
    Sxx=Sxx.tolist(), Sxy=Sxy.tolist(), b_centered=[b0c, *b12],
    xbar=Xtr[:, 1:].mean(0).tolist(), ybar=ytr.mean(),
)

# Méthodes Python de vérification
lr = LinearRegression().fit(Xtr[:, 1:], ytr)
beta_sk = np.r_[lr.intercept_, lr.coef_]
beta_ls = np.linalg.lstsq(Xtr, ytr, rcond=None)[0]
bfgs = optimize.minimize(J, np.zeros(3), jac=gradJ, method="BFGS",
                         options=dict(gtol=1e-13, maxiter=10_000))
beta_bfgs = bfgs.x
# Méthode de Newton (région de confiance) : une seule itération suffit pour une quadratique
newton = optimize.minimize(J, np.zeros(3), jac=gradJ, hess=lambda b: XtX / n_tr, method="trust-exact")
beta_newton = newton.x


def gd(Xm, ym, beta_star, max_iter, tol=1e-10):
    """Descente de gradient à pas fixe 1/L ; renvoie beta et l'historique J - J*."""
    A = Xm.T @ Xm / len(ym)
    c = Xm.T @ ym / len(ym)
    L = np.linalg.eigvalsh(A)[-1]
    beta = np.zeros(3)
    ks, gaps = [], []
    rec = set(np.unique(np.geomspace(1, max_iter, 400).astype(int)))
    for k in range(1, max_iter + 1):
        beta = beta - (A @ beta - c) / L
        e = beta - beta_star
        if k in rec:
            ks.append(k)
            gaps.append(0.5 * e @ A @ e)       # J(beta_k) - J* (exact pour une quadratique)
        if np.linalg.norm(e) <= tol * np.linalg.norm(beta_star):
            ks.append(k)
            gaps.append(0.5 * e @ A @ e)
            break
    return beta, k, np.array(ks), np.array(gaps), L


# (a) variables brutes : mal conditionné
beta_gd, k_gd, ks_raw, gaps_raw, L_raw = gd(Xtr, ytr, beta_ne, max_iter=3_000_000)
# (b) variables standardisées (moyenne/écart-type d'entraînement), puis retour aux unités d'origine
m12, s12 = Xtr[:, 1:].mean(0), Xtr[:, 1:].std(0)
Xs = np.column_stack([np.ones(n_tr), (Xtr[:, 1:] - m12) / s12])
gam_star = np.linalg.solve(Xs.T @ Xs, Xs.T @ ytr)
gam, k_gds, ks_std, gaps_std, L_std = gd(Xs, ytr, gam_star, max_iter=100_000)
beta_gds = np.r_[gam[0] - (gam[1:] * m12 / s12).sum(), gam[1:] / s12]
Hs_eig = np.linalg.eigvalsh(Xs.T @ Xs / n_tr)

methods = {
    "Équation normale (np.linalg.solve)": beta_ne,
    "sklearn LinearRegression": beta_sk,
    "np.linalg.lstsq (SVD)": beta_ls,
    "scipy.optimize.minimize (BFGS)": beta_bfgs,
    "scipy.optimize.minimize (Newton, trust-exact)": beta_newton,
    "Descente de gradient (variables brutes)": beta_gd,
    "Descente de gradient (variables standardisées)": beta_gds,
}
res["methods"] = {k: dict(beta=v.tolist(), dev=float(np.max(np.abs(v - beta_ne))), J=float(J(v)))
                  for k, v in methods.items()}
res["gd"] = dict(k_raw=k_gd, k_std=k_gds, L_raw=L_raw, L_std=L_std, cond_raw=eig[-1] / eig[0],
                 cond_std=Hs_eig[-1] / Hs_eig[0], bfgs_nit=int(bfgs.nit), bfgs_msg=str(bfgs.message),
                 bfgs_success=bool(bfgs.success), newton_nit=int(newton.nit),
                 newton_success=bool(newton.success))

# --- Figure 4 : convergence de la descente de gradient
kr = eig[-1] / eig[0]
ex = int(np.floor(np.log10(kr)))
fig, ax = plt.subplots(figsize=(7.2, 3.0))
mt = lambda x, d=2: fr_num(x, d).replace(",", "{,}")   # virgule décimale en mathtext
k_gd_txt = f"{k_gd:,}".replace(",", " ")          # séparateur de milliers français
ax.loglog(ks_raw, gaps_raw, color=BLUE, lw=1.6,
          label=rf"Variables brutes ($\kappa \approx {mt(kr / 10**ex)}\cdot 10^{{{ex}}}$, {k_gd_txt} itérations)")
ax.loglog(ks_std, gaps_std, color=ORANGE, lw=1.6,
          label=rf"Variables standardisées ($\kappa \approx {mt(Hs_eig[-1] / Hs_eig[0])}$, {k_gds} itérations)")
ax.set_title(r"Descente de gradient à pas $1/L$ : écart $J(\beta^{(k)})-J(\hat\beta)$")
ax.set_xlabel("Itération k (échelle log)")
ax.set_ylabel(r"$J(\beta^{(k)})-J(\hat\beta)$ (échelle log)")
ax.legend(loc="lower left")
save(fig, "fig4_gd")

# Écarts-types (classiques et robustes de White HC1), tests de Student
e_tr = ytr - Xtr @ beta_ne
dfree = n_tr - 3
s2 = e_tr @ e_tr / dfree
XtX_inv = np.linalg.inv(XtX)
se = np.sqrt(np.diag(s2 * XtX_inv))
meat = (Xtr * e_tr[:, None] ** 2).T @ Xtr
se_w = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv * n_tr / dfree))
tval, tval_w = beta_ne / se, beta_ne / se_w
pval = 2 * stats.t.sf(np.abs(tval), dfree)
pval_w = 2 * stats.t.sf(np.abs(tval_w), dfree)
res["inference"] = dict(se=se.tolist(), t=tval.tolist(), p=pval.tolist(),
                        se_w=se_w.tolist(), t_w=tval_w.tolist(), p_w=pval_w.tolist(),
                        sigma=float(np.sqrt(s2)))

# Prédictions et métriques
yhat_tr, yhat_te = Xtr @ beta_ne, Xte @ beta_ne
ybar = ytr.mean()
base_tr, base_te = np.full(n_tr, ybar), np.full(n_te, ybar)


def metrics(yv, pv):
    return dict(mse=mean_squared_error(yv, pv), r2=r2_score(yv, pv),
                rmse=float(np.sqrt(mean_squared_error(yv, pv))))


res["metrics"] = dict(
    model_tr=metrics(ytr, yhat_tr), model_te=metrics(yte, yhat_te),
    base_tr=metrics(ytr, base_tr), base_te=metrics(yte, base_te),
    # R² hors échantillon relatif à la moyenne d'entraînement (Campbell-Thompson)
    r2_os=1 - ((yte - yhat_te) ** 2).sum() / ((yte - ybar) ** 2).sum(),
    dir_tr=float(np.mean(np.sign(yhat_tr) == np.sign(ytr))),
    dir_te=float(np.mean(np.sign(yhat_te) == np.sign(yte))),
    up_tr=float(np.mean(ytr > 0)), up_te=float(np.mean(yte > 0)),
    yhat_tr_range=[float(yhat_tr.min()), float(yhat_tr.max())],
    yhat_te_range=[float(yhat_te.min()), float(yhat_te.max())],
    yhat_tr_std=float(yhat_tr.std()), yhat_te_std=float(yhat_te.std()),
    pos_pred_tr=float(np.mean(yhat_tr > 0)), pos_pred_te=float(np.mean(yhat_te > 0)),
    resid_kurt=float(stats.kurtosis(e_tr)), resid_skew=float(stats.skew(e_tr)),
    jb_p=float(stats.jarque_bera(e_tr).pvalue),
)

# Robustesse : mêmes calculs sur le fichier brut (lignes du 25/12 conservées)
close_r = raw["Price"].str.replace(",", "").astype(float).to_numpy()[::-1]
high_r = raw["High"].str.replace(",", "").astype(float).to_numpy()[::-1]
low_r = raw["Low"].str.replace(",", "").astype(float).to_numpy()[::-1]
Rr = np.r_[np.nan, np.log(close_r[1:] / close_r[:-1])]
ir = np.arange(1, len(close_r) - 1)
Xr = np.column_stack([np.ones(len(ir)), Rr[ir], (high_r[ir] - low_r[ir]) / close_r[ir]])
yr = Rr[ir + 1]
ntr_r = int(0.8 * len(ir))
beta_r = np.linalg.solve(Xr[:ntr_r].T @ Xr[:ntr_r], Xr[:ntr_r].T @ yr[:ntr_r])
res["robust_raw"] = dict(n=len(ir), n_tr=ntr_r, beta=beta_r.tolist(),
                         r2_tr=r2_score(yr[:ntr_r], Xr[:ntr_r] @ beta_r),
                         r2_te=r2_score(yr[ntr_r:], Xr[ntr_r:] @ beta_r))

# Sensibilité à l'épisode Covid : entraînement sans les séances du 20/02/2020 au 30/04/2020
covid = (d_obs[tr] >= np.datetime64("2020-02-20")) & (d_obs[tr] <= np.datetime64("2020-04-30"))
Xc, yc = Xtr[~covid], ytr[~covid]
beta_c = np.linalg.solve(Xc.T @ Xc, Xc.T @ yc)
e_c = yc - Xc @ beta_c
XtX_c_inv = np.linalg.inv(Xc.T @ Xc)
se_wc = np.sqrt(np.diag(XtX_c_inv @ ((Xc * e_c[:, None] ** 2).T @ Xc) @ XtX_c_inv * len(yc) / (len(yc) - 3)))
res["no_covid"] = dict(n_removed=int(covid.sum()), beta=beta_c.tolist(), t_w=(beta_c / se_wc).tolist(),
                       r2_tr=r2_score(yc, Xc @ beta_c), r2_te=r2_score(yte, Xte @ beta_c),
                       mse_te=mean_squared_error(yte, Xte @ beta_c))

# --- Figure 3 : nuages y vs x1 et y vs x2 (entraînement) + droites partielles
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
a1.scatter(x1[tr], ytr, s=7, color=BLUE, alpha=0.35, lw=0, label="Observations (entraînement)")
g = np.linspace(x1[tr].min(), x1[tr].max(), 50)
a1.plot(g, beta_ne[0] + beta_ne[1] * g + beta_ne[2] * x2[tr].mean(), color=ORANGE, lw=1.8,
        label=r"$\hat y$ avec $x_2=\bar x_2$")
a1.set_title(r"$y_t=R_{t+1}$ en fonction de $x_{1,t}=R_t$")
a1.set_xlabel("$x_{1,t}$ (rendement du jour)")
a1.set_ylabel("$y_t$ (rendement du lendemain)")
g2 = np.linspace(x2[tr].min(), x2[tr].max(), 50)
a2.scatter(x2[tr], ytr, s=7, color=BLUE, alpha=0.35, lw=0, label="Observations (entraînement)")
a2.plot(g2, beta_ne[0] + beta_ne[1] * x1[tr].mean() + beta_ne[2] * g2, color=ORANGE, lw=1.8,
        label=r"$\hat y$ avec $x_1=\bar x_1$")
a2.set_title(r"$y_t$ en fonction de $x_{2,t}$")
a2.set_xlabel("$x_{2,t}$ (amplitude relative du jour)")
for a in (a1, a2):
    a.xaxis.set_major_formatter(pct(0))
    a.yaxis.set_major_formatter(pct(0))
    a.legend(loc="lower right" if a is a2 else "lower left", fontsize=7.8)
fig.tight_layout()
save(fig, "fig3_nuages")

# --- Figure 5 : test - rendements réels vs prédits
fig, ax = plt.subplots(figsize=(7.2, 3.0))
ax.plot(d_obs[te], yte, color=AXIS, lw=0.8, label="Rendement réel $y_t$")
ax.plot(d_obs[te], yhat_te, color=BLUE, lw=1.4, label=r"Prédiction $\hat y_t$ (régression)")
ax.axhline(ybar, color=ORANGE, lw=1.4, label=r"Modèle de référence $\bar y_{\mathrm{train}}$")
ax.set_title("Période de test : rendements réels et rendements prédits")
ax.set_xlabel("Date de la séance t (la cible est le rendement de t+1)")
ax.set_ylabel("Rendement")
ax.yaxis.set_major_formatter(pct(0))
ax.set_xlim(d_obs[n_tr], d_obs[-1])
MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{MOIS[mdates.num2date(v).month - 1]} {mdates.num2date(v).year}"))
ax.legend(loc="lower left", ncol=3)
save(fig, "fig5_test")

# --- Figure 6 : résidus (entraînement) - histogramme et diagramme quantile-quantile
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0))
sd = e_tr.std()
a1.hist(e_tr, bins=120, density=True, color=BLUE, alpha=0.8, lw=0)
gg = np.linspace(e_tr.min(), e_tr.max(), 400)
a1.plot(gg, stats.norm.pdf(gg, 0, sd), color=ORANGE, lw=1.6, label="Loi normale de même écart-type")
a1.set_xlim(-6 * sd, 6 * sd)
a1.set_title("Distribution des résidus (entraînement)")
a1.set_xlabel(r"Résidu $y_t-\hat y_t$")
a1.set_ylabel("Densité")
a1.xaxis.set_major_formatter(pct(0))
a1.legend(loc="upper left", fontsize=7.8)
(osm, osr), _ = stats.probplot(e_tr / sd, dist="norm")
a2.scatter(osm, osr, s=7, color=BLUE, alpha=0.6, lw=0, label="Résidus standardisés")
lim = [osm.min(), osm.max()]
a2.plot(lim, lim, color=ORANGE, lw=1.6, label="Droite y = x (normalité)")
a2.set_title("Diagramme quantile-quantile")
a2.set_xlabel("Quantiles théoriques N(0, 1)")
a2.set_ylabel("Quantiles observés")
a2.legend(loc="upper left", fontsize=7.8)
fig.tight_layout()
save(fig, "fig6_residus")

# ----------------------------------------------------------------------------
# Partie II - SVM linéaire
# ----------------------------------------------------------------------------
z = np.where(y > 0, 1, -1)
ztr, zte = z[tr], z[te]
mu, sig = np.array([x1[tr].mean(), x2[tr].mean()]), np.array([x1[tr].std(), x2[tr].std()])
F = (np.column_stack([x1, x2]) - mu) / sig
Ftr, Fte = F[tr], F[te]
res["svm_data"] = dict(mu=mu.tolist(), sig=sig.tolist(),
                       up_tr=int((ztr == 1).sum()), down_tr=int((ztr == -1).sum()),
                       up_te=int((zte == 1).sum()), down_te=int((zte == -1).sum()))

C0 = 1.0


def fit_svm(C, **kw):
    m = SVC(kernel="linear", C=C, **kw).fit(Ftr, ztr)
    w, b = m.coef_.ravel(), m.intercept_[0]
    f_tr = Ftr @ w + b
    alpha = np.abs(m.dual_coef_.ravel())             # alpha_i (dual_coef_ = alpha_i z_i)
    cw = m.class_weight_[np.searchsorted(m.classes_, ztr)]  # poids par observation (1 sans pondération)
    hinge = np.maximum(0, 1 - ztr * f_tr)
    primal = 0.5 * w @ w + C * (cw * hinge).sum()
    dual = alpha.sum() - 0.5 * w @ w
    sv_cw = cw[m.support_]
    bounded = np.isclose(alpha, C * sv_cw, rtol=1e-6, atol=1e-10)
    p_tr, p_te = np.sign(Ftr @ w + b), np.sign(Fte @ w + b)
    p_tr[p_tr == 0] = 1
    p_te[p_te == 0] = 1
    return m, dict(
        C=C, w=w.tolist(), b=float(b), norm_w=float(np.linalg.norm(w)),
        margin=float(2 / np.linalg.norm(w)), n_sv=int(len(m.support_)),
        n_sv_bounded=int(bounded.sum()), n_sv_free=int((~bounded).sum()),
        n_sv_pos=int(m.n_support_[list(m.classes_).index(1)]),
        n_sv_neg=int(m.n_support_[list(m.classes_).index(-1)]),
        primal=float(primal), dual=float(dual), gap_rel=float((primal - dual) / abs(primal)),
        w_from_alpha_err=float(np.max(np.abs(m.dual_coef_.ravel() @ Ftr[m.support_] - w))),
        sum_alpha_z=float(m.dual_coef_.ravel().sum()),
        n_misclassified_tr=int((ztr * f_tr < 0).sum()), n_in_margin=int(((ztr * f_tr >= 0) & (ztr * f_tr < 1)).sum()),
        acc_tr=float(np.mean(p_tr == ztr)), acc_te=float(np.mean(p_te == zte)),
        pred_up_tr=float(np.mean(p_tr == 1)), pred_up_te=float(np.mean(p_te == 1)),
        cm_tr=confusion_matrix(ztr, p_tr, labels=[1, -1]).tolist(),
        cm_te=confusion_matrix(zte, p_te, labels=[1, -1]).tolist(),
        f_range_tr=[float(f_tr.min()), float(f_tr.max())],
        # hyperplan dans les unités d'origine : a1 x1 + a2 x2 + c = 0
        orig=[float(w[0] / sig[0]), float(w[1] / sig[1]), float(b - (w * mu / sig).sum())],
    )


svm, r_svm = fit_svm(C0)
res["svm"] = r_svm
res["svm_C"] = [fit_svm(C)[1] for C in (0.001, 0.01, 0.1, 1, 10, 100)]
svm_bal, r_bal = fit_svm(C0, class_weight="balanced")
r_bal["class_weight"] = {int(c): float(wc) for c, wc in zip(svm_bal.classes_, svm_bal.class_weight_)}
res["svm_bal"] = r_bal
res["svm_baseline"] = dict(acc_tr=float(np.mean(ztr == 1)), acc_te=float(np.mean(zte == 1)))

# Tolérance de libsvm resserrée (1e-3 -> 1e-5) : w doit tendre vers 0 si (w*, b*) = (0, 1)
m_tight = SVC(kernel="linear", C=C0, tol=1e-5).fit(Ftr, ztr)
res["svm_tight"] = dict(w=m_tight.coef_.ravel().tolist(), b=float(m_tight.intercept_[0]),
                        norm_w=float(np.linalg.norm(m_tight.coef_)))

# Certificat d'optimalité de (w, b) = (0, 1) : le barycentre de la classe « baisse »
# doit appartenir à l'enveloppe convexe réduite de la classe « hausse », c.-à-d.
# il existe lambda_i dans [0, 1/n_-], somme = 1, avec sum lambda_i x_i = barycentre(baisse).
P_up, P_dn = Ftr[ztr == 1], Ftr[ztr == -1]
n_up, n_dn = len(P_up), len(P_dn)
A_eq = np.vstack([P_up.T, np.ones(n_up)])
b_eq = np.r_[P_dn.mean(0), 1.0]
lp = optimize.linprog(np.zeros(n_up), A_eq=A_eq, b_eq=b_eq, bounds=[(0, 1 / n_dn)] * n_up, method="highs")
# plus petit poids maximal possible (marge de faisabilité du certificat)
lp2 = optimize.linprog(np.r_[np.zeros(n_up), 1.0],
                       A_ub=np.hstack([np.eye(n_up), -np.ones((n_up, 1))]), b_ub=np.zeros(n_up),
                       A_eq=np.hstack([A_eq, np.zeros((3, 1))]), b_eq=b_eq,
                       bounds=[(0, None)] * (n_up + 1), method="highs")
res["svm_certificate"] = dict(
    feasible=bool(lp.status == 0), min_max_weight=float(lp2.fun), bound=1 / n_dn,
    bary_up=P_up.mean(0).tolist(), bary_dn=P_dn.mean(0).tolist(),
    primal_trivial=float(2 * C0 * n_dn),
)


def class_scatter(a, Fm, zm, s=9, alpha=0.5, chunks=12):
    # tracé entrelacé par paquets pour qu'aucune classe ne masque l'autre
    rng = np.random.default_rng(0)
    order = rng.permutation(len(zm))
    for k, part in enumerate(np.array_split(order, chunks)):
        up, dn = part[zm[part] == 1], part[zm[part] == -1]
        a.scatter(*Fm[up].T, s=s, marker="^", color=BLUE, alpha=alpha, lw=0,
                  label="Hausse ($z_t=+1$)" if k == 0 else None)
        a.scatter(*Fm[dn].T, s=s, marker="v", color=ORANGE, alpha=alpha, lw=0,
                  label="Baisse ($z_t=-1$)" if k == 0 else None)


# --- Figure 7 : les deux classes dans le plan (x1, x2) standardisé
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.6))
class_scatter(a1, Ftr, ztr, s=8)
a1.set_title("Entraînement : vue d'ensemble")
class_scatter(a2, Ftr, ztr, s=12)
a2.set_xlim(-3, 3)
a2.set_ylim(-1.6, 3)
a2.set_title("Zoom sur le cœur du nuage")
for a in (a1, a2):
    a.set_xlabel(r"$\tilde x_1$ (rendement $R_t$ standardisé)")
    a.set_ylabel(r"$\tilde x_2$ (amplitude standardisée)")
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.legend(handles=[Line2D([], [], marker="^", ls="", color=BLUE, label=f"Hausse ($z_t=+1$), {n_up} jours"),
                    Line2D([], [], marker="v", ls="", color=ORANGE, label=f"Baisse ($z_t=-1$), {n_dn} jours")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
save(fig, "fig7_classes")


def draw_svm(a, model, Fm, zm, xlim, ylim, title, lines=True):
    w, b = model.coef_.ravel(), model.intercept_[0]
    gx, gy = np.meshgrid(np.linspace(*xlim, 400), np.linspace(*ylim, 400))
    f = w[0] * gx + w[1] * gy + b
    a.contourf(gx, gy, np.sign(f), levels=[-2, 0, 2], colors=[ORANGE, BLUE], alpha=0.10)
    class_scatter(a, Fm, zm, s=11, alpha=0.55)
    if lines:
        a.contour(gx, gy, f, levels=[-1, 0, 1], colors=[INK2, INK, INK2],
                  linewidths=[0.9, 1.8, 0.9], linestyles=["--", "-", "--"])
    a.set_xlim(*xlim)
    a.set_ylim(*ylim)
    a.set_title(title)
    a.set_xlabel(r"$\tilde x_1$ (rendement standardisé)")
    a.set_ylabel(r"$\tilde x_2$ (amplitude standardisée)")


def svm_legend(fig, lines=True, **kw):
    h = [Line2D([], [], marker="^", ls="", color=BLUE, label="Hausse ($z=+1$)"),
         Line2D([], [], marker="v", ls="", color=ORANGE, label="Baisse ($z=-1$)")]
    if lines:
        h += [Line2D([], [], color=INK, lw=1.8, label=r"Hyperplan $w^Tx+b=0$"),
              Line2D([], [], color=INK2, lw=0.9, ls="--", label=r"Marges $w^Tx+b=\pm 1$")]
    h += [Patch(color=BLUE, alpha=0.18, label="Région prédite « hausse »"),
          Patch(color=ORANGE, alpha=0.18, label="Région prédite « baisse »")]
    fig.legend(handles=h, **kw)


xl, yl = (-4, 4), (-2, 4.5)
for tag, model in (("C1", svm), ("bal", svm_bal)):
    lines = tag == "bal"   # pour C = 1, w ~ 0 : les courbes f = 0, ±1 ne sont que du bruit numérique
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.7), sharey=True)
    lab = "C = 1" if tag == "C1" else "C = 1, poids équilibrés"
    draw_svm(a1, model, Ftr, ztr, xl, yl, f"Entraînement ({lab})", lines)
    draw_svm(a2, model, Fte, zte, xl, yl, f"Test ({lab})", lines)
    a2.set_ylabel("")
    if not lines:
        lo, hi = r_svm["f_range_tr"]
        dist = abs(r_svm["b"]) / r_svm["norm_w"]
        a1.text(0.03, 0.97,
                rf"$\|w\| \approx {mt(r_svm['norm_w'] * 1e5)}\cdot 10^{{-5}}$, $b \approx {mt(r_svm['b'], 5)}$" "\n"
                rf"$w^Tx+b \in [{mt(lo, 4)}\,;\,{mt(hi, 4)}]$ sur tout l'échantillon" "\n"
                rf"hyperplan $w^Tx+b=0$ à $\approx {mt(dist / 1e4, 1)}\cdot 10^{{4}}$ écarts-types :" "\n"
                "hors de la figure, tout le plan est classé « hausse »",
                transform=a1.transAxes, va="top", ha="left", fontsize=7.6, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=AXIS, lw=0.6))
    fig.tight_layout(rect=(0, 0.11 if lines else 0.07, 1, 1))
    svm_legend(fig, lines, loc="lower center", ncol=3 if lines else 4, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    save(fig, f"fig8_svm_{tag}")

# Points en dehors de la fenêtre des figures 8 (valeurs extrêmes)
inside = lambda Fm: (Fm[:, 0] >= xl[0]) & (Fm[:, 0] <= xl[1]) & (Fm[:, 1] >= yl[0]) & (Fm[:, 1] <= yl[1])
out_tr_dates = pd.to_datetime(d_obs[tr][~inside(Ftr)])
out_te_dates = pd.to_datetime(d_obs[te][~inside(Fte)])
res["svm_plot_note"] = dict(
    n_outside_tr=int(len(out_tr_dates)), n_outside_te=int(len(out_te_dates)),
    n_outside_tr_mar2020=int(((out_tr_dates.year == 2020) & (out_tr_dates.month == 3)).sum()),
    outside_te_dates=out_te_dates.strftime("%d/%m/%Y").tolist(),
)

# Essai complémentaire : classifieurs non linéaires sur les mêmes variables standardisées
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def bal_acc(zv, pv):
    return 0.5 * (np.mean(pv[zv == 1] == 1) + np.mean(pv[zv == -1] == -1))


res["nonlin"] = {}
for name, mdl in [("SVM noyau RBF", SVC(kernel="rbf", C=1)),
                  ("SVM noyau RBF, poids équilibrés", SVC(kernel="rbf", C=1, class_weight="balanced")),
                  ("51 plus proches voisins", KNeighborsClassifier(51)),
                  ("Régression logistique", LogisticRegression())]:
    p_te = mdl.fit(Ftr, ztr).predict(Fte)
    res["nonlin"][name] = dict(acc_te=float(np.mean(p_te == zte)), bacc_te=float(bal_acc(zte, p_te)),
                               pred_up_te=float(np.mean(p_te == 1)))


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


(BASE / "resultats.json").write_text(json.dumps(to_py(res), indent=2, ensure_ascii=False), encoding="utf-8")
print("beta (équation normale) :", beta_ne)
print("R² train / test :", res["metrics"]["model_tr"]["r2"], res["metrics"]["model_te"]["r2"])
print("SVM C=1 : w =", r_svm["w"], "b =", r_svm["b"], "| certificat (0,1) :", res["svm_certificate"]["feasible"])
print("Résultats écrits dans resultats.json, figures dans", FIG)
