# # app.py
# # Streamlit DOE + Metamodel app (stalactite demo) — with background toggle,
# # LHS bins & Grid lines overlays, and problem image.

# import os
# import io
# import base64
# import math
# import numpy as np
# import pandas as pd
# import streamlit as st
# import matplotlib.pyplot as plt
# from scipy.stats import qmc
# from sklearn import linear_model, gaussian_process
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# from sklearn.metrics import mean_squared_error, r2_score

# st.set_page_config(page_title="Metamodel of Problem", layout="wide")

# TRUTH_FILE = "grid100.csv"  # columns: L, d, f_true (no header)
# NOTE_BAND = (55.0, 392.0)   # A1..G4

# N_MIN, N_MAX, N_STEP = 15, 200, 5
# TRAIN_MIN, TRAIN_MAX, TRAIN_STEP = 0.50, 0.80, 0.05

# # ------------------------------- helpers ----------------------------------

# import base64
# import os

# def img_tag(path: str, alt: str = "") -> str:
#     with open(path, "rb") as f:
#         b64 = base64.b64encode(f.read()).decode("ascii")
#     # responsive, rounded, full-width inside its container
#     return (
#         f'<img alt="{alt}" loading="lazy" '
#         f'style="width:100%;height:auto;border-radius:8px;display:block;" '
#         f'src="data:image/png;base64,{b64}"/>'
#     )


# @st.cache_data(show_spinner=False)
# def load_truth_grid(path: str):
#     arr = np.loadtxt(path, delimiter=",")
#     Lc, Dc, Fc = arr[:, 0], arr[:, 1], arr[:, 2]
#     Lg = np.unique(Lc)                         # NL
#     Dg = np.unique(Dc)                         # ND
#     F = Fc.reshape(Dg.size, Lg.size, order="F")
#     return Lg, Dg, F

# def bilinear_interp(Ls, Ds, Lg, Dg, F):
#     Ls = np.asarray(Ls); Ds = np.asarray(Ds)
#     NL, ND = Lg.size, Dg.size
#     Ls = np.clip(Ls, Lg[0], Lg[-1]); Ds = np.clip(Ds, Dg[0], Dg[-1])
#     iL = np.clip(np.searchsorted(Lg, Ls, side="right") - 1, 0, NL-2)
#     iD = np.clip(np.searchsorted(Dg, Ds, side="right") - 1, 0, ND-2)
#     L0, L1 = Lg[iL], Lg[iL+1]; D0, D1 = Dg[iD], Dg[iD+1]
#     t = (Ls - L0) / (L1 - L0 + 1e-12); u = (Ds - D0) / (D1 - D0 + 1e-12)
#     f00 = F[iD,   iL]; f10 = F[iD,   iL+1]
#     f01 = F[iD+1, iL]; f11 = F[iD+1, iL+1]
#     return (1-t)*(1-u)*f00 + t*(1-u)*f10 + (1-t)*u*f01 + t*u*f11

# def sample_design(design, n, Lmin, Lmax, Dmin, Dmax, seed=42):
#     """Return (L, D, n_eff, k). For 'Grid' we place points exactly on
#     the k×k intersections so they align with plotted grid lines."""
#     rng = np.random.default_rng(seed)

#     if design == "LHS":
#         U = qmc.LatinHypercube(d=2, seed=seed).random(n)
#         k = None
#         n_eff = n
#         L = Lmin + U[:, 0] * (Lmax - Lmin)
#         D = Dmin + U[:, 1] * (Dmax - Dmin)

#     elif design == "Halton":
#         U = qmc.Halton(d=2, scramble=True, seed=seed).random(n)
#         k = None
#         n_eff = n
#         L = Lmin + U[:, 0] * (Lmax - Lmin)
#         D = Dmin + U[:, 1] * (Dmax - Dmin)

#     elif design == "MC":
#         U = rng.random((n, 2))
#         k = None
#         n_eff = n
#         L = Lmin + U[:, 0] * (Lmax - Lmin)
#         D = Dmin + U[:, 1] * (Dmax - Dmin)

#     elif design == "Grid":
#         # choose k so k*k ≈ n, and place points exactly at the intersections
#         k = max(2, round(math.sqrt(n)))
#         L_vec = np.linspace(Lmin, Lmax, k)
#         D_vec = np.linspace(Dmin, Dmax, k)
#         LL, DD = np.meshgrid(L_vec, D_vec)
#         L = LL.ravel(order="C")
#         D = DD.ravel(order="C")
#         n_eff = L.size

#     else:
#         raise ValueError("Unknown design")

#     return L, D, n_eff, k


# def design_matrix_linear(L, D):
#     return np.c_[np.ones_like(L), L, D]

# def design_matrix_quadratic(L, D):
#     return np.c_[np.ones_like(L), L, D, L**2, D**2, L*D]

# def fit_linear(Ltr, Dtr, ytr):
#     reg = linear_model.LinearRegression().fit(design_matrix_linear(Ltr,Dtr), ytr)
#     return reg

# def fit_quadratic(Ltr, Dtr, ytr):
#     reg = linear_model.LinearRegression().fit(design_matrix_quadratic(Ltr,Dtr), ytr)
#     return reg

# def predict_linear(reg, L, D):
#     return reg.predict(design_matrix_linear(L,D))

# def predict_quadratic(reg, L, D):
#     return reg.predict(design_matrix_quadratic(L,D))

# def fit_gpr(Ltr, Dtr, ytr):
#     X = np.c_[Ltr, Dtr]
#     kernel = C(1.0,(1e-3,1e3)) * RBF([0.1,0.05], (1e-3,1e3))
#     return gaussian_process.GaussianProcessRegressor(
#         kernel=kernel, alpha=1e-6, normalize_y=True, random_state=42
#     ).fit(X, ytr)

# def predict_gpr(gpr, L, D):
#     return gpr.predict(np.c_[L, D])

# def metrics_table(y_true, y_pred):
#     return {
#         "MSE": float(mean_squared_error(y_true, y_pred)),
#         "R²": float(r2_score(y_true, y_pred)),
#         "Max|SE|": float(np.max(np.abs(y_true - y_pred)))
#     }

# def natural_note_freqs(fmin=55.0, fmax=392.0):
#     A4 = 440.0
#     names_all = np.array(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])
#     freqs, labels = [], []
#     for n in range(21,109):  # A0..C8
#         f = A4 * 2 ** ((n-69)/12)
#         name = names_all[n % 12]
#         if "#" in name: 
#             continue
#         if fmin <= f <= fmax:
#             freqs.append(f); labels.append(name)
#     return np.array(freqs), labels

# def _domain_axes(ax, Lg, Dg):
#     ax.set_xlim(Lg[0], Lg[-1]); ax.set_ylim(Dg[0], Dg[-1])
#     ax.set_xlabel("Length L (m)"); ax.set_ylabel("Base diameter d (m)")

# def plot_preview(Lg, Dg, F, pts=None, *, design=None, n_req=None, k=None,
#                  show_truth=True):
#     """Preview plot with optional truth background and DOE overlays.
#     LHS shows black bin boxes; Grid shows black dashed grid lines."""
#     fig, ax = plt.subplots()

#     # Background
#     if show_truth:
#         CS = ax.contourf(Lg, Dg, F, levels=24)
#         cbar = fig.colorbar(CS, ax=ax)
#         cbar.set_label("f_true (Hz)")
#         notes, _ = natural_note_freqs(*NOTE_BAND)
#         if notes.size > 0:
#             cn = ax.contour(Lg, Dg, F, levels=notes, colors="k", linestyles=":")
#             ax.clabel(cn, fmt=lambda v: f"{v:.0f}", fontsize=8)
#     else:
#         ax.set_facecolor("0.97")

#     # Very light dotted guide (replaces the default axis grid)
#     ax.grid(True, which="both", color="0.88", linestyle=":", linewidth=0.6)

#     Lmin, Lmax = Lg[0], Lg[-1]
#     Dmin, Dmax = Dg[0], Dg[-1]

#     # Overlays
#     if design == "Grid" and (k is not None):
#         # black dashed lines through the k×k intersections
#         xlines = np.linspace(Lmin, Lmax, k)
#         ylines = np.linspace(Dmin, Dmax, k)
#         for x in xlines:
#             ax.axvline(x, color="k", lw=1.0, ls="--", alpha=0.9, zorder=1)
#         for y in ylines:
#             ax.axhline(y, color="k", lw=1.0, ls="--", alpha=0.9, zorder=1)

#     if design == "LHS" and (n_req is not None):
#         # black bin boxes (strata). Too many lines can get heavy, so thin if needed.
#         max_lines = 60
#         step = max(1, int(np.ceil(n_req / max_lines)))
#         xedges = np.linspace(Lmin, Lmax, n_req + 1)[::step]
#         yedges = np.linspace(Dmin, Dmax, n_req + 1)[::step]
#         for x in xedges:
#             ax.axvline(x, color="k", lw=1.0, ls="-", alpha=0.9, zorder=1)
#         for y in yedges:
#             ax.axhline(y, color="k", lw=1.0, ls="-", alpha=0.9, zorder=1)

#     # DOE points
#     if pts is not None:
#         Lp, Dp, mask_train = pts
#         ax.scatter(Lp[~mask_train], Dp[~mask_train], s=34, edgecolors="k",
#                    facecolors="white", label="Test", zorder=3)
#         ax.scatter(Lp[ mask_train], Dp[ mask_train], s=34, edgecolors="k",
#                    facecolors="tab:orange", label="Train", zorder=3)
#         ax.legend(loc="best", frameon=True)

#     # Axes cosmetics
#     ax.set_xlim(Lmin, Lmax); ax.set_ylim(Dmin, Dmax)
#     ax.set_xlabel("Length(m)"); ax.set_ylabel("Base Diameter(m)")
#     ax.set_title("DOE")
#     fig.tight_layout()
#     return fig


# def plot_model_and_error(Lg, Dg, F_truth, F_model, title_model="Model", title_err="|Error|"):
#     fig1, ax1 = plt.subplots()
#     CS1 = ax1.contourf(Lg, Dg, F_model, levels=24)
#     c1 = fig1.colorbar(CS1, ax=ax1); c1.set_label("f̂ (Hz)")
#     notes, _ = natural_note_freqs(*NOTE_BAND)
#     if notes.size > 0:
#         cn = ax1.contour(Lg, Dg, F_model, levels=notes, colors="k", linestyles=":")
#         ax1.clabel(cn, fmt=lambda v: f"{v:.0f}", fontsize=8)
#     ax1.set_xlabel("Length L (m)"); ax1.set_ylabel("Base diameter d (m)")
#     ax1.set_title(title_model); fig1.tight_layout()

#     E = np.abs(F_model - F_truth)
#     fig2, ax2 = plt.subplots()
#     CS2 = ax2.contourf(Lg, Dg, E, levels=24)
#     c2 = fig2.colorbar(CS2, ax=ax2); c2.set_label("|f_hat - f_true| (Hz)")
#     ax2.set_xlabel("Length L (m)"); ax2.set_ylabel("Base diameter (m)")
#     ax2.set_title(title_err); fig2.tight_layout()
#     return fig1, fig2

# def equation_string_linear(reg):
#     b = reg.coef_; a = reg.intercept_
#     return f"f̂(L,d) = {a:.3f} + {b[1]:.3f}·L + {b[2]:.3f}·d"

# def equation_string_quadratic(reg):
#     b = reg.coef_; a = reg.intercept_
#     return ("f̂(L,d) = "
#             f"{a:.3f} + {b[1]:.3f}·L + {b[2]:.3f}·d "
#             f"+ {b[3]:.3f}·L² + {b[4]:.3f}·d² + {b[5]:.3f}·L·d")

# # ---------------------------------- UI ------------------------------------
# st.title("Metamodel of Problem")

# colA, colB, colC = st.columns([1.3, 1.3, 1.3], gap="large")

# with colA:
#     st.subheader("Problem Definition")
#     # image on top (falls back to caption if file missing)
#     img_path = next((p for p in ("problem.png", "assets/problem.png") if os.path.exists(p)), None)
#     if img_path:
#         st.markdown(img_tag(img_path, alt="Problem"), unsafe_allow_html=True)
#         st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)  # small spacer
#     else:
#         st.caption("Add `problem.png` (or place it in `assets/`).")

    
#     st.markdown(
#         """
# - You're building a metamodel that maps **stalactite geometry to resonance (notes for 3 octaves)**.  
# - Create a DOE plan for experimental testing
# - Fit **Linear**, **Quadratic**, and **GPR** metamodels. See how sample number, DoE, and training splits effect your model  
#         """
#     )




# with colB:
#     st.subheader("DOE")
#     design = st.selectbox("Select type", ["LHS", "Halton", "MC", "Grid"])
#     n = st.slider("Sample size (n)", N_MIN, N_MAX, 25, step=N_STEP)
#     train_frac = st.slider("Train proportion (%)", int(TRAIN_MIN*100), int(TRAIN_MAX*100),
#                            60, step=int(TRAIN_STEP*100)) / 100
#     show_truth_bg = st.checkbox("Show truth background", value=True)
#     run_doe = st.button("Run DOE", use_container_width=True)

# with colC:
#     st.subheader("Preview")
#     Lg, Dg, Fg = load_truth_grid(TRUTH_FILE)
#     Lmin, Lmax = float(Lg[0]), float(Lg[-1]); Dmin, Dmax = float(Dg[0]), float(Dg[-1])

# if "DOE" not in st.session_state:
#     st.session_state["DOE"] = None

# if run_doe:
#     L, D, n_eff, k = sample_design(design, n, Lmin, Lmax, Dmin, Dmax, seed=42)
#     y = bilinear_interp(L, D, Lg, Dg, Fg)
#     rng = np.random.default_rng(123)
#     idx = rng.permutation(len(L))
#     n_train = int(round(train_frac * len(L)))
#     mask_train = np.zeros(len(L), dtype=bool); mask_train[idx[:n_train]] = True
#     st.session_state["DOE"] = {
#         "design": design, "n_req": n, "n_eff": n_eff, "k": k,
#         "L": L, "D": D, "y": y, "mask_train": mask_train,
#         "show_truth_bg": show_truth_bg
#     }

# with colC:
#     dd = st.session_state["DOE"]
#     if dd is None:
#         fig = plot_preview(Lg, Dg, Fg, pts=None, design=None, show_truth=show_truth_bg)
#         st.pyplot(fig, use_container_width=True)
#     else:
#         fig = plot_preview(
#             Lg, Dg, Fg,
#             pts=(dd["L"], dd["D"], dd["mask_train"]),
#             design=dd["design"], n_req=dd["n_req"], k=dd["k"],
#             show_truth=dd.get("show_truth_bg", True)
#         )
#         st.pyplot(fig, use_container_width=True)

# st.markdown("---")
# st.subheader("Run Regression Models")
# run_models = st.button("Fit Models", type="primary")

# if run_models and st.session_state["DOE"] is None:
#     st.warning("Run DOE first.")
# elif run_models:
#     dd = st.session_state["DOE"]
#     L, D, y, mtr = dd["L"], dd["D"], dd["y"], dd["mask_train"]
#     Ltr, Dtr, ytr = L[mtr], D[mtr], y[mtr]
#     Lte, Dte, yte = L[~mtr], D[~mtr], y[~mtr]

#     LLg, DDg = np.meshgrid(Lg, Dg)
#     gridL = LLg.ravel(order="C"); gridD = DDg.ravel(order="C")

#     c1, c2, c3 = st.columns(3)

#     with c1:
#         st.markdown("**Linear Regression**")
#         if len(Ltr) < 3:
#             st.info("Not enough train points (need ≥3).")
#         else:
#             reg = fit_linear(Ltr, Dtr, ytr)
#             st.code(equation_string_linear(reg), language="text")
#             ytr_hat = predict_linear(reg, Ltr, Dtr)
#             yte_hat = predict_linear(reg, Lte, Dte) if len(Lte) else np.array([])
#             mt = pd.DataFrame([metrics_table(yte, yte_hat) if len(yte_hat) else {"MSE":np.nan,"R²":np.nan,"Max|SE|":np.nan}])
#             st.table(mt)
#             grid_hat = predict_linear(reg, gridL, gridD).reshape(Fg.shape)
#             fM, fE = plot_model_and_error(Lg, Dg, Fg, grid_hat,
#                                           "Linear — model", "Linear — |error|")
#             st.pyplot(fM, use_container_width=True); st.pyplot(fE, use_container_width=True)

#     with c2:
#         st.markdown("**Least Squares Regression (Quadratic)**")
#         if len(Ltr) < 6:
#             st.info("Not enough train points (need ≥6).")
#         else:
#             reg = fit_quadratic(Ltr, Dtr, ytr)
#             st.code(equation_string_quadratic(reg), language="text")
#             ytr_hat = predict_quadratic(reg, Ltr, Dtr)
#             yte_hat = predict_quadratic(reg, Lte, Dte) if len(Lte) else np.array([])
#             mt = pd.DataFrame([metrics_table(yte, yte_hat) if len(yte_hat) else {"MSE":np.nan,"R²":np.nan,"Max|SE|":np.nan}])
#             st.table(mt)
#             grid_hat = predict_quadratic(reg, gridL, gridD).reshape(Fg.shape)
#             fM, fE = plot_model_and_error(Lg, Dg, Fg, grid_hat,
#                                           "Quadratic — model", "Quadratic — |error|")
#             st.pyplot(fM, use_container_width=True); st.pyplot(fE, use_container_width=True)

#     with c3:
#         st.markdown("**Gaussian Process Regression**")
#         gpr = fit_gpr(Ltr, Dtr, ytr)
#         st.caption("RBF kernel, normalized y")
#         ytr_hat = predict_gpr(gpr, Ltr, Dtr)
#         yte_hat = predict_gpr(gpr, Lte, Dte) if len(Lte) else np.array([])
#         mt = pd.DataFrame([metrics_table(yte, yte_hat) if len(yte_hat) else {"MSE":np.nan,"R²":np.nan,"Max|SE|":np.nan}])
#         st.table(mt)
#         grid_hat = predict_gpr(gpr, gridL, gridD).reshape(Fg.shape)
#         fM, fE = plot_model_and_error(Lg, Dg, Fg, grid_hat,
#                                       "GPR — model", "GPR — |error|")
#         st.pyplot(fM, use_container_width=True); st.pyplot(fE, use_container_width=True)

# st.markdown("---")
# with st.expander("Copy-pastable: DOE + models (vanilla code)", expanded=False):
#     st.code(r'''
# from scipy.stats import qmc
# import numpy as np
# from sklearn import linear_model, gaussian_process
# from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# def design_LHS(n, seed=42): return qmc.LatinHypercube(2, seed=seed).random(n)
# def design_Halton(n, seed=42): return qmc.Halton(2, scramble=True, seed=seed).random(n)
# def design_MC(n, seed=42): return np.random.default_rng(seed).random((n,2))
# def design_Grid(n):
#     k = max(2, round(np.sqrt(n))); g = np.linspace(0,1,k); U,V = np.meshgrid(g,g)
#     return np.c_[U.ravel(), V.ravel()], k*k

# def X_lin(L,D): return np.c_[np.ones_like(L), L, D]
# def X_quad(L,D): return np.c_[np.ones_like(L), L, D, L**2, D**2, L*D]

# def fit_lin(L,D,y): m=linear_model.LinearRegression(); m.fit(X_lin(L,D),y); return m
# def fit_quad(L,D,y): m=linear_model.LinearRegression(); m.fit(X_quad(L,D),y); return m
# def fit_gpr(L,D,y):
#     k=C(1.0,(1e-3,1e3))*RBF([0.1,0.05],(1e-3,1e3))
#     return gaussian_process.GaussianProcessRegressor(kernel=k, alpha=1e-6, normalize_y=True).fit(np.c_[L,D], y)
# ''', language="python")
# streamlit_app.py
# Metamodel of Problem — DOE + Regression playground (stalactite demo)
# streamlit_app.py
# Metamodel DOE + models (stalactite demo)
# - DOE preview on the right, controls under the plot (single-click form)
# - LHS bin boxes & Grid dashed lines
# - Model panels appear only after "Fit Models"
# - MAE / R² / Max|SE| on TEST split
# - d axis labeled in cm
# - Problem image loaded via base64 (no PIL)

# streamlit_app.py — stalactite metamodel playground (clean layout)

import os, math, base64
import numpy as np
import pandas as pd
import streamlit as st

# Headless Matplotlib (works with st.pyplot)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from scipy.stats import qmc
from sklearn import linear_model, gaussian_process
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ConstantKernel as C
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------- Page + constants ------------------------------
st.set_page_config(page_title="Metamodel of a Stalactite Organ", layout="wide")

TRUTH_FILE = "grid100.csv"      # numeric CSV: L,d,f_true (no header)
NOTE_BAND  = (55.0, 392.0)      # A1..G4

N_MIN, N_MAX, N_STEP = 15, 200, 5
TRAIN_MIN, TRAIN_MAX, TRAIN_STEP = 0.50, 0.80, 0.05

COL_TRAIN = "#ff0000"   # dark orange
COL_TEST  = "#ffffff"   # white
COL_EDGE  = "#000000"   # black
GRID_FAINT = "#E0E0E0"

# --------------------------- Session defaults ------------------------------
ss = st.session_state
ss.setdefault("DOE", None)
ss.setdefault("models_ready", False)

# ------------------------------ Helpers ------------------------------------
def img_tag(path: str, alt: str = "") -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return (
        f'<img alt="{alt}" loading="lazy" '
        f'style="width:100%;height:auto;border-radius:8px;display:block;" '
        f'src="data:image/png;base64,{b64}"/>'
    )

@st.cache_data(show_spinner=False)
def load_truth_grid(path: str):
    arr = np.loadtxt(path, delimiter=",")
    Lc, Dc, Fc = arr[:, 0], arr[:, 1], arr[:, 2]
    Lg = np.unique(Lc)                         # NL
    Dg = np.unique(Dc)                         # ND
    F  = Fc.reshape(Dg.size, Lg.size, order="F")
    return Lg, Dg, F

def bilinear_interp(Ls, Ds, Lg, Dg, F):
    Ls = np.asarray(Ls); Ds = np.asarray(Ds)
    NL, ND = Lg.size, Dg.size
    Ls = np.clip(Ls, Lg[0], Lg[-1]); Ds = np.clip(Ds, Dg[0], Dg[-1])
    iL = np.clip(np.searchsorted(Lg, Ls, side="right") - 1, 0, NL-2)
    iD = np.clip(np.searchsorted(Dg, Ds, side="right") - 1, 0, ND-2)
    L0, L1 = Lg[iL], Lg[iL+1]; D0, D1 = Dg[iD], Dg[iD+1]
    t = (Ls - L0) / (L1 - L0 + 1e-12); u = (Ds - D0) / (D1 - D0 + 1e-12)
    f00 = F[iD,   iL]; f10 = F[iD,   iL+1]
    f01 = F[iD+1, iL]; f11 = F[iD+1, iL+1]
    return (1-t)*(1-u)*f00 + t*(1-u)*f10 + (1-t)*u*f01 + t*u*f11

def sample_design(design, n, Lmin, Lmax, Dmin, Dmax, seed=42):
    """Return (L, D, n_eff, k). Grid aligns exactly to intersections."""
    rng = np.random.default_rng(seed)

    if design == "LHS":
        U = qmc.LatinHypercube(d=2, seed=seed).random(n)
        k = None; n_eff = n
        L = Lmin + U[:, 0] * (Lmax - Lmin)
        D = Dmin + U[:, 1] * (Dmax - Dmin)

    elif design == "Halton":
        U = qmc.Halton(d=2, scramble=True, seed=seed).random(n)
        k = None; n_eff = n
        L = Lmin + U[:, 0] * (Lmax - Lmin)
        D = Dmin + U[:, 1] * (Dmax - Dmin)

    elif design == "MC":
        U = rng.random((n, 2))
        k = None; n_eff = n
        L = Lmin + U[:, 0] * (Lmax - Lmin)
        D = Dmin + U[:, 1] * (Dmax - Dmin)

    elif design == "Grid":
        k = max(2, round(math.sqrt(n)))
        L_vec = np.linspace(Lmin, Lmax, k)
        D_vec = np.linspace(Dmin, Dmax, k)
        LL, DD = np.meshgrid(L_vec, D_vec, indexing="xy")
        L = LL.ravel(order="C")
        D = DD.ravel(order="C")
        n_eff = L.size

    else:
        raise ValueError("Unknown design")

    return L, D, n_eff, k

def natural_note_freqs(fmin=55.0, fmax=392.0):
    A4 = 440.0
    names_all = np.array(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])
    freqs, labels = [], []
    for n in range(21, 109):  # MIDI A0..C8
        f = A4 * 2 ** ((n-69)/12)
        name = names_all[n % 12]
        if "#" in name:
            continue
        if fmin <= f <= fmax:
            freqs.append(f); labels.append(name)
    return np.array(freqs), labels

def format_d_cm():
    return FuncFormatter(lambda y, pos: f"{y*100:.0f}")

# ---------- Plotting --------------------------------------------------------
def plot_doe_preview(
    Lg, Dg, F, pts=None, *, design=None, n_req=None, k=None,
    show_truth=False, title="DOE", figsize=(9.5, 7.0)
):
    Lmin, Lmax = float(Lg[0]), float(Lg[-1])
    Dmin, Dmax = float(Dg[0]), float(Dg[-1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("auto")

    if show_truth:
        CS = ax.contourf(Lg, Dg, F, levels=24)
        cbar = fig.colorbar(CS, ax=ax); cbar.set_label("f_true (Hz)")
        notes, _ = natural_note_freqs(*NOTE_BAND)
        if notes.size:
            cn = ax.contour(Lg, Dg, F, levels=notes, colors="k", linestyles=":")
            ax.clabel(cn, fmt=lambda v: f"{v:.0f}", fontsize=8)
    else:
        ax.set_facecolor("#f7f7f7")

    ax.grid(True, which="both", color=GRID_FAINT, linestyle=":", linewidth=0.6, zorder=0)

    if design == "Grid" and k is not None:
        for x in np.linspace(Lmin, Lmax, int(k)):
            ax.axvline(x, color="k", lw=1.0, ls="--", alpha=0.9, zorder=1)
        for y in np.linspace(Dmin, Dmax, int(k)):
            ax.axhline(y, color="k", lw=1.0, ls="--", alpha=0.9, zorder=1)

    if design == "LHS" and n_req is not None:
        n_req = int(n_req)
        max_lines = 60
        step = max(1, int(np.ceil(n_req / max_lines)))
        for x in np.linspace(Lmin, Lmax, n_req + 1)[::step]:
            ax.axvline(x, color="k", lw=1.0, ls="-", alpha=0.9, zorder=1)
        for y in np.linspace(Dmin, Dmax, n_req + 1)[::step]:
            ax.axhline(y, color="k", lw=1.0, ls="-", alpha=0.9, zorder=1)

    if pts is not None:
        Lp, Dp, mask_train = pts
        ax.scatter(Lp[~mask_train], Dp[~mask_train], s=36, edgecolors=COL_EDGE,
                   facecolors=COL_TEST, label="Test", zorder=3)
        ax.scatter(Lp[ mask_train], Dp[ mask_train], s=36, edgecolors=COL_EDGE,
                   facecolors=COL_TRAIN, label="Train", zorder=3)
        ax.legend(loc="best", frameon=True)

    ax.set_xlim(Lmin, Lmax); ax.set_ylim(Dmin, Dmax)
    ax.set_xlabel("Length L (m)"); ax.set_ylabel("Base diameter d (cm)")
    ax.yaxis.set_major_formatter(format_d_cm())
    ax.set_title(title)
    fig.tight_layout()
    return fig

def model_and_error_figs(
    Lg, Dg, F_truth, F_model, *, overlay_truth_notes=False, title_prefix="Model"
):
    # Model surface
    fig1, ax1 = plt.subplots(figsize=(7.3, 5.3))
    CS1 = ax1.contourf(Lg, Dg, F_model, levels=24)
    fig1.colorbar(CS1, ax=ax1).set_label("f̂ (Hz)")

    # Predicted note contours (always on)
    notes, _ = natural_note_freqs(*NOTE_BAND)
    if notes.size:
        cn_pred = ax1.contour(Lg, Dg, F_model, levels=notes, colors="k", linestyles=":")
        ax1.clabel(cn_pred, fmt=lambda v: f"{v:.0f}", fontsize=8)
        if overlay_truth_notes:
            ax1.contour(Lg, Dg, F_truth, levels=notes, colors="#cc0000", linestyles="-", linewidths=1.0)

    ax1.set_xlabel("Length (m)")
    ax1.set_ylabel("Base diameter (cm)")
    ax1.yaxis.set_major_formatter(format_d_cm())
    ax1.set_title(f"{title_prefix} — model")
    fig1.tight_layout()

    # |error|
    E = np.abs(F_model - F_truth)
    fig2, ax2 = plt.subplots(figsize=(7.3, 5.3))
    CS2 = ax2.contourf(Lg, Dg, E, levels=24)
    fig2.colorbar(CS2, ax=ax2).set_label("|f_hat - f_true| (Hz)")
    ax2.set_xlabel("Length L (m)")
    ax2.set_ylabel("Base diameter d (cm)")
    ax2.yaxis.set_major_formatter(format_d_cm())
    ax2.set_title(f"{title_prefix} — |error|")
    fig2.tight_layout()
    return fig1, fig2

# ---------------- Design matrices (Linear + labeled polynomial forms) ------
def X_linear(L,D):  return np.c_[np.ones_like(L), L, D]

POLY_FORMS = {
    "A": {"key":"A_L_Ld",       "labels":["1","L","d","L d"]},
    "B": {"key":"B_Q_noLD",     "labels":["1","L","d","L^2","d^2"]},
    "C": {"key":"C_Q_withLD",   "labels":["1","L","d","L^2","d^2","L d"]},
    "D": {"key":"D_Cubic_lite", "labels":["1","L","d","L^2","d^2","L d","L^3","d^3"]},
}
POLY_FORMS_LATEX = {
    "A_L_Ld":       r"A: \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{12}Ld",
    "B_Q_noLD":     r"B: \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{11}L^2+\beta_{22}d^2",
    "C_Q_withLD":   r"C: \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{11}L^2+\beta_{22}d^2+\beta_{12}Ld",
    "D_Cubic_lite": r"D: \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{11}L^2+\beta_{22}d^2+\beta_{12}Ld+\beta_{111}L^3+\beta_{222}d^3",
}

def X_poly(L, D, form_key: str):
    L2, d2 = L**2, D**2
    L3, d3 = L**3, D**3
    if form_key == "A_L_Ld":
        terms = [np.ones_like(L), L, D, L*D]
        labels = ["1","L","d","L d"]
    elif form_key == "B_Q_noLD":
        terms = [np.ones_like(L), L, D, L2, d2]
        labels = ["1","L","d","L^2","d^2"]
    elif form_key == "C_Q_withLD":
        terms = [np.ones_like(L), L, D, L2, d2, L*D]
        labels = ["1","L","d","L^2","d^2","L d"]
    elif form_key == "D_Cubic_lite":
        terms = [np.ones_like(L), L, D, L2, d2, L*D, L3, d3]
        labels = ["1","L","d","L^2","d^2","L d","L^3","d^3"]
    else:
        raise ValueError("Unknown polynomial form")
    return np.column_stack(terms), labels

def fit_linear(Ltr, Dtr, ytr): return linear_model.LinearRegression().fit(X_linear(Ltr,Dtr), ytr)

def fit_poly(Ltr, Dtr, ytr, form_key):
    X, labels = X_poly(Ltr, Dtr, form_key)
    reg = linear_model.LinearRegression().fit(X, ytr)
    return reg, labels

def predict_linear(reg, L, D): return reg.predict(X_linear(L,D))
def predict_poly(reg, L, D, form_key): return reg.predict(X_poly(L,D,form_key)[0])

# Numeric fitted-equation printers
def latex_linear_numeric(reg):
    b = reg.coef_; a = reg.intercept_
    return r"\hat f(L,d) = " + f"{a:.3f} + {b[1]:.3f}L + {b[2]:.3f}d"

def latex_poly_numeric(reg, labels):
    pieces = [f"{reg.intercept_:.3f}"]
    for lab, c in zip(labels[1:], reg.coef_[1:]):  # skip '1'
        if abs(c) < 1e-12:  # keep it clean
            continue
        lab_tex = lab.replace(" ", r"\,")
        pieces.append(f"{c:.3f}{lab_tex}")
    return r"\hat f(L,d) = " + " + ".join(pieces)

# ----------------------------- GPR helpers ---------------------------------
def build_kernel(name: str):
    if name == "RBF":
        return C(1.0, (1e-3, 1e3)) * RBF(length_scale=(0.1, 0.05), length_scale_bounds=(1e-3, 1e3))
    elif name == "RationalQuadratic":
        return C(1.0, (1e-3, 1e3)) * RationalQuadratic(length_scale=0.1, alpha=1.0,
                                                       length_scale_bounds=(1e-3, 1e3))
    else:
        raise ValueError("Unknown kernel")

def fit_gpr(Ltr, Dtr, ytr, kernel_name, alpha):
    X = np.c_[Ltr, Dtr]
    kernel = build_kernel(kernel_name)
    return gaussian_process.GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True, random_state=42
    ).fit(X, ytr)

def predict_gpr(gpr, L, D): return gpr.predict(np.c_[L, D])

def metrics_table(y_true, y_pred):
    if y_pred.size == 0:
        return {"MAE": np.nan, "R²": np.nan, "MaxAE": np.nan}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R²":  float(r2_score(y_true, y_pred)),
        "MaxAE": float(np.max(np.abs(y_true - y_pred)))
    }

# ---------------------------------- UI -------------------------------------
st.title("Metamodel of Problem")

left, right = st.columns([1.1, 1.6], gap="large")

# -------------------- Problem definition (left; only card) -----------------
with left:
    card = st.container(border=True)
    with card:
        st.subheader("Problem Definition")
        img_path = next((p for p in ("problem.png", "assets/problem.png") if os.path.exists(p)), None)
        if img_path:
            st.markdown(img_tag(img_path, alt="Problem"), unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
- Build a **metamodel** mapping stalactite geometry to **resonant frequency**.
- Inputs: **Length L (m)** and **base diameter d (m)**.
- Choose a **DOE**, then fit **Linear**, **Polynomial**, and **GPR** models.
            """
        )

# --------------------------- DOE (right) -----------------------------------
with right:
    st.subheader("DOE")

    # Load truth grid (once per run)
    Lg, Dg, Fg = load_truth_grid(TRUTH_FILE)
    Lmin, Lmax = float(Lg[0]), float(Lg[-1])
    Dmin, Dmax = float(Dg[0]), float(Dg[-1])

    dd = st.session_state["DOE"]

    # ---- Live toggle (dynamic) --------------------------------------------
    # This is the ONLY control that updates the preview without pressing Run DOE.
    live_truth_bg = st.checkbox(
        "Show Theoretical Model",
        value=(dd.get("show_truth_bg", False) if dd else False),
        key="doe_truth_bg_live",
        help="Toggle the ground-truth surface under the DOE preview."
    )

    # ---- Preview uses saved DOE + live toggle -----------------------------
    fig_prev = plot_doe_preview(
        Lg, Dg, Fg,
        pts=(dd["L"], dd["D"], dd["mask_train"]) if dd else None,
        design=(dd["design"] if dd else None),
        n_req=(dd["n_req"] if dd else None),
        k=(dd["k"] if dd else None),
        show_truth=live_truth_bg,                 # <-- live update here
        title="DOE", figsize=(9.5, 7.0),
    )
    st.pyplot(fig_prev, use_container_width=True)
    plt.close(fig_prev)

    # ---- DOE controls (buffered; apply on submit) -------------------------
    with st.form("doe_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1.0, 1.0, 1.0])

        with c1:
            design = st.selectbox(
                "Type", ["LHS", "Halton", "MC", "Grid"],
                index=(["LHS","Halton","MC","Grid"].index(dd["design"]) if dd else 0)
            )
        with c2:
            n = st.slider("n", N_MIN, N_MAX,
                          value=(int(dd["n_req"]) if dd else 25),
                          step=N_STEP)
        with c3:
            default_train_pct = int(round(100*np.mean(dd["mask_train"]))) if dd else 60
            train_pct = st.slider("Train %",
                                  int(TRAIN_MIN*100), int(TRAIN_MAX*100),
                                  value=default_train_pct,
                                  step=int(TRAIN_STEP*100))

        submit = st.form_submit_button("Run DOE", type="primary", use_container_width=True)

    # ---- Apply changes only on submit -------------------------------------
    if submit:
        train_frac = train_pct / 100.0
        L, D, n_eff, k = sample_design(design, n, Lmin, Lmax, Dmin, Dmax, seed=42)
        y = bilinear_interp(L, D, Lg, Dg, Fg)

        rng = np.random.default_rng(123)
        idx = rng.permutation(len(L))
        n_train = int(round(train_frac * len(L)))
        mask_train = np.zeros(len(L), dtype=bool); mask_train[idx[:n_train]] = True

        st.session_state["DOE"] = {
            "design": design, "n_req": n, "n_eff": n_eff, "k": k,
            "L": L, "D": D, "y": y, "mask_train": mask_train,
            # persist whatever the live toggle is showing at submit time
            "show_truth_bg": bool(live_truth_bg)
        }
        st.session_state["models_ready"] = False
        st.rerun()


# ------------------------- Models entry point ------------------------------
st.divider()
st.subheader("Regression Models")
btn_models = st.button("Fit Initial Models", type="primary")
if btn_models and ss.get("DOE") is None:
    st.warning("Run DOE first.")
elif btn_models:
    ss["models_ready"] = True

if not ss.get("models_ready", False):
    st.info("Results appear here after running initial models.")
else:
    # ---- Gather split & grid
    dd = ss["DOE"]
    L, D, y, mtr = dd["L"], dd["D"], dd["y"], dd["mask_train"]
    Ltr, Dtr, ytr = L[mtr], D[mtr], y[mtr]
    Lte, Dte, yte = L[~mtr], D[~mtr], y[~mtr]
    Lg, Dg, Fg = load_truth_grid(TRUTH_FILE)
    LLg, DDg = np.meshgrid(Lg, Dg, indexing="xy")
    gridL = LLg.ravel(order="C"); gridD = DDg.ravel(order="C")

    # ===== Linear ===========================================================
    st.markdown("### Linear Regression")
    cols = st.columns([1.2, 0.9, 0.9])   # [model fig, error fig, right panel]

    # Right panel: HPs + metrics + overlay toggle (auto updates)
    with cols[2]:
        st.markdown("**Options**")
        lin_overlay_truth = st.checkbox("Overlay ground-truth note lines (red)", value=False, key="lin_overlay_truth")
        st.markdown("**Error metrics (test)**")
        yte_hat_tmp = predict_linear(fit_linear(Ltr, Dtr, ytr), Lte, Dte) if len(Lte) else np.array([])
        st.table(pd.DataFrame([metrics_table(yte, yte_hat_tmp)]))

    # Fit + plots with current overlay
    reg_lin = fit_linear(Ltr, Dtr, ytr)
    yte_hat = predict_linear(reg_lin, Lte, Dte) if len(Lte) else np.array([])
    grid_hat = predict_linear(reg_lin, gridL, gridD).reshape(Fg.shape)
    figM, figE = model_and_error_figs(Lg, Dg, Fg, grid_hat,
                                      overlay_truth_notes=lin_overlay_truth,
                                      title_prefix="Linear")
    with cols[0]: st.pyplot(figM, use_container_width=True)
    with cols[1]: st.pyplot(figE, use_container_width=True)

    # Equation row: span middle + right columns
    spacer, eqcol = st.columns([1.2, 1.8])
    with eqcol:
        st.markdown("**Equation (fitted)**")
        st.latex(latex_linear_numeric(reg_lin))

    st.divider()

    # ===== Polynomial (LSR) ================================================
    st.markdown("### Polynomial Regression")
    cols = st.columns([1.2, 0.9, 0.9])

    # Right panel: form choice + overlay + metrics + static list of forms
    with cols[2]:
        st.markdown("**Hyperparameters**")
        poly_choice = st.selectbox("Form", list(POLY_FORMS.keys()), index=2)  # default C
        form_key = POLY_FORMS[poly_choice]["key"]
        poly_overlay_truth = st.checkbox("Overlay theoretical data", value=False, key="poly_overlay_truth")
        st.markdown("**Error metrics**")
        # quick metrics using temp fit on current form
        tmp_reg, _ = fit_poly(Ltr, Dtr, ytr, form_key)
        yte_hat_tmp = predict_poly(tmp_reg, Lte, Dte, form_key) if len(Lte) else np.array([])
        st.table(pd.DataFrame([metrics_table(yte, yte_hat_tmp)]))

        st.markdown("---")
        st.markdown("**Polynomial Models:**")
        st.latex(POLY_FORMS_LATEX["A_L_Ld"])
        st.latex(POLY_FORMS_LATEX["B_Q_noLD"])
        st.latex(POLY_FORMS_LATEX["C_Q_withLD"])
        st.latex(POLY_FORMS_LATEX["D_Cubic_lite"])

    # Fit + plots with selected form and overlay
    reg_poly, labels = fit_poly(Ltr, Dtr, ytr, form_key)
    grid_hat = predict_poly(reg_poly, gridL, gridD, form_key).reshape(Fg.shape)
    figM, figE = model_and_error_figs(Lg, Dg, Fg, grid_hat,
                                      overlay_truth_notes=poly_overlay_truth,
                                      title_prefix="Polynomial")
    with cols[0]: st.pyplot(figM, use_container_width=True)
    with cols[1]: st.pyplot(figE, use_container_width=True)

    spacer, eqcol = st.columns([1.2, 1.8])
    with eqcol:
        st.markdown("**Equation**")
        st.latex(latex_poly_numeric(reg_poly, labels))

    st.divider()

    # ===== GPR =============================================================
    st.markdown("### Gaussian Process Regression")
    cols = st.columns([1.2, 0.9, 0.9])

    with cols[2]:
        st.markdown("**Hyperparameters**")
        kernel_name = st.selectbox("Kernel", ["RBF", "RationalQuadratic"],
                                   help="RBF: smooth; RQ: multi-scale smoothness")
        alpha_choice = st.select_slider("Noise",
                          options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                          value=1e-6, format_func=lambda v: f'{v:.0e}')
        gpr_overlay_truth = st.checkbox("Overlay theoretical data", value=False, key="gpr_overlay_truth")
        st.markdown(r"RBF: $k(x,x')=\sigma^2\exp\!\left(-\tfrac12\sum_i \frac{(x_i-x'_i)^2}{\ell_i^2}\right)$")
        st.markdown(r"RQ:  $k(x,x')=\sigma^2\left(1+\frac{\|x-x'\|^2}{2\alpha\ell^2}\right)^{-\alpha}$")

        # quick metrics with temp fit
        tmp_gpr = fit_gpr(Ltr, Dtr, ytr, kernel_name, alpha_choice)
        yte_hat_tmp = predict_gpr(tmp_gpr, Lte, Dte) if len(Lte) else np.array([])
        st.markdown("**Error metrics**")
        st.table(pd.DataFrame([metrics_table(yte, yte_hat_tmp)]))

    gpr = fit_gpr(Ltr, Dtr, ytr, kernel_name, alpha_choice)
    grid_hat = predict_gpr(gpr, gridL, gridD).reshape(Fg.shape)
    figM, figE = model_and_error_figs(Lg, Dg, Fg, grid_hat,
                                      overlay_truth_notes=gpr_overlay_truth,
                                      title_prefix="GPR")
    with cols[0]: st.pyplot(figM, use_container_width=True)
    with cols[1]: st.pyplot(figE, use_container_width=True)

    spacer, eqcol = st.columns([1.2, 1.8])
    with eqcol:
        st.markdown("**Predictive mean**")
        st.latex(r"\hat f(x) = k(x,X)\,[K(X,X)+\alpha I]^{-1}y")

# ----------------------- Sample-code placeholders --------------------------
st.divider()
c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("DOE — sample code")
    st.code("# sample code", language="python")
with c2:
    st.subheader("Models — sample code")
    st.code("# sample code", language="python")
