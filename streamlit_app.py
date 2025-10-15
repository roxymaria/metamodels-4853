

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

COL_TRAIN = "#ff8400"   # dark orange
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

    elif design == "Monte Carlo":
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

def latex_left(expr: str):
    # Left-align the *next* KaTeX blocks
    st.markdown("<style>.katex-display{ text-align:left !important; }</style>", unsafe_allow_html=True)
    st.latex(expr)

def latex_center(expr: str):
    # Center-align the *next* KaTeX blocks
    st.markdown("<style>.katex-display{ text-align:center !important; }</style>", unsafe_allow_html=True)
    st.latex(expr)

_NOTE_NAMES = np.array(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])

def _freq_to_midi(f: float) -> int:
    # 12-TET mapping; A4=440 Hz. Round to nearest MIDI note.
    return int(np.round(69 + 12*np.log2(float(f)/440.0)))

def _midi_to_name(m: int) -> str:
    name = _NOTE_NAMES[m % 12]
    octave = m // 12 - 1
    return f"{name}{octave}"

def _note_diff_at_max_error(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return label like 'A3→C4 (+3 st)' at the index of MaxAE."""
    if y_pred.size == 0 or y_true.size == 0:
        return ""
    err = np.abs(y_true - y_pred)
    i = int(np.argmax(err))
    ft = float(y_true[i]); fp = float(y_pred[i])
    # guard against nonpositive frequencies
    if ft <= 0 or fp <= 0:
        return ""
    mt = _freq_to_midi(ft); mp = _freq_to_midi(fp)
    dt = mp - mt
    sign = "+" if dt >= 0 else ""
    return f"{_midi_to_name(mt)}→{_midi_to_name(mp)}"

# --- Musical keyboard helpers (drop-in) ------------------------------------
_NOTE_NAMES = np.array(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"])
_WHITES = {0, 2, 4, 5, 7, 9, 11}
_BLACKS = {1, 3, 6, 8, 10}

def _freq_to_midi_float(f: float) -> float:
    return 69.0 + 12.0*np.log2(float(f)/440.0)

def _freq_to_midi_rounded(f: float) -> int:
    return int(np.round(_freq_to_midi_float(f)))

def _is_white(pc: int) -> bool: return pc in _WHITES
def _is_black(pc: int) -> bool: return pc in _BLACKS

def _keyboard_window(m_true: int, m_pred: int, *, octaves: int = 3) -> tuple[int,int]:
    """
    Pick a 3-octave [start,end] (inclusive) that includes both m_true and m_pred.
    Snap the start to a C (multiple of 12). Keeps the window stable in size.
    """
    span = 12 * octaves         # number of semitones
    half = span // 2            # 18 for 3 octaves
    m_min, m_max = sorted((m_true, m_pred))

    # Try centering the window on the midpoint, then snap start to C.
    m_mid = int(np.round((m_min + m_max) / 2))
    start_guess = m_mid - half
    start = (start_guess // 12) * 12   # snap down to C
    end = start + span - 1

    # If either note falls out, adjust to include both (still snapped to C).
    if m_min < start or m_max > end:
        # Ensure top fits
        start = int(np.ceil((m_max - (span - 1)) / 12.0)) * 12
        end = start + span - 1
        # If bottom still out, slide down to include it
        if m_min < start:
            start = (m_min // 12) * 12
            end = start + span - 1

    return start, end


## keyboard helpers
def _freqs_to_midi_rounded(f_arr):
    f = np.asarray(f_arr, dtype=float)
    m = 69.0 + 12.0*np.log2(np.maximum(f, 1e-12)/440.0)
    return np.round(m).astype(int)

def pick_maxae_in_keyboard_range(y_true, y_pred, *, note_band=NOTE_BAND, octaves=3):
    """
    Return (idx, m_true, m_pred) for the test point that has the largest |error|
    among samples whose **true note** lies inside the fixed 3-octave organ range.
    If none are inside, returns (None, None, None).
    """
    if y_true.size == 0 or y_pred.size == 0:
        return None, None, None
    start, end = _fixed_keyboard_window(note_band, octaves)  # from your keyboard helper
    m_true_all = _freqs_to_midi_rounded(y_true)
    m_pred_all = _freqs_to_midi_rounded(y_pred)
    mask = (m_true_all >= start) & (m_true_all <= end)  # only notes on the organ
    if not np.any(mask):
        return None, None, None
    err = np.abs(y_true - y_pred)
    idx_rel = int(np.argmax(err[mask]))
    idx = int(np.flatnonzero(mask)[idx_rel])
    return idx, int(m_true_all[idx]), int(m_pred_all[idx])


def _fixed_keyboard_window(note_band=(55.0, 392.0), octaves: int = 3) -> tuple[int, int]:
    """Start at the C at/below the low band edge; span exactly `octaves`."""
    mmin = _freq_to_midi_float(note_band[0])
    start = int(np.floor(mmin / 12.0) * 12)        # snap down to C
    end = start + 12 * octaves - 1
    return start, end

def render_keyboard(m_true: int, m_pred: int, *,
                    note_band=(55.0, 392.0), octaves: int = 3, fit_window: bool = False):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon

    # Pick the window: fixed (with arrows for OOR) or auto-fit (always includes both)
    if fit_window:
        start, end = _keyboard_window(m_true, m_pred, octaves=octaves)
    else:
        start, end = _fixed_keyboard_window(note_band, octaves=octaves)

    # Map x for whites
    white_x, white_idx = {}, 0
    for m in range(start, end + 1):
        if _is_white(m % 12):
            white_x[m] = white_idx
            white_idx += 1
    total_white = white_idx

    # Figure
    fig_w = max(6.0, total_white * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, 1.6))
    ax.set_xlim(0, total_white)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # 1) whites
    for m in range(start, end + 1):
        if _is_white(m % 12):
            x = white_x[m]
            ax.add_patch(Rectangle((x, 0.0), 1.0, 1.0, facecolor="white",
                                   edgecolor="#444", linewidth=1.0, zorder=1))

    # 2) white highlights
    def _highlight_white(midi: int, color: str, alpha: float = 0.45):
        if midi is None: return
        if start <= midi <= end and _is_white(midi % 12):
            x = white_x[midi]
            ax.add_patch(Rectangle((x, 0.0), 1.0, 1.0,
                                   facecolor=color, edgecolor=None, alpha=alpha, zorder=2))
    _highlight_white(m_true, "#4dff00" )
    _highlight_white(m_pred, "#ff0000" )

    # Black center on the boundary line between whites (your preferred look)
    def _black_center(midi: int) -> float:
        left_white = midi - 1
        return white_x[left_white] + 1.0  # center on the divider line

    # 3) blacks
    BK_W, BK_H, BK_Y = 0.62, 0.64, 0.36
    for m in range(start, end + 1):
        if _is_black(m % 12):
            xc = _black_center(m)
            x = xc - BK_W / 2
            ax.add_patch(Rectangle((x, BK_Y), BK_W, BK_H,
                                   facecolor="black", edgecolor="#222", linewidth=1.0, zorder=3))

    # 4) black highlights
    def _highlight_black(midi: int, color: str, alpha: float = 0.8):
        if midi is None: return
        if start <= midi <= end and _is_black(midi % 12):
            xc = _black_center(midi)
            x = xc - BK_W / 2
            ax.add_patch(Rectangle((x, BK_Y), BK_W, BK_H,
                                   facecolor=color, edgecolor=None, alpha=alpha, zorder=4))
    _highlight_black(m_true, "tab:blue")
    _highlight_black(m_pred, "tab:red")

    # 5) out-of-range arrows at edges (blue=true, red=pred), stacked if both
    def _edge_arrow(side: str, color: str, ymid: float):
        w, h = 0.45, 0.40
        if side == "left":
            x_tip = 0.12
            verts = [(x_tip, ymid), (x_tip + w, ymid + h/2), (x_tip + w, ymid - h/2)]
        else:  # right
            x_tip = total_white - 0.12
            verts = [(x_tip, ymid), (x_tip - w, ymid + h/2), (x_tip - w, ymid - h/2)]
        ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor="none", alpha=0.85, zorder=5))

    if m_true < start: _edge_arrow("left",  "tab:blue", 0.72)
    if m_pred < start: _edge_arrow("left",  "tab:red",  0.28)
    if m_true > end:   _edge_arrow("right", "tab:blue", 0.72)
    if m_pred > end:   _edge_arrow("right", "tab:red",  0.28)

    # optional outer frame
    ax.add_patch(Rectangle((0, 0), total_white, 1.0, facecolor="none", edgecolor="#ddd", linewidth=2.0, zorder=0))

    fig.tight_layout(pad=0)
    return fig




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
        ax.scatter(Lp[~mask_train], Dp[~mask_train], s=60, edgecolors=COL_EDGE,
                   facecolors=COL_TEST, label="Test", zorder=3)
        ax.scatter(Lp[ mask_train], Dp[ mask_train], s=60, edgecolors=COL_EDGE,
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

    # Predicted note contours 
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
    if name == "Radial Basis Function (RBF)":
        return C(1.0, (1e-3, 1e3)) * RBF(length_scale=(0.1, 0.05), length_scale_bounds=(1e-3, 1e3))
    elif name == "Rational Quadratic (RQ)":
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
        return {"MAE": np.nan, "R²": np.nan, "MaxAE": np.nan, "Max Note Δ": ""}
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R²":  float(r2_score(y_true, y_pred)),
        "MaxAE": float(np.max(np.abs(y_true - y_pred))),
        # "Max Note Δ": _note_diff_at_max_error(y_true, y_pred),
    }


# ---------------------------------- UI -------------------------------------
st.title("Metamodel of a Stalactite Organ")

left, right = st.columns([1.1, 1.6], gap="large")

# -------------------- Problem definition (left; only card) -----------------
with left:
    st.subheader("Problem Definition")
    card = st.container(border=True)
    with card:
        
        img_path = next((p for p in ("problem.png", "assets/problem.png") if os.path.exists(p)), None)
        if img_path:
            st.markdown(img_tag(img_path, alt="Problem"), unsafe_allow_html=True)
            st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
- Build a **metamodel** mapping stalactite geometry to **resonant frequency**.
- Inputs: Length **L (m)** and base diameter **d (cm)**.
- Choose a **DOE**, then run initial fits for Linear, Polynomial, and GPR **metamodels**.
- Try: different DoEs & Training/Testing Splits. Can you minimize the number of experiments?
- Try: different polynomial equation forms
- Try: changing GPR kernels "influence rules"
- Try: changing GPR noise level (too small -> GPR becomes unstable; too large -> GPR loses signal)
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
        "Show Dataset Model",
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
                "Type", ["Monte Carlo", "LHS", "Halton", "Grid"],
                index=(["Monte Carlo", "LHS","Halton","Grid"].index(dd["design"]) if dd else 0)
            )
        with c2:
            n = st.slider("n", N_MIN, N_MAX,
                          value=(int(dd["n_req"]) if dd else 30),
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
    st.stop()   # <— prevents Ltr/Dtr/ytr NameError and hides model UIs

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
        lin_overlay_truth = st.checkbox("Overlay Dataset Note Frequencies", value=False, key="lin_overlay_truth")
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

    with cols[1]:
        idx, m_true, m_pred = pick_maxae_in_keyboard_range(yte, yte_hat, note_band=NOTE_BAND, octaves=3)
        if idx is not None:
            st.markdown("**Max Error Within 3-Octave Organ**")
            fig_kb = render_keyboard(m_true, m_pred, note_band=NOTE_BAND, octaves=3, fit_window=False)
            st.pyplot(fig_kb, use_container_width=True)
            plt.close(fig_kb)

    # Final linear equation — centered across full width
    st.markdown("---")
    st.markdown("**Least Squares Regression Metamodel (Linear):**")
    latex_center(latex_linear_numeric(reg_lin))

    st.divider()

    # ===== Polynomial (LSR) ================================================
    st.markdown("### Polynomial Regression")
    cols = st.columns([1.2, 0.9, 0.9])

    # Right panel: form choice + overlay + metrics
    with cols[2]:
        st.markdown("**Options**")
        poly_overlay_truth = st.checkbox("Overlay Dataset Note Frequencies", value=False, key="poly_overlay_truth")
        st.markdown("**Hyperparameters**")
        poly_choice = st.selectbox("Form", list(POLY_FORMS.keys()), index=2)  # default C
        form_key = POLY_FORMS[poly_choice]["key"]

        st.markdown("**Error metrics**")
        tmp_reg, _ = fit_poly(Ltr, Dtr, ytr, form_key)
        yte_hat_tmp = predict_poly(tmp_reg, Lte, Dte, form_key) if len(Lte) else np.array([])
        st.table(pd.DataFrame([metrics_table(yte, yte_hat_tmp)]))

    # ABCD formulas — inline math, indented within the right pane
    spacer, eqcol = st.columns([1.2, 1.8])
    with eqcol:
        PAD = 0.35
        gutter, body = st.columns([PAD, 1 - PAD])
        with body:
            st.markdown("---")
            st.markdown("**Polynomial Models:**")
            st.markdown(r"**A:** $\displaystyle \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{12}Ld$")
            st.markdown(r"**B:** $\displaystyle \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{11}L^2+\beta_{22}d^2$")
            st.markdown(r"**C:** $\displaystyle \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{11}L^2+\beta_{22}d^2+\beta_{12}Ld$")
            st.markdown(r"**D:** $\displaystyle \hat f(L,d)=\beta_0+\beta_1L+\beta_2d+\beta_{11}L^2+\beta_{22}d^2+\beta_{12}Ld+\beta_{111}L^3+\beta_{222}d^3$")

    # Fit + plots with selected form and overlay
    reg_poly, labels = fit_poly(Ltr, Dtr, ytr, form_key)
    yte_hat_poly = predict_poly(reg_poly, Lte, Dte, form_key) if len(Lte) else np.array([])  # <-- add this
    grid_hat = predict_poly(reg_poly, gridL, gridD, form_key).reshape(Fg.shape)

    figM, figE = model_and_error_figs(Lg, Dg, Fg, grid_hat,
                                    overlay_truth_notes=poly_overlay_truth,
                                    title_prefix="Polynomial")
    with cols[0]: st.pyplot(figM, use_container_width=True)
    with cols[1]: st.pyplot(figE, use_container_width=True)

    with cols[1]:
        idx, m_true, m_pred = pick_maxae_in_keyboard_range(yte, yte_hat_poly, note_band=NOTE_BAND, octaves=3)
        if idx is not None:
            st.markdown("**Max Error Within 3-Octave Organ**")
            fig_kb = render_keyboard(m_true, m_pred, note_band=NOTE_BAND, octaves=3, fit_window=False)
            st.pyplot(fig_kb, use_container_width=True)
            plt.close(fig_kb)


    # Final polynomial equation — centered across full width
    st.markdown("---")
    st.markdown("**Least Squares Regression Metamodel (Polynomial):**")
    latex_center(latex_poly_numeric(reg_poly, labels))

    st.divider()




    # ===== GPR =============================================================
    st.markdown("### Gaussian Process Regression")
    cols = st.columns([1.2, 0.9, 0.9])

    with cols[2]:
        st.markdown("**Options**")
        gpr_overlay_truth = st.checkbox("Overlay Dataset Note Frequencies", value=False, key="gpr_overlay_truth")

        st.markdown("**Hyperparameters**")
        kernel_name = st.selectbox("Kernel", ["Radial Basis Function (RBF)", "Rational Quadratic (RQ)"],
                                help="RBF: smooth; RQ: multi-scale smoothness")
        alpha_choice = st.select_slider("Noise",
                        options=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e1, 1e2],
                        value=1e-1, format_func=lambda v: f'{v:.0e}')
        st.markdown("---")
        st.markdown(r"RBF: $k(x,x')=\sigma^2\exp\!\left(-\tfrac12\sum_i \frac{(x_i-x'_i)^2}{\ell_i^2}\right)$")
        st.markdown(r"RQ:  $k(x,x')=\sigma^2\left(1+\frac{\|x-x'\|^2}{2\alpha\ell^2}\right)^{-\alpha}$")
        st.markdown("RBF: smooth; RQ: multi-scale smoothness")

        # quick metrics with temp fit
        tmp_gpr = fit_gpr(Ltr, Dtr, ytr, kernel_name, alpha_choice)
        yte_hat_tmp = predict_gpr(tmp_gpr, Lte, Dte) if len(Lte) else np.array([])
        st.markdown("**Error metrics**")
        st.table(pd.DataFrame([metrics_table(yte, yte_hat_tmp)]))

    # train final model + predictions
    gpr = fit_gpr(Ltr, Dtr, ytr, kernel_name, alpha_choice)
    yte_hat_gpr = predict_gpr(gpr, Lte, Dte) if len(Lte) else np.array([])   # <-- add this line
    grid_hat = predict_gpr(gpr, gridL, gridD).reshape(Fg.shape)

    figM, figE = model_and_error_figs(Lg, Dg, Fg, grid_hat,
                                    overlay_truth_notes=gpr_overlay_truth,
                                    title_prefix="GPR")
    with cols[0]: st.pyplot(figM, use_container_width=True)
    with cols[1]: st.pyplot(figE, use_container_width=True)

    # Keyboard under error plot (GPR)
    with cols[1]:
        idx, m_true, m_pred = pick_maxae_in_keyboard_range(yte, yte_hat_gpr, note_band=NOTE_BAND, octaves=3)
        if idx is not None:
            st.markdown("**Max Error Within 3-Octave Organ**")
            fig_kb = render_keyboard(m_true, m_pred, note_band=NOTE_BAND, octaves=3, fit_window=False)
            st.pyplot(fig_kb, use_container_width=True)
            plt.close(fig_kb)




