# app.py
# ===============================================================
# ⚽ SCOUTING TOOL — Streamlit (ML, fit por grupos, ranking, RAG, impacto + radar)
# ===============================================================
# - Multi-archivo (XLSX/CSV) con autodetección de hoja (XLSX)
# - Timeframe (fecha/temporada)
# - XGBoost para "Market value" (early stopping) + métricas
# - Perfil de club por GRUPO posicional (PCA), vecinos y centroides ponderados por minutos
# - Ranking EXCLUYENDO jugadores del club destino + filtros por grupos
# - Impacto de fichaje (puedes cargar dataset externo) + RADAR y títulos con plot_text
# - RAG (TF-IDF + ChatOpenAI) compatible Py 3.13 (sin FAISS)
# - Estilo visual “chalkboard” + Comic Sans en todos los gráficos
# - Dinero formateado a millones, descargas CSV
# ===============================================================

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor

# ---- RAG (chat) con OpenAI ----
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# ---- Soccerplots (Radar + plot_text) ----
try:
    from soccerplots.radar_chart import Radar
    try:
        from soccerplots.plot_text import plot_text  # según docs
    except Exception:
        from soccerplots.utils import plot_text  # fallback
    SOCCERPLOTS_AVAILABLE = True
except Exception:
    SOCCERPLOTS_AVAILABLE = False
    plot_text = None  # type: ignore

# ===============================================================
# Estilo global “chalkboard” + Comic Sans
# ===============================================================
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = '#2E2E2E'  # pizarra
plt.rcParams['axes.edgecolor'] = 'white'
plt.rcParams['grid.color'] = 'white'
plt.rcParams['font.family'] = 'Comic Sans MS'
plt.rcParams['figure.facecolor'] = '#2E2E2E'
plt.rcParams['savefig.facecolor'] = '#2E2E2E'
plt.rcParams['text.color'] = 'white'

# ===============================================================
# Streamlit UI base
# ===============================================================
st.set_page_config(page_title="⚽ Scouting Tool", layout="wide", page_icon="⚽")
st.markdown(
    """
    <style>
      .small-note { color: #94a3b8; font-size: 0.92em; }
      .metric-card { background: #111827; border-radius: 12px; padding: 12px 16px; border: 1px solid #1f2937; }
      .tight { line-height: 1.2; }
      .ok { color:#34d399 } .warn { color:#f59e0b } .bad { color:#f87171 }
      .chalk { color:#e5e7eb; }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================================================
# Config
# ===============================================================
@dataclass
class Config:
    TARGET: str = "Market value"
    ID_COLS: Tuple[str, ...] = ("Player","Team","Position","Age","Contract expires")
    POS_COL: str = "Position"
    RANDOM_STATE: int = 42
    PCA_COMPONENTS: int = 12
    TOP_K_RECS: int = 20

CFG = Config()
np.random.seed(CFG.RANDOM_STATE)

# ===============================================================
# Normalización de posiciones (diccionario) + Vecindarios
# ===============================================================
RAW_TO_GROUP = {
  'GK': 'Goalkeeper',
  'LB': 'Fullback',  'LWB': 'Fullback',
  'RB': 'Fullback',  'RWB': 'Fullback',
  'LCB': 'Defender', 'CB': 'Defender', 'RCB': 'Defender',
  'AMF': 'Midfielder', 'CMF': 'Midfielder',
  'DMF': 'Midfielder', 'LDMF': 'Midfielder', 'RDMF': 'Midfielder',
  'LAMF': 'Wingers', 'RAMF': 'Wingers', 'LW': 'Wingers', 'LWF': 'Wingers',
  'RW': 'Wingers', 'RWF': 'Wingers',
  'CF': 'Forward'
}
GROUP_NEIGHBORS = {
    'Goalkeeper': set(),
    'Defender': {'Fullback', 'Midfielder'},
    'Fullback': {'Defender', 'Wingers'},
    'Midfielder': {'Defender', 'Wingers', 'Forward'},
    'Wingers': {'Fullback', 'Midfielder', 'Forward'},
    'Forward': {'Wingers', 'Midfielder'}
}
ALL_GROUPS = ['Goalkeeper','Fullback','Defender','Midfielder','Wingers','Forward']

def _tokens(s: str) -> list:
    s = str(s).replace("/", ",").replace("-", " ")
    s = " ".join(s.split())
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def normalize_pos_to_group(pos_str: str) -> list:
    if pd.isna(pos_str) or str(pos_str).strip() == "":
        return []
    out = []
    for t in _tokens(pos_str):
        if t in RAW_TO_GROUP:
            out.append(RAW_TO_GROUP[t]); continue
        if t.title() in ALL_GROUPS:
            out.append(t.title()); continue
    # de-dup
    seen, dedup = set(), []
    for g in out:
        if g and g not in seen:
            seen.add(g); dedup.append(g)
    return dedup

# ===============================================================
# Utilidades de IO y limpieza
# ===============================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [" ".join(str(c).replace("\n"," ").replace("\r"," ").split()) for c in df.columns]
    return df

def format_millions(v):
    try:
        return f"{float(v)/1_000_000:.2f} M"
    except Exception:
        return v

def display_money(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [CFG.TARGET, "predicted_value", "delta_pred_real"]:
        if col in out.columns:
            out[col] = out[col].apply(format_millions)
    return out

def auto_pick_sheet(xls: pd.ExcelFile, filename: str, required={"Player","Team","Market value"}) -> str:
    cands = [s for s in xls.sheet_names if any(k in s.lower() for k in ("search","result","players","jugadores","sheet","datos"))]
    if cands: return cands[0]
    best, best_score = None, -1
    for s in xls.sheet_names:
        try:
            preview = pd.read_excel(filename, sheet_name=s, nrows=5)
            cols_norm = {str(c).strip() for c in preview.columns}
            score = len(required.intersection(cols_norm))
            if score > best_score:
                best, best_score = s, score
        except Exception:
            pass
    return best or xls.sheet_names[0]

@st.cache_data(show_spinner=False)
def read_any(file_bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        xls = pd.ExcelFile(file_bytes)
        chosen = auto_pick_sheet(xls, filename)
        df = pd.read_excel(file_bytes, sheet_name=chosen)
        df["__source_file__"] = filename
        df["__source_sheet__"] = chosen
    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(file_bytes)
        df["__source_file__"] = filename
        df["__source_sheet__"] = ""
    else:
        raise ValueError("Formato no soportado. Usa .xlsx o .csv")
    return normalize_columns(df)

def detect_time_columns(df: pd.DataFrame):
    date_cols, season_cols = [], []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date","fecha","match")):
            try:
                pd.to_datetime(df[c], errors="raise")
                date_cols.append(c)
            except Exception:
                pass
        elif np.issubdtype(df[c].dtype, np.datetime64):
            date_cols.append(c)
        if any(k in lc for k in ("season","temporada","year","año")):
            season_cols.append(c)
    return date_cols, season_cols

def apply_timeframe(df: pd.DataFrame,
                    date_col: Optional[str], date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
                    season_col: Optional[str], seasons_selected: Optional[List]) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns and date_range:
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out[(out[date_col] >= date_range[0]) & (out[date_col] <= date_range[1])]
    if season_col and season_col in out.columns and seasons_selected:
        out = out[out[season_col].isin(seasons_selected)]
    return out.reset_index(drop=True)

def select_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, List[str]]:
    if CFG.TARGET not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{CFG.TARGET}'.")
    df2 = df.copy()
    df2[CFG.TARGET] = pd.to_numeric(df2[CFG.TARGET], errors="coerce")
    num_cols = []
    for c in df2.columns:
        if c == CFG.TARGET: continue
        if np.issubdtype(df2[c].dtype, np.number):
            num_cols.append(c)
        else:
            coerced = pd.to_numeric(df2[c], errors="coerce")
            if coerced.notna().any():
                df2[c] = coerced; num_cols.append(c)
    if not num_cols:
        raise ValueError("No hay columnas numéricas suficientes para entrenar.")
    df2[num_cols] = df2[num_cols].replace([np.inf, -np.inf], np.nan)
    df2[CFG.TARGET] = df2[CFG.TARGET].replace([np.inf, -np.inf], np.nan)
    df2 = df2.dropna(subset=[CFG.TARGET])
    X = df2[num_cols].fillna(0.0).astype(np.float64)
    y = df2[CFG.TARGET].astype(np.float64)
    if len(X) == 0:
        raise ValueError("Tras la limpieza, no quedan filas para entrenar.")
    if np.allclose(y.values, y.values[0]):
        st.warning("El target es constante tras el filtro; el modelo puede no entrenar correctamente.")
    return df2, X, y, num_cols

# ===============================================================
# Espacios latentes & Modelo (guardamos scaler y pca para externos)
# ===============================================================
class LatentSpaces:
    def __init__(self, n_components=CFG.PCA_COMPONENTS):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=CFG.RANDOM_STATE)
    def fit_transform(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        Xs_np = self.scaler.fit_transform(X)
        Xs = pd.DataFrame(Xs_np, columns=X.columns, index=X.index)
        pca_latent_np = self.pca.fit_transform(Xs)
        pca_latent = pd.DataFrame(pca_latent_np, index=X.index)
        return {"scaled": Xs, "pca": pca_latent, "scaler": self.scaler, "pca_model": self.pca}

class MarketValueModel:
    def __init__(self, fast_mode: bool = True):
        self.fast_mode = fast_mode
        self.model = XGBRegressor(
            n_estimators=300 if fast_mode else 600,
            learning_rate=0.08 if fast_mode else 0.06,
            max_depth=6 if fast_mode else 8,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.5,
            tree_method="hist", n_jobs=0, random_state=CFG.RANDOM_STATE
        )
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=CFG.RANDOM_STATE)
        Xtr_np = np.ascontiguousarray(np.asarray(Xtr, dtype=np.float32))
        Xte_np = np.ascontiguousarray(np.asarray(Xte, dtype=np.float32))
        ytr_np = np.ascontiguousarray(np.asarray(ytr, dtype=np.float32)).ravel()
        yte_np = np.ascontiguousarray(np.asarray(yte, dtype=np.float32)).ravel()
        if (np.isnan(Xtr_np).any() or np.isnan(Xte_np).any() or np.isnan(ytr_np).any() or np.isnan(yte_np).any()):
            raise ValueError("Hay NaN en X o y tras preprocesar.")
        if (np.isinf(Xtr_np).any() or np.isinf(Xte_np).any()):
            raise ValueError("Hay valores Inf en X tras preprocesar.")
        try:
            self.model.fit(Xtr_np, ytr_np, eval_set=[(Xte_np, yte_np)], eval_metric="mae", verbose=False, early_stopping_rounds=50)
        except TypeError:
            self.model.fit(Xtr_np, ytr_np)
        y_pred = self.model.predict(Xte_np)
        holdout_mae = float(mean_absolute_error(yte_np, y_pred))
        r2 = float(r2_score(yte_np, y_pred))
        if getattr(self, "fast_mode", True):
            cv_mae = float("nan")
        else:
            n = min(4000, len(X))
            if n < len(X):
                rng = np.random.default_rng(CFG.RANDOM_STATE)
                idx = rng.choice(len(X), size=n, replace=False)
                X_cv = np.ascontiguousarray(X.iloc[idx].to_numpy(dtype=np.float32))
                y_cv = np.ascontiguousarray(y.iloc[idx].to_numpy(dtype=np.float32)).ravel()
            else:
                X_cv = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
                y_cv = np.ascontiguousarray(y.to_numpy(dtype=np.float32)).ravel()
            kf = KFold(n_splits=3, shuffle=True, random_state=CFG.RANDOM_STATE)
            scores = []
            for tr_idx, te_idx in kf.split(X_cv):
                mdl = XGBRegressor(**self.model.get_xgb_params())
                Xtr_k, Xte_k = X_cv[tr_idx], X_cv[te_idx]
                ytr_k, yte_k = y_cv[tr_idx], y_cv[te_idx]
                try:
                    mdl.fit(Xtr_k, ytr_k, eval_set=[(Xte_k, yte_k)], eval_metric="mae", verbose=False, early_stopping_rounds=30)
                except TypeError:
                    mdl.fit(Xtr_k, ytr_k)
                from sklearn.metrics import mean_absolute_error as _mae
                scores.append(_mae(yte_k, mdl.predict(Xte_k)))
            cv_mae = float(np.mean(scores)) if scores else float("nan")
        return {"cv_mae": cv_mae, "holdout_mae": holdout_mae, "r2": r2}

def nice_barh(ax, labels, values, title=None):
    """Barra horizontal moderna con tema chalkboard."""
    ax.barh(labels, values, color='#94a3b8', edgecolor='white', alpha=0.9, linewidth=1.2)
    ax.grid(True, axis='x', linestyle='--', alpha=0.35)
    ax.invert_yaxis()
    if title:
        ax.set_title(title, fontsize=16, pad=10)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

def plot_importances(model: XGBRegressor, X: pd.DataFrame, top_n: int = 25):
    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(9,6))
        nice_barh(ax, list(np.array(X.columns)[idx][::-1]), list(np.array(importances)[idx][::-1]),
                  title="Importancia de características (XGBoost)")
        st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.info(f"Importancias no disponibles: {e}")

# ===============================================================
# ClubFit (grupos + vecinos + minutos ponderados) + Impacto
# ===============================================================
def _pick_minutes_column(df: pd.DataFrame) -> Optional[str]:
    cand = [c for c in df.columns if "minute" in c.lower() or c.lower() in ("min","mins","minutes","90s","minutes played")]
    if not cand: return None
    for k in ("Minutes","minutes","MIN","Min","mins","90s","Minutes played","minutes played"):
        if k in df.columns: return k
    return cand[0]

class ClubFit:
    """
    Fit-score: similitud coseno entre PCA del jugador y centroides del club por GRUPO.
    Centroides ponderados por minutos si hay columna de minutos.
    position_match: 1.0 exacto / 0.7 vecino / 0.2 sin matching.
    signing_score = position_match * (0.55*fit + 0.35*z(pred)) - 0.10*z(Market value)
    """
    def __init__(self, df_full: pd.DataFrame, spaces: Dict[str, pd.DataFrame]):
        self.df = df_full.copy()
        self.latent = spaces["pca"]
        self.scaled = spaces["scaled"]
        self.scaler = spaces.get("scaler", None)
        self.pca_model = spaces.get("pca_model", None)
        self.df["_GroupPos"] = self.df.get(CFG.POS_COL, "").apply(normalize_pos_to_group)
        self.minutes_col = _pick_minutes_column(self.df)
        self.centroids = self._compute_centroids()

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _compute_centroids(self) -> Dict[tuple, np.ndarray]:
        d = {}
        if "Team" not in self.df.columns: return d
        for club, g in self.df.groupby("Team"):
            bucket: Dict[str, List[int]] = {}
            for i, row in g.iterrows():
                for grp in row.get("_GroupPos", []):
                    bucket.setdefault(grp, []).append(i)
            for grp, idxs in bucket.items():
                if not idxs: continue
                if self.minutes_col and self.minutes_col in self.df.columns:
                    w = self.df.loc[idxs, self.minutes_col].astype(float).fillna(0).to_numpy()
                    if w.sum() > 0:
                        mat = self.latent.loc[idxs].to_numpy()
                        d[(club, grp)] = np.average(mat, axis=0, weights=w); continue
                d[(club, grp)] = self.latent.loc[idxs].mean(axis=0).to_numpy()
        return d

    def _best_similarity_vs_club(self, vec: np.ndarray, player_groups: list, target_club: str) -> tuple[float, float]:
        best_exact, best_neighbor = 0.0, 0.0
        has_exact, has_neighbor = False, False
        for g in player_groups:
            key = (target_club, g)
            if key in self.centroids:
                has_exact = True
                best_exact = max(best_exact, self._cosine(vec, self.centroids[key]))
            for ng in GROUP_NEIGHBORS.get(g, set()):
                keyn = (target_club, ng)
                if keyn in self.centroids:
                    has_neighbor = True
                    best_neighbor = max(best_neighbor, self._cosine(vec, self.centroids[keyn]))
        if has_exact: return best_exact, 1.0
        if has_neighbor: return best_neighbor, 0.7
        return 0.0, 0.2

    def fit_score(self, idx: int, target_club: str) -> float:
        if ("Team" not in self.df.columns) or (target_club not in self.df["Team"].unique()):
            return 0.0
        vec = self.latent.loc[idx].to_numpy()
        groups = self.df.loc[idx].get("_GroupPos", [])
        base, _ = self._best_similarity_vs_club(vec, groups, target_club)
        return float(base)

    def _with_scores(self, target_club: str,
                     required_groups: Optional[set] = None,
                     strict_groups: bool = False) -> pd.DataFrame:
        df = self.df.copy()
        if "predicted_value" not in df or CFG.TARGET not in df:
            raise ValueError("Faltan columnas 'predicted_value' o target.")
        base_fit, pmatch = [], []
        for i in df.index:
            base, pm = self._best_similarity_vs_club(self.latent.loc[i].to_numpy(),
                                                     df.loc[i, "_GroupPos"], target_club)
            base_fit.append(base); pmatch.append(pm)
        df["fit_score"] = base_fit
        df["position_match"] = pmatch
        for col in ["predicted_value", CFG.TARGET]:
            mu, sd = df[col].mean(), df[col].std() + 1e-9
            df[f"{col}_z"] = (df[col]-mu)/sd
        df["signing_score"] = df["position_match"] * (0.55*df["fit_score"] + 0.35*df["predicted_value_z"]) - 0.10*df[f"{CFG.TARGET}_z"]
        if required_groups:
            if strict_groups:
                df = df[df["_GroupPos"].apply(lambda lst: any(g in required_groups for g in lst))]
            else:
                def any_close(lst):
                    for g in lst:
                        if g in required_groups: return True
                        if GROUP_NEIGHBORS.get(g, set()) & required_groups: return True
                    return False
                df = df[df["_GroupPos"].apply(any_close)]
        return df

    def rank_signings(self, target_club: str, top_k: int = CFG.TOP_K_RECS,
                      required_groups: Optional[set] = None, strict_groups: bool = False) -> pd.DataFrame:
        df = self._with_scores(target_club, required_groups, strict_groups)
        df = df[df["Team"] != target_club]
        cols = ["Player","Team","Position","_GroupPos","Age",CFG.TARGET,"predicted_value","fit_score","position_match","signing_score"]
        cols = [c for c in cols if c in df.columns]
        return df.sort_values("signing_score", ascending=False).head(top_k)[cols]

    # -------- Impacto de fichaje --------
    def impact_of_signing(self, target_club: str, player_vec_pca: np.ndarray,
                          player_groups: List[str], assumed_minutes: float) -> Dict[str, float]:
        # elegir grupo principal
        grp_chosen = None
        for g in player_groups:
            if (target_club, g) in self.centroids:
                grp_chosen = g; break
        if not grp_chosen:
            for g in player_groups:
                for ng in GROUP_NEIGHBORS.get(g, set()):
                    if (target_club, ng) in self.centroids:
                        grp_chosen = ng; break
                if grp_chosen: break
        if not grp_chosen and player_groups:
            grp_chosen = player_groups[0]

        club_mask = (self.df["Team"] == target_club) & self.df["_GroupPos"].apply(lambda lst: grp_chosen in lst if isinstance(lst, list) else False)
        idxs = self.df[club_mask].index.tolist()

        base_fit, pm = self._best_similarity_vs_club(player_vec_pca, player_groups, target_club)
        if not idxs:
            return {
                "player_fit": base_fit, "position_match": pm, "centroid_drift": 0.0,
                "group_cohesion_delta": 0.0, "group_size": 0, "grp_for_radar": grp_chosen or "Midfielder"
            }

        if self.minutes_col and self.minutes_col in self.df.columns:
            w = self.df.loc[idxs, self.minutes_col].astype(float).fillna(0).to_numpy()
            if w.sum() > 0:
                centroid_old = np.average(self.latent.loc[idxs].to_numpy(), axis=0, weights=w)
            else:
                centroid_old = self.latent.loc[idxs].mean(axis=0).to_numpy()
        else:
            centroid_old = self.latent.loc[idxs].mean(axis=0).to_numpy()

        assumed_minutes = max(0.0, float(assumed_minutes))
        if self.minutes_col and self.minutes_col in self.df.columns:
            w = self.df.loc[idxs, self.minutes_col].astype(float).fillna(0).to_numpy()
            mat = self.latent.loc[idxs].to_numpy()
            sum_w = w.sum()
            if sum_w + assumed_minutes > 0:
                centroid_new = (mat.T @ w + player_vec_pca * assumed_minutes) / (sum_w + assumed_minutes)
            else:
                centroid_new = centroid_old
        else:
            centroid_new = (self.latent.loc[idxs].to_numpy().sum(axis=0) + player_vec_pca) / (len(idxs) + 1)

        centroid_drift = float(np.linalg.norm(centroid_new - centroid_old))

        sims_old = [self._cosine(self.latent.loc[i].to_numpy(), centroid_old) for i in idxs]
        sims_new = [self._cosine(self.latent.loc[i].to_numpy(), centroid_new) for i in idxs]
        group_cohesion_delta = float(np.mean(sims_new) - np.mean(sims_old))

        return {
            "player_fit": base_fit,
            "position_match": pm,
            "centroid_drift": centroid_drift,
            "group_cohesion_delta": group_cohesion_delta,
            "group_size": len(idxs),
            "grp_for_radar": grp_chosen or "Midfielder"
        }

# ===============================================================
# Sidebar (inputs) — keys ÚNICOS para evitar StreamlitDuplicateElementId
# ===============================================================
with st.sidebar:
    st.header("① Carga de archivos")
    uploaded_files = st.file_uploader("Sube uno o varios (.xlsx / .csv)", type=["xlsx","csv"], accept_multiple_files=True, key="u_main")

    st.header("② Filtro timeframe")
    st.caption("Se aplica ANTES de entrenar el modelo y calcular encaje/ranking.")

    st.header("③ Clave OpenAI (RAG)")
    default_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=default_key, help="Usa Secrets en producción; aquí puedes pegarla temporalmente.", key="openai_key")

    st.header("④ Parámetros")
    fast_mode = st.checkbox("Modo rápido (entrena más veloz, sin CV)", value=True, key="fast_mode")
    top_k = st.number_input("Top K (ranking)", min_value=5, max_value=200, value=CFG.TOP_K_RECS, step=1, key="topk_rank")
    show_importances = st.checkbox("Mostrar importancias del modelo", value=True, key="show_imps")

st.title("⚽ Scouting Tool — Avanzada (Chalkboard theme)")

if not uploaded_files:
    st.info("Sube al menos un archivo para comenzar.")
    st.stop()

# ===============================================================
# Merge datasets
# ===============================================================
frames = []
for f in uploaded_files:
    try:
        df_tmp = read_any(f, f.name)
        frames.append(df_tmp)
    except Exception as e:
        st.error(f"Error leyendo {f.name}: {e}")
        st.stop()

df_raw = pd.concat(frames, ignore_index=True, sort=False)
date_cols, season_cols = detect_time_columns(df_raw)

# ===============================================================
# Timeframe controls
# ===============================================================
st.subheader("Vista previa y timeframe")
st.dataframe(df_raw.head(15), use_container_width=True)

col1, col2 = st.columns(2)
date_col = season_col = None
date_range = seasons_selected = None

with col1:
    chosen_date_col = st.selectbox("Columna de fecha (opcional)", options=["(ninguna)"]+date_cols, index=0, key="datecol")
    if chosen_date_col != "(ninguna)":
        series_dt = pd.to_datetime(df_raw[chosen_date_col], errors="coerce").dropna()
        if series_dt.empty:
            st.warning("La columna de fecha seleccionada no tiene fechas válidas.")
        else:
            min_dt, max_dt = series_dt.min(), series_dt.max()
            date_range = st.slider("Rango de fechas", min_value=min_dt.to_pydatetime(), max_value=max_dt.to_pydatetime(),
                                   value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()), key="dateslider")
            date_col = chosen_date_col

with col2:
    chosen_season_col = st.selectbox("Columna de temporada/año (opcional)", options=["(ninguna)"]+season_cols, index=0, key="seasoncol")
    if chosen_season_col != "(ninguna)":
        uniq = sorted([x for x in df_raw[chosen_season_col].dropna().unique().tolist()])
        seasons_selected = st.multiselect("Temporadas / Años", options=uniq, default=uniq, key="seasonmulti")
        season_col = chosen_season_col

st.caption("Todo se calcula **sólo** con datos dentro del timeframe elegido (Team within selected timeframe).")

# ===============================================================
# Prep data
# ===============================================================
df_f = apply_timeframe(df_raw, date_col, date_range, season_col, seasons_selected)
if df_f.empty:
    st.warning("El filtro de timeframe dejó el dataset vacío. Ajusta los filtros.")
    st.stop()

try:
    df_for_model, X, y, num_cols = select_numeric(df_f)
except Exception as e:
    st.error(f"Error preparando datos: {e}")
    st.stop()

# ===============================================================
# Train (cache)
# ===============================================================
@st.cache_resource(show_spinner=False)
def train_pipeline(df_in: pd.DataFrame, X: pd.DataFrame, y: pd.Series, fast_mode: bool):
    status = st.empty()
    prog = st.progress(0)
    status.info("Inicializando modelo…"); prog.progress(10)
    model = MarketValueModel(fast_mode=fast_mode)
    status.info("Entrenando con early stopping…"); prog.progress(55)
    metrics = model.train(X, y)
    status.info("Generando predicciones…"); prog.progress(80)
    X_np_all = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    y_hat = model.model.predict(X_np_all)
    status.info("Calculando espacios latentes (PCA)…")
    spaces = LatentSpaces().fit_transform(X)
    prog.progress(100); status.success("Entrenamiento finalizado.")
    return model, metrics, y_hat, spaces

st.subheader("Entrenamiento del modelo (dentro del timeframe)")
model, metrics, y_hat, spaces = train_pipeline(df_for_model, X, y, fast_mode)

df_pred = df_for_model.copy()
df_pred["predicted_value"] = y_hat
df_pred["delta_pred_real"] = df_pred["predicted_value"] - df_pred[CFG.TARGET]

# ---- asegurar _GroupPos globalmente (siempre disponible y en formato lista) ----
if "_GroupPos" not in df_pred.columns:
    pos_series = df_pred.get(CFG.POS_COL, pd.Series([""] * len(df_pred), index=df_pred.index))
    df_pred["_GroupPos"] = pos_series.apply(normalize_pos_to_group)
else:
    df_pred["_GroupPos"] = df_pred["_GroupPos"].apply(
        lambda v: v if isinstance(v, (list, tuple, set)) else normalize_pos_to_group(v)
    )

# Métricas (tarjetas)
m1, m2, m3 = st.columns(3)
m1.markdown(f"<div class='metric-card tight chalk'><b>CV MAE</b><br>{'—' if np.isnan(metrics['cv_mae']) else f'{metrics['cv_mae']:,.0f}'}</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card tight chalk'><b>Holdout MAE</b><br>{metrics['holdout_mae']:,.0f}</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card tight chalk'><b>R²</b><br>{metrics['r2']:.3f}</div>", unsafe_allow_html=True)

if show_importances:
    st.markdown("**Importancia de variables**")
    plot_importances(model.model, X, top_n=25)

# Fit scorer
fitter = ClubFit(df_pred, spaces)

# ===============================================================
# Tabs
# ===============================================================
tab_pred, tab_fit, tab_rank, tab_impact, tab_rag, tab_help = st.tabs(
    ["📈 Predicciones", "🎯 Encaje jugador→club", "🏟 Mejores incorporaciones", "🧩 Impacto de fichaje", "💬 Chat (RAG)", "📘 Glosario"]
)

with tab_pred:
    st.subheader("Predicciones de valor de mercado")
    cols_show = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","delta_pred_real"] if c in df_pred.columns]
    st.dataframe(display_money(df_pred[cols_show].head(500)), use_container_width=True)
    st.caption("Se muestran 500 filas por rendimiento. Descarga para ver completo.")
    st.download_button("Descargar predicciones (CSV)",
                       data=df_pred[cols_show].to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv", key="dl_pred")

with tab_fit:
    st.subheader("Encaje de un jugador en un club")
    if ("Player" not in df_pred.columns) or ("Team" not in df_pred.columns):
        st.error("Necesito columnas 'Player' y 'Team'.")
    else:
        players = sorted(df_pred["Player"].dropna().unique().tolist())
        clubs = sorted(df_pred["Team"].dropna().unique().tolist())
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.selectbox("Jugador", options=players, key="fit_player_sel")
        with c2:
            target_club = st.selectbox("Club destino", options=clubs, key="fit_club_sel")
        if st.button("Evaluar encaje", key="btn_fit_eval"):
            try:
                res = fitter.rank_signings(target_club, top_k=len(df_pred))
                out = fitter.eval_player_in_club(player_name, target_club)
                st.dataframe(display_money(out), use_container_width=True)
                st.markdown(
                    """
                    <div class='small-note'>
                    <b>_GroupPos</b>: grupos posicionales normalizados (Goalkeeper, Fullback, Defender, Midfielder, Wingers, Forward).<br>
                    <b>fit_score (0–1)</b>: similitud del jugador con el perfil del club (PCA, con centroides ponderados por minutos).<br>
                    <b>rank_for_target</b>: posición del jugador en el ranking del club (1=mejor), excluyendo a quienes ya están en ese club.
                    </div>
                    """, unsafe_allow_html=True
                )
            except Exception as e:
                st.error(str(e))

with tab_rank:
    st.subheader("Ranking de mejores incorporaciones (excluye jugadores del club)")
    if "Team" not in df_pred.columns:
        st.error("Necesito columna 'Team'.")
    else:
        clubs = sorted(df_pred["Team"].dropna().unique().tolist())
        target_club_r = st.selectbox("Club destino", options=clubs, key="rank_club_sel_unique")
        req_groups = st.multiselect("Grupos requeridos (opcional)", options=ALL_GROUPS, default=[], key="rank_groups")
        strict_groups = st.checkbox("Modo estricto por grupo", value=False, key="rank_strict")
        if st.button("Calcular ranking", key="btn_rank"):
            try:
                req_set = set(req_groups) if req_groups else None
                board = fitter.rank_signings(target_club_r, top_k=int(top_k),
                                             required_groups=req_set, strict_groups=strict_groups)
                st.dataframe(display_money(board), use_container_width=True)
                st.caption("Centroides por grupo ponderados por minutos (si hay); se consideran vecinos de rol.")
                st.download_button(f"Descargar ranking {target_club_r} (CSV)",
                                   data=board.to_csv(index=False).encode("utf-8"),
                                   file_name=f"ranking_{target_club_r}.csv", mime="text/csv", key="dl_rank")
            except Exception as e:
                st.error(str(e))

# ===============================================================
# 🧩 Impacto de fichaje (dataset externo + radar + plot_text)
# ===============================================================
with tab_impact:
    st.subheader("Impacto de fichaje (dataset externo opcional + radar)")

    c_ext1, c_ext2 = st.columns([2,1])
    with c_ext1:
        ext_file = st.file_uploader("Dataset externo (opcional) — otra liga (.xlsx/.csv)", type=["xlsx","csv"], accept_multiple_files=False, key="ext_upl")
    with c_ext2:
        expected_minutes = st.number_input("Minutos esperados del jugador", min_value=0, max_value=6000, value=1800, step=90, key="exp_min")

    df_ext = None
    if ext_file is not None:
        try:
            df_ext = read_any(ext_file, ext_file.name)
        except Exception as e:
            st.warning(f"No se pudo leer el dataset externo: {e}")

    pool_main = sorted(df_pred["Player"].dropna().unique().tolist()) if "Player" in df_pred.columns else []
    pool_ext = sorted(df_ext["Player"].dropna().unique().tolist()) if (df_ext is not None and "Player" in df_ext.columns) else []

    c_src1, c_src2 = st.columns(2)
    with c_src1:
        player_source = st.radio("Fuente del jugador a evaluar", options=["Dataset principal","Dataset externo"], horizontal=True, key="impact_src")
    with c_src2:
        clubs_imp = sorted(df_pred["Team"].dropna().unique().tolist()) if "Team" in df_pred.columns else []
        club_target_imp = st.selectbox("Club destino", options=clubs_imp, key="impact_club_sel")

    if player_source == "Dataset principal":
        player_sel = st.selectbox("Jugador (principal)", options=pool_main, key="impact_player_main")
    else:
        if df_ext is None:
            st.info("Sube un dataset externo para usar esta opción.")
            player_sel = None
        else:
            player_sel = st.selectbox("Jugador (externo)", options=pool_ext, key="impact_player_ext")

    # Transformar externo al espacio del modelo con scaler y pca entrenados
    def external_to_pca(df_ext_in: pd.DataFrame, name: str) -> Optional[Dict]:
        if df_ext_in is None or name is None: return None
        rows = df_ext_in[df_ext_in["Player"].astype(str).str.lower() == str(name).lower()]
        if rows.empty: return None
        row = rows.iloc[0]
        groups = normalize_pos_to_group(row.get(CFG.POS_COL, ""))
        # features como X
        vec_raw = {col: pd.to_numeric(row.get(col, 0), errors="coerce") for col in X.columns}
        x_row = pd.DataFrame([vec_raw], index=[0]).fillna(0.0)
        scaler = spaces["scaler"]
        pca = spaces["pca_model"]
        x_scaled = scaler.transform(x_row[scaler.feature_names_in_]) if hasattr(scaler, "feature_names_in_") else scaler.transform(x_row.values)
        pca_vec = pca.transform(x_scaled)[0]
        return {"groups": groups if groups else ["Midfielder"], "pca_vec": pca_vec, "x_row": x_row}

    # radar helpers
    def pick_radar_features(cols: List[str]) -> List[str]:
        preferred = [c for c in cols if any(k in c.lower() for k in [" per 90","per90","xg","xa","pass","shot","duel","dribble","tackle","interception","key pass","cross","progress","press"])]
        if len(preferred) >= 8:
            return preferred[:8]
        try:
            imp = model.model.feature_importances_
            order = np.argsort(imp)[::-1]
            extras = [cols[i] for i in order if cols[i] not in preferred]
            return (preferred + extras)[:8]
        except Exception:
            return cols[:8]

    def radar_and_title(player_label: str, club_label: str, grp: str, values_player, values_centroid, ranges, params):
        if not SOCCERPLOTS_AVAILABLE:
            st.info("Instala `soccerplots` para ver radar y plot_text.")
            return
        radar = Radar(params=params, low=[r[0] for r in ranges], high=[r[1] for r in ranges], round_int=[False]*len(params))
        fig, ax = radar.plot_radar(
            ranges=ranges, params=params, values=[values_player, values_centroid],
            radar_color=['#10b981', '#3b82f6'], alphas=[0.65, 0.45],
            title=dict(title_name=player_label, title_color='white', subtitle=f"vs. perfil {grp} de {club_label}", subtitle_color='#94a3b8'),
            compare=True
        )
        try:
            if plot_text is not None:
                plot_text(ax=fig.axes[0], x=0.5, y=1.08,
                          s="Comparativa de Radar — Chalkboard",
                          color="#fde68a", size=14, ha="center", va="bottom", highlight=True)
        except Exception:
            pass
        st.pyplot(fig, use_container_width=True)

    if st.button("Simular impacto", key="btn_impact"):
        if player_sel is None:
            st.warning("Elige un jugador.")
        else:
            if player_source == "Dataset principal":
                row_mask = df_pred["Player"].astype(str).str.lower() == player_sel.lower()
                if not row_mask.any():
                    st.error("Jugador no encontrado en el dataset principal.")
                else:
                    idx = df_pred[row_mask].index[0]
                    p_groups = df_pred.loc[idx].get("_GroupPos", normalize_pos_to_group(df_pred.loc[idx].get(CFG.POS_COL, "")))
                    p_vec_pca = spaces["pca"].loc[idx].to_numpy()
                    impact = fitter.impact_of_signing(club_target_imp, p_vec_pca, p_groups, assumed_minutes=expected_minutes)
                    tag = "Titular probable" if (impact["player_fit"]>=0.65 and impact["position_match"]>=0.7) else ("Rotación competitiva" if impact["player_fit"]>=0.45 else "Ajuste gradual")
                    colA, colB, colC, colD = st.columns(4)
                    colA.metric("Fit del jugador", f"{impact['player_fit']:.2f}")
                    colB.metric("Match posicional", f"{impact['position_match']:.2f}")
                    colC.metric("Δ Cohesión grupo", f"{impact['group_cohesion_delta']:+.3f}")
                    colD.metric("Deriva centroide", f"{impact['centroid_drift']:.3f}")
                    st.markdown(f"**Dictamen:** {tag}")
                    # Radar
                    grp = impact.get("grp_for_radar", (p_groups[0] if p_groups else "Midfielder"))
                    features = pick_radar_features(list(X.columns))
                    base = df_pred.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                    ranges = list(zip(base.min().tolist(), base.max().tolist()))
                    # ---- mask_grp robusto (1/2) ----
                    has_grp = df_pred["_GroupPos"].apply(lambda lst: (grp in lst) if isinstance(lst, (list, tuple, set)) else False)
                    mask_grp = (df_pred["Team"] == club_target_imp) & has_grp
                    centroid_vals = df_pred.loc[mask_grp, features].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean().to_numpy()
                    player_vals = df_pred.loc[idx, features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
                    radar_and_title(player_sel, club_target_imp, grp, player_vals, centroid_vals, ranges, features)
                    # Summary
                    st.markdown(
                        f"""
                        **Resumen automático**  
                        - Encaje en **{grp}** del **{club_target_imp}** → fit **{impact['player_fit']:.2f}**, match **{impact['position_match']:.2f}**.  
                        - La **cohesión** del grupo cambiaría **{impact['group_cohesion_delta']:+.3f}**; el **centroide** se movería **{impact['centroid_drift']:.3f}** (PCA).  
                        - Con **{expected_minutes}′** previstos → **{tag}**.
                        """.strip()
                    )
            else:
                if df_ext is None:
                    st.error("Carga primero un dataset externo.")
                else:
                    ext = external_to_pca(df_ext, player_sel)
                    if not ext:
                        st.error("No se pudo construir el perfil del jugador externo (¿faltan columnas?).")
                    else:
                        impact = fitter.impact_of_signing(club_target_imp, ext["pca_vec"], ext["groups"], assumed_minutes=expected_minutes)
                        tag = "Titular probable" if (impact["player_fit"]>=0.65 and impact["position_match"]>=0.7) else ("Rotación competitiva" if impact["player_fit"]>=0.45 else "Ajuste gradual")
                        colA, colB, colC, colD = st.columns(4)
                        colA.metric("Fit del jugador", f"{impact['player_fit']:.2f}")
                        colB.metric("Match posicional", f"{impact['position_match']:.2f}")
                        colC.metric("Δ Cohesión grupo", f"{impact['group_cohesion_delta']:+.3f}")
                        colD.metric("Deriva centroide", f"{impact['centroid_drift']:.3f}")
                        st.markdown(f"**Dictamen:** {tag}")
                        # Radar: si el externo no trae todas las cols, 0s para faltantes
                        grp = impact.get("grp_for_radar", (ext["groups"][0] if ext["groups"] else "Midfielder"))
                        features = pick_radar_features(list(X.columns))
                        base = df_pred.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
                        ranges = list(zip(base.min().tolist(), base.max().tolist()))
                        # ---- mask_grp robusto (2/2) ----
                        has_grp = df_pred["_GroupPos"].apply(lambda lst: (grp in lst) if isinstance(lst, (list, tuple, set)) else False)
                        mask_grp = (df_pred["Team"] == club_target_imp) & has_grp
                        centroid_vals = df_pred.loc[mask_grp, features].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean().to_numpy()
                        v = []
                        for c in features:
                            try:
                                v.append(float(df_ext.loc[df_ext["Player"].astype(str).str.lower()==player_sel.lower(), c].iloc[0]))
                            except Exception:
                                v.append(0.0)
                        player_vals = np.array(v, dtype=float)
                        radar_and_title(player_sel, club_target_imp, grp, player_vals, centroid_vals, ranges, features)
                        st.markdown(
                            f"""
                            **Resumen automático**  
                            - Encaje en **{grp}** del **{club_target_imp}** → fit **{impact['player_fit']:.2f}**, match **{impact['position_match']:.2f}**.  
                            - La **cohesión** del grupo cambiaría **{impact['group_cohesion_delta']:+.3f}**; el **centroide** se movería **{impact['centroid_drift']:.3f}** (PCA).  
                            - Con **{expected_minutes}′** previstos → **{tag}**.
                            """.strip()
                        )

# ===============================================================
# 💬 RAG
# ===============================================================
with tab_rag:
    st.subheader("Chat con RAG (TF-IDF, OpenAI)")
    if not LANGCHAIN_AVAILABLE:
        st.info("RAG deshabilitado: falta langchain-openai. Revisa requirements.txt.")
    else:
        q_default = "Dame 5 posibles fichajes con alto fit y buen precio para {club}"
        question = st.text_input("Pregunta", value=q_default, key="rag_q")
        clubs = sorted(df_pred["Team"].dropna().unique().tolist()) if "Team" in df_pred.columns else []
        club_for_ctx = st.selectbox("Club para enriquecer contexto (opcional)", options=clubs, key="rag_club") if clubs else ""
        if st.button("Preguntar", key="btn_rag"):
            if not api_key:
                st.warning("Pega tu OPENAI_API_KEY en la barra lateral o define st.secrets/ENV.")
            else:
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    df_ctx = df_pred.copy()
                    if club_for_ctx:
                        tmp = fitter.rank_signings(club_for_ctx, top_k=len(df_ctx))
                        if "Player" in df_ctx.columns:
                            df_ctx = df_ctx.merge(tmp[["Player","fit_score","position_match","signing_score"]], on="Player", how="left")
                    docs = []
                    for _, r in df_ctx.iterrows():
                        lines = []
                        for c in CFG.ID_COLS:
                            if c in r: lines.append(f"{c}: {r[c]}")
                        for k in ["_GroupPos","predicted_value","fit_score","position_match","signing_score","Goals per 90","Assists per 90","xG per 90","xA per 90"]:
                            if k in df_ctx.columns and pd.notna(r.get(k, np.nan)):
                                lines.append(f"{k}: {r[k]}")
                        if CFG.TARGET in r:
                            lines.append(f"{CFG.TARGET}: {r.get(CFG.TARGET, np.nan)}")
                        docs.append("\n".join(lines))
                    ctx_q = question.replace("{club}", club_for_ctx) if club_for_ctx else question
                    vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
                    Xtf = vect.fit_transform(docs + [ctx_q])
                    sims = cosine_similarity(Xtf[-1], Xtf[:-1]).ravel()
                    top_idx = np.argsort(sims)[::-1][:6]
                    context_text = "\n\n---\n\n".join([docs[i] for i in top_idx])
                    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    msgs = [
                        SystemMessage(content="Eres analista de scouting. Responde con datos del contexto y recomendaciones accionables."),
                        HumanMessage(content=f"Contexto:\n{context_text}\n\nPregunta: {ctx_q}")
                    ]
                    answer = chat.invoke(msgs).content
                    st.write(answer)
                except Exception as e:
                    st.error(f"RAG error: {e}")

# ===============================================================
# Glosario
# ===============================================================
with tab_help:
    st.subheader("Glosario y criterios (tema chalkboard)")
    st.markdown(
        """
        - **Team within selected timeframe**: cálculos usando sólo tu rango de fechas/temporadas.  
        - **_GroupPos**: grupos posicionales normalizados: Goalkeeper, Fullback, Defender, Midfielder, Wingers, Forward.  
        - **Centroides ponderados**: si hay columna de minutos, ponderamos los centroides del club por minutos.  
        - **fit_score (0–1)**: similitud PCA jugador↔perfil del club (grupo exacto y vecinos).  
        - **position_match**: 1.0 si match exacto; 0.7 si vecino; 0.2 si ninguno.  
        - **signing_score**: `position_match * (0.55*fit + 0.35*z(predicted)) - 0.10*z(Market value)`.  
        - **Impacto de fichaje**: Δ cohesión del grupo y deriva del centroide al añadir minutos del jugador.  
        - **Radar**: comparación del jugador vs. centroide del grupo en features clave.  
        """.strip()
    )

st.success("Listo. Carga datasets, filtra timeframe y explora predicciones, encaje, ranking e impacto. 👌")
