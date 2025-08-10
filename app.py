# app.py
# ===============================================================
# ⚽ SCOUTING TOOL — Streamlit (multi-archivo, timeframe, ML, fit, ranking, RAG)
# ===============================================================
# - Carga de 1+ archivos (XLSX/CSV) con autodetección de hoja (XLSX)
# - Filtro por timeframe (fecha / temporada)
# - Modelo ML (XGBoost) para "Market value" + métricas rápidas (early stopping)
# - Perfil de club por GRUPO de posición (PCA) y fit_score jugador→club (0–1)
# - Centroides ponderados por minutos si hay columna de minutos
# - Ranking de incorporaciones EXCLUYENDO jugadores del club destino
# - Filtros por grupos requeridos y “modo estricto”
# - RAG (chat) con TF-IDF + ChatOpenAI (sin FAISS; compatible Py 3.13)
# - Valores monetarios redondeados a millones (display)
# - Descarga de CSVs y glosario detallado
# - Uso seguro de OPENAI_API_KEY: st.secrets / env var / input
# ===============================================================

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor

# RAG (chat) con OpenAI
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# -------------------------
# Config & estilo
# -------------------------
st.set_page_config(page_title="⚽ Scouting Tool", layout="wide", page_icon="⚽")
st.markdown(
    """
    <style>
      .small-note { color: #64748b; font-size: 0.92em; }
      .metric-card { background: #0f172a10; border-radius: 12px; padding: 12px 16px; }
      .tight { line-height: 1.2; }
      .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#0ea5e91a; color:#0369a1; font-size:12px; }
      .ok { color:#059669 } .warn { color:#b45309 } .bad { color:#dc2626 }
    </style>
    """,
    unsafe_allow_html=True
)

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

# =========================
# Normalización de posiciones (tu mapeo) + Vecinos
# =========================
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
    s = str(s)
    s = s.replace("/", ",").replace("-", " ")
    s = " ".join(s.split())
    return [t.strip().upper() for t in s.split(",") if t.strip()]

def normalize_pos_to_group(pos_str: str) -> list:
    """Convierte la(s) posición(es) crudas a grupos (tu diccionario). Ej: 'RB, CB' -> ['Fullback','Defender']"""
    if pd.isna(pos_str) or str(pos_str).strip() == "":
        return []
    out = []
    for t in _tokens(pos_str):
        if t in RAW_TO_GROUP:
            out.append(RAW_TO_GROUP[t]); continue
        # backoff: si viene exacto como grupo, lo aceptamos
        if t.title() in ALL_GROUPS:
            out.append(t.title()); continue
        # otras siglas comunes
        if t in RAW_TO_GROUP:
            out.append(RAW_TO_GROUP[t])
    # de-dup preservando orden
    seen, dedup = set(), []
    for g in out:
        if g and g not in seen:
            seen.add(g); dedup.append(g)
    return dedup

# -------------------------
# Utilidades base
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = str(c).replace("\n"," ").replace("\r"," ")
        c2 = " ".join(c2.split())
        new_cols.append(c2)
    df.columns = new_cols
    return df

def format_millions(v):
    """Formatea números a millones (3.45 M)."""
    try:
        return f"{float(v)/1_000_000:.2f} M"
    except Exception:
        return v

def display_money(df: pd.DataFrame) -> pd.DataFrame:
    """Copia con Market value / predicted_value / delta formateados a millones."""
    out = df.copy()
    for col in [CFG.TARGET, "predicted_value", "delta_pred_real"]:
        if col in out.columns:
            out[col] = out[col].apply(format_millions)
    return out

def auto_pick_sheet(xls: pd.ExcelFile, filename: str, required={"Player","Team","Market value"}) -> str:
    # Heurística por nombre
    cands = [s for s in xls.sheet_names if any(k in s.lower() for k in ("search","result","players","jugadores","sheet","datos"))]
    if cands:
        return cands[0]
    # Scoring por columnas
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
    """Lee XLSX/CSV, autodetecta hoja, normaliza columnas, añade __source_*."""
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
    """Detecta columnas de fecha y temporada/año."""
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
    """Valida TARGET, fuerza numéricos, elimina NaN/Inf y prepara X (float64) e y (float64)."""
    if CFG.TARGET not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{CFG.TARGET}'.")

    df2 = df.copy()
    df2[CFG.TARGET] = pd.to_numeric(df2[CFG.TARGET], errors="coerce")

    num_cols = []
    for c in df2.columns:
        if c == CFG.TARGET:
            continue
        if np.issubdtype(df2[c].dtype, np.number):
            num_cols.append(c)
        else:
            coerced = pd.to_numeric(df2[c], errors="coerce")
            if coerced.notna().any():
                df2[c] = coerced
                num_cols.append(c)

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

# -------------------------
# Espacios latentes & Modelo
# -------------------------
class LatentSpaces:
    def __init__(self, n_components=CFG.PCA_COMPONENTS):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=CFG.RANDOM_STATE)
    def fit_transform(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        Xs = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        pca_latent = pd.DataFrame(self.pca.fit_transform(Xs), index=X.index)
        return {"scaled": Xs, "pca": pca_latent}

class MarketValueModel:
    def __init__(self, fast_mode: bool = True):
        self.fast_mode = fast_mode
        self.model = XGBRegressor(
            n_estimators=300 if fast_mode else 600,
            learning_rate=0.08 if fast_mode else 0.06,
            max_depth=6 if fast_mode else 8,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.5,
            tree_method="hist",
            n_jobs=0,
            random_state=CFG.RANDOM_STATE
        )
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        # Holdout
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=CFG.RANDOM_STATE
        )

        # Conversión robusta
        Xtr_np = np.ascontiguousarray(np.asarray(Xtr, dtype=np.float32))
        Xte_np = np.ascontiguousarray(np.asarray(Xte, dtype=np.float32))
        ytr_np = np.ascontiguousarray(np.asarray(ytr, dtype=np.float32)).ravel()
        yte_np = np.ascontiguousarray(np.asarray(yte, dtype=np.float32)).ravel()

        if (np.isnan(Xtr_np).any() or np.isnan(Xte_np).any() or
            np.isnan(ytr_np).any() or np.isnan(yte_np).any()):
            raise ValueError("Hay NaN en X o y tras preprocesar; revisa el dataset o el filtro de timeframe.")
        if (np.isinf(Xtr_np).any() or np.isinf(Xte_np).any()):
            raise ValueError("Hay valores Inf en X tras preprocesar.")

        try:
            self.model.fit(
                Xtr_np, ytr_np,
                eval_set=[(Xte_np, yte_np)],
                eval_metric="mae",
                verbose=False,
                early_stopping_rounds=50
            )
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
                    mdl.fit(
                        Xtr_k, ytr_k,
                        eval_set=[(Xte_k, yte_k)],
                        eval_metric="mae",
                        verbose=False,
                        early_stopping_rounds=30
                    )
                except TypeError:
                    mdl.fit(Xtr_k, ytr_k)
                yhat_k = mdl.predict(Xte_k)
                scores.append(mean_absolute_error(yte_k, yhat_k))
            cv_mae = float(np.mean(scores)) if scores else float("nan")

        return {"cv_mae": cv_mae, "holdout_mae": holdout_mae, "r2": r2}

def plot_importances(model: XGBRegressor, X: pd.DataFrame, top_n: int = 25):
    """Importancias de XGBoost (ligero)."""
    fig, ax = plt.subplots(figsize=(8,5))
    try:
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        ax.barh(np.array(X.columns)[idx][::-1], np.array(importances)[idx][::-1])
        ax.set_title("Importancia de características (XGBoost)")
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Importancias no disponibles: {e}")

# -------------------------
# ClubFit (grupos + vecinos + minutos ponderados)
# -------------------------
def _pick_minutes_column(df: pd.DataFrame) -> Optional[str]:
    """Intenta detectar la columna de minutos para ponderar centroides."""
    cand = [c for c in df.columns if "minute" in c.lower() or c.lower() in ("min","mins","minutes","90s","minutes played")]
    if not cand:
        return None
    # Prioriza 'Minutes' exacto si existe
    for k in ("Minutes","minutes","MIN","Min","mins","90s"):
        if k in df.columns:
            return k
    return cand[0]

class ClubFit:
    """
    Fit-score: similitud coseno entre PCA del jugador y centroides del club por GRUPO
    (Goalkeeper, Fullback, Defender, Midfielder, Wingers, Forward), calculados dentro del timeframe.
    Centroides ponderados por minutos si existe una columna de minutos.
    position_match: 1.0 exacto / 0.7 vecino / 0.2 sin matching.
    signing_score = position_match * (0.55*fit + 0.35*z(pred)) - 0.10*z(Market value)
    """
    def __init__(self, df_full: pd.DataFrame, spaces: Dict[str, pd.DataFrame]):
        self.df = df_full.copy()
        self.latent = spaces["pca"]
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
        if "Team" not in self.df.columns:
            return d
        for club, g in self.df.groupby("Team"):
            bucket: Dict[str, List[int]] = {}
            for i, row in g.iterrows():
                for grp in row.get("_GroupPos", []):
                    bucket.setdefault(grp, []).append(i)
            for grp, idxs in bucket.items():
                if not idxs:
                    continue
                # ponderación por minutos si hay columna válida
                if self.minutes_col and self.minutes_col in self.df.columns:
                    w = self.df.loc[idxs, self.minutes_col].astype(float).fillna(0).to_numpy()
                    if w.sum() > 0:
                        mat = self.latent.loc[idxs].to_numpy()
                        d[(club, grp)] = np.average(mat, axis=0, weights=w)
                        continue
                # fallback: media simple
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
        if has_exact:
            return best_exact, 1.0
        if has_neighbor:
            return best_neighbor, 0.7
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

        # filtros por grupos requeridos
        if required_groups:
            if strict_groups:
                df = df[df["_GroupPos"].apply(lambda lst: any(g in required_groups for g in lst))]
            else:
                def any_close(lst):
                    for g in lst:
                        if g in required_groups:
                            return True
                        if GROUP_NEIGHBORS.get(g, set()) & required_groups:
                            return True
                    return False
                df = df[df["_GroupPos"].apply(any_close)]

        return df

    def rank_signings(self, target_club: str, top_k: int = CFG.TOP_K_RECS,
                      required_groups: Optional[set] = None, strict_groups: bool = False) -> pd.DataFrame:
        df = self._with_scores(target_club, required_groups, strict_groups)
        df = df[df["Team"] != target_club]  # excluir jugadores del club destino
        cols = ["Player","Team","Position","_GroupPos","Age",CFG.TARGET,"predicted_value","fit_score","position_match","signing_score"]
        cols = [c for c in cols if c in df.columns]
        return df.sort_values("signing_score", ascending=False).head(top_k)[cols]

    def eval_player_in_club(self, player_name: str, target_club: str) -> pd.DataFrame:
        df_scored = self._with_scores(target_club)
        board = df_scored[df_scored["Team"] != target_club].sort_values("signing_score", ascending=False).reset_index(drop=True)
        cand = df_scored[df_scored["Player"].str.lower()==str(player_name).lower()]
        if cand.empty:
            raise ValueError(f"Jugador '{player_name}' no encontrado en el timeframe filtrado.")
        out_cols = [c for c in ["Player","Team","Position","_GroupPos","Age",CFG.TARGET,"predicted_value","fit_score","position_match"] if c in df_scored.columns]
        out = cand[out_cols].copy()
        out["target_club"] = target_club
        mask = board["Player"].str.lower()==str(player_name).lower()
        out["rank_for_target"] = int(board.index[mask][0]) + 1 if mask.any() else -1
        return out

# -------------------------
# Sidebar (inputs)
# -------------------------
with st.sidebar:
    st.header("① Carga de archivos")
    uploaded_files = st.file_uploader("Sube uno o varios (.xlsx / .csv)", type=["xlsx","csv"], accept_multiple_files=True)

    st.header("② Filtro timeframe")
    st.caption("Se aplica ANTES de entrenar el modelo y calcular encaje/ranking.")

    st.header("③ Clave OpenAI (RAG)")
    default_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    api_key = st.text_input("OPENAI_API_KEY", type="password", value=default_key, help="Usa Secrets en producción; aquí puedes pegarla temporalmente.")

    st.header("④ Parámetros")
    fast_mode = st.checkbox("Modo rápido (entrena más veloz, sin CV)", value=True)
    top_k = st.number_input("Top K (ranking)", min_value=5, max_value=200, value=CFG.TOP_K_RECS, step=1)
    show_importances = st.checkbox("Mostrar importancias del modelo", value=True)

st.title("⚽ Scouting Tool — Avanzada")
st.caption("Multi-archivo · Timeframe · ML · Fit por grupos · Vecinos de rol · Minutos ponderados · Ranking · RAG")

if not uploaded_files:
    st.info("Sube al menos un archivo para comenzar.")
    st.stop()

# -------------------------
# Unir archivos
# -------------------------
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

# -------------------------
# Controles timeframe
# -------------------------
st.subheader("Vista previa y timeframe")
st.dataframe(df_raw.head(15), use_container_width=True)

col1, col2 = st.columns(2)
date_col = season_col = None
date_range = seasons_selected = None

with col1:
    chosen_date_col = st.selectbox("Columna de fecha (opcional)", options=["(ninguna)"]+date_cols, index=0)
    if chosen_date_col != "(ninguna)":
        series_dt = pd.to_datetime(df_raw[chosen_date_col], errors="coerce").dropna()
        if series_dt.empty:
            st.warning("La columna de fecha seleccionada no tiene fechas válidas.")
        else:
            min_dt, max_dt = series_dt.min(), series_dt.max()
            date_range = st.slider("Rango de fechas", min_value=min_dt.to_pydatetime(), max_value=max_dt.to_pydatetime(),
                                   value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()))
            date_col = chosen_date_col

with col2:
    chosen_season_col = st.selectbox("Columna de temporada/año (opcional)", options=["(ninguna)"]+season_cols, index=0)
    if chosen_season_col != "(ninguna)":
        uniq = sorted([x for x in df_raw[chosen_season_col].dropna().unique().tolist()])
        seasons_selected = st.multiselect("Temporadas / Años", options=uniq, default=uniq)
        season_col = chosen_season_col

st.caption("Todo se calcula **sólo** con datos dentro del timeframe elegido (Team within selected timeframe).")

# -------------------------
# Aplicar timeframe y preparar datos
# -------------------------
df_f = apply_timeframe(df_raw, date_col, date_range, season_col, seasons_selected)
if df_f.empty:
    st.warning("El filtro de timeframe dejó el dataset vacío. Ajusta los filtros.")
    st.stop()

try:
    df_for_model, X, y, num_cols = select_numeric(df_f)
except Exception as e:
    st.error(f"Error preparando datos: {e}")
    st.stop()

# === entrenamiento con progreso (cacheado) ===
@st.cache_resource(show_spinner=False)
def train_pipeline(df_in: pd.DataFrame, X: pd.DataFrame, y: pd.Series, fast_mode: bool):
    status = st.empty()
    prog = st.progress(0)
    status.info("Inicializando modelo…")
    prog.progress(10)

    model = MarketValueModel(fast_mode=fast_mode)

    status.info("Entrenando con early stopping…")
    prog.progress(55)
    metrics = model.train(X, y)

    status.info("Generando predicciones…")
    prog.progress(80)
    X_np_all = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
    y_hat = model.model.predict(X_np_all)

    status.info("Calculando espacios latentes (PCA)…")
    spaces = LatentSpaces().fit_transform(X)
    prog.progress(100)
    status.success("Entrenamiento finalizado.")

    return model, metrics, y_hat, spaces

st.subheader("Entrenamiento del modelo (dentro del timeframe)")
model, metrics, y_hat, spaces = train_pipeline(df_for_model, X, y, fast_mode)

# Ensamble predicciones
df_pred = df_for_model.copy()
df_pred["predicted_value"] = y_hat
df_pred["delta_pred_real"] = df_pred["predicted_value"] - df_pred[CFG.TARGET]

# Métricas
m1, m2, m3 = st.columns(3)
m1.markdown(f"<div class='metric-card tight'><b>CV MAE</b><br>{'—' if np.isnan(metrics['cv_mae']) else f'{metrics['cv_mae']:,.0f}'}</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card tight'><b>Holdout MAE</b><br>{metrics['holdout_mae']:,.0f}</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card tight'><b>R²</b><br>{metrics['r2']:.3f}</div>", unsafe_allow_html=True)

if show_importances:
    st.markdown("**Importancia de variables**")
    plot_importances(model.model, X, top_n=25)

# Fit scorer
fitter = ClubFit(df_pred, spaces)

# -------------------------
# Tabs
# -------------------------
tab_pred, tab_fit, tab_rank, tab_rag, tab_help = st.tabs(
    ["📈 Predicciones", "🎯 Encaje jugador→club", "🏟 Mejores incorporaciones", "💬 Chat (RAG)", "📘 Glosario"]
)

with tab_pred:
    st.subheader("Predicciones de valor de mercado")
    cols_show = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","delta_pred_real"] if c in df_pred.columns]
    st.dataframe(display_money(df_pred[cols_show].head(500)), use_container_width=True)
    st.caption("Se muestran 500 filas por rendimiento. Descarga para ver completo.")
    st.download_button("Descargar predicciones (CSV)",
                       data=df_pred[cols_show].to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")

with tab_fit:
    st.subheader("Encaje de un jugador en un club")
    if ("Player" not in df_pred.columns) or ("Team" not in df_pred.columns):
        st.error("Necesito columnas 'Player' y 'Team'.")
    else:
        players = sorted(df_pred["Player"].dropna().unique().tolist())
        clubs = sorted(df_pred["Team"].dropna().unique().tolist())
        c1, c2 = st.columns(2)
        with c1:
            player_name = st.selectbox("Jugador", options=players)
        with c2:
            target_club = st.selectbox("Club destino", options=clubs)
        if st.button("Evaluar encaje"):
            try:
                res = fitter.eval_player_in_club(player_name, target_club)
                st.dataframe(display_money(res), use_container_width=True)
                st.markdown(
                    """
                    <div class='small-note'>
                    <b>_GroupPos</b>: grupo(s) posicional(es) normalizados del jugador (Goalkeeper, Fullback, Defender, Midfielder, Wingers, Forward).<br>
                    <b>fit_score (0–1)</b>: similitud del jugador con el perfil del club por grupo (PCA), ponderado por minutos en los centroides.<br>
                    <b>position_match</b>: 1.0 (grupo exacto), 0.7 (grupo vecino), 0.2 (sin matching).<br>
                    <b>rank_for_target</b>: puesto del jugador en el ranking del club (1 = mejor), excluyendo jugadores que ya están en el club.
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
        target_club_r = st.selectbox("Club destino", options=clubs, key="rank_club_sel")

        # Filtros por grupos y modo estricto
        req_groups = st.multiselect("Grupos requeridos (opcional)", options=ALL_GROUPS, default=[])
        strict_groups = st.checkbox("Modo estricto por grupo", value=False,
                                    help="Si está activo, sólo se listan jugadores cuyo grupo esté exactamente en la selección. Si no, también cuentan vecinos.")

        if st.button("Calcular ranking"):
            try:
                req_set = set(req_groups) if req_groups else None
                board = fitter.rank_signings(target_club_r, top_k=int(top_k),
                                             required_groups=req_set, strict_groups=strict_groups)
                st.dataframe(display_money(board), use_container_width=True)
                st.caption(
                    "Centroides por grupo ponderados por minutos (si hay columna de minutos). "
                    "Se consideran grupos vecinos para similitud de estilo."
                )
                st.download_button(f"Descargar ranking {target_club_r} (CSV)",
                                   data=board.to_csv(index=False).encode("utf-8"),
                                   file_name=f"ranking_{target_club_r}.csv", mime="text/csv")
            except Exception as e:
                st.error(str(e))

with tab_rag:
    st.subheader("Chat con RAG (TF-IDF, OpenAI)")
    if not LANGCHAIN_AVAILABLE:
        st.info("RAG deshabilitado: falta langchain-openai. Revisa requirements.txt.")
    else:
        q_default = "Dame 5 posibles fichajes con alto fit y buen precio para {club}"
        question = st.text_input("Pregunta", value=q_default)
        clubs = sorted(df_pred["Team"].dropna().unique().tolist()) if "Team" in df_pred.columns else []
        club_for_ctx = st.selectbox("Club para enriquecer contexto (opcional)", options=clubs) if clubs else ""
        if st.button("Preguntar"):
            if not api_key:
                st.warning("Pega tu OPENAI_API_KEY en la barra lateral o define st.secrets/ENV.")
            else:
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    # Construimos corpus contextual (agregando fit/signing si hay club)
                    df_ctx = df_pred.copy()
                    if club_for_ctx:
                        tmp = fitter.rank_signings(club_for_ctx, top_k=len(df_ctx))
                        if "Player" in df_ctx.columns:
                            df_ctx = df_ctx.merge(tmp[["Player","fit_score","position_match","signing_score"]], on="Player", how="left")
                    # Textos para TF-IDF
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
                    # Retrieval TF-IDF
                    ctx_q = question.replace("{club}", club_for_ctx) if club_for_ctx else question
                    vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
                    Xtf = vect.fit_transform(docs + [ctx_q])
                    sims = cosine_similarity(Xtf[-1], Xtf[:-1]).ravel()
                    top_idx = np.argsort(sims)[::-1][:6]
                    context_text = "\n\n---\n\n".join([docs[i] for i in top_idx])
                    # Chat
                    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    msgs = [
                        SystemMessage(content="Eres analista de scouting. Responde con datos del contexto y recomendaciones accionables."),
                        HumanMessage(content=f"Contexto:\n{context_text}\n\nPregunta: {ctx_q}")
                    ]
                    answer = chat.invoke(msgs).content
                    st.write(answer)
                except Exception as e:
                    st.error(f"RAG error: {e}")

with tab_help:
    st.subheader("Glosario y criterios")
    st.markdown(
        """
        - **Team within selected timeframe**: todo se calcula usando sólo registros que caen en tu **rango de fechas** y/o **temporadas** seleccionadas.  
        - **Market value**: valor de mercado real (del dataset).  
        - **predicted_value**: valor estimado por el modelo (XGBoost) entrenado con variables numéricas del timeframe.  
        - **delta_pred_real**: `predicted_value - Market value` (positivo → posible infravaloración).  
        - **_GroupPos**: grupo(s) posicional(es) normalizados con tu diccionario: Goalkeeper, Fullback, Defender, Midfielder, Wingers, Forward.  
        - **Centroides ponderados**: si hay columna de minutos (p. ej., *Minutes*, *Min*, *90s*), los centroides por grupo del club se ponderan por esos minutos.  
        - **fit_score (0–1)**: similitud del vector PCA del jugador respecto al/los centroids del club en su grupo (y vecinos si hace falta).  
        - **position_match**: 1.0 si hay centroides en el grupo exacto; 0.7 si sólo en grupos **vecinos** (estilo compatible); 0.2 si no hay matching.  
        - **signing_score**: `position_match * (0.55*fit + 0.35*z(predicted_value)) - 0.10*z(Market value)`; favorece encaje y proyección, penaliza costo actual.  
        - **rank_for_target**: puesto del jugador en el ranking del club (1 = mejor), **excluyendo** jugadores ya pertenecientes al club destino.  
        - **Top K**: número de candidatos a mostrar (no es métrica).  
        - **RAG**: chat que responde con base en el contenido de tu dataset usando TF-IDF + OpenAI.  
        """.strip()
    )

st.success("Listo. Carga archivos, filtra el timeframe y explora predicciones, encaje y ranking. 👌")
