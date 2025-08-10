# app.py
# ======================================================================
# ⚽ SCOUTING TOOL PRO — Streamlit (multi-archivo, timeframe, ML, fit, ranking, RAG + “Libro de pases”)
# ======================================================================
# Incluye:
# 1) Ranking de mejores incorporaciones por club (EXCLUYE jugadores del club) + resumen por jugador (LLM)
# 2) Encaje jugador→club con resumen “pros & cons” (LLM)
# 3) Mejores destinos para un jugador (fit + estilo del club) con resumen (LLM)
# 4) “Libro de pases”: tablero de shortlist, filtros por posición/edad/presupuesto y exportación
# 5) Timeframe (fecha/temporada), carga multi-archivo (XLSX/CSV), modelo ML XGBoost para Market value,
#    PCA para estilos de equipo, TF-IDF RAG contextual, gráficos modernos (radar & barras)
#
# *** NOTA DE SEGURIDAD ***
# A petición tuya, se incluye la OPENAI_API_KEY directamente en el código.
# En producción NO es recomendable. Aquí lo hacemos porque lo solicitaste explícitamente.
# ======================================================================

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBRegressor

# LLM (OpenAI vía LangChain)
try:
    from langchain_openai import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LLM_OK = True
except Exception:
    LLM_OK = False

# =========================
# ⚙️ Configuración & estilo
# =========================
st.set_page_config(page_title="⚽ Scouting Tool PRO", layout="wide", page_icon="⚽")
st.markdown("""
<style>
.small-note { color:#6b7280; font-size:0.92em; }
.metric-card { background:#0f172a10; border-radius:12px; padding:10px 14px; }
.tight { line-height:1.15; }
.hr { height:1px; background:linear-gradient(90deg,#0000,#94a3b8,#0000); border:0; margin:8px 0 16px;}
.section { padding:10px 12px; background:#0ea5e91a; border-radius:12px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; background:#0ea5e91a; color:#0369a1; font-size:12px; margin-left:6px;}
.warn { color:#b45309 } .ok { color:#059669 } .bad { color:#dc2626 }
</style>
""", unsafe_allow_html=True)

# 🔑 API KEY embebida (a tu solicitud). En producción NO se recomienda.
DEFAULT_OPENAI_KEY = "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"

@dataclass
class Config:
    TARGET: str = "Market value"
    ID_COLS: Tuple[str, ...] = ("Player","Team","Position","Age","Contract expires")
    POS_COL: str = "Position"
    RANDOM_STATE: int = 42
    PCA_COMPONENTS: int = 12
    TOP_K_RECS: int = 20
    RADAR_STATS: Tuple[str, ...] = (
        "Goals per 90","Assists per 90","xG per 90","xA per 90",
        "Shots per 90","Key passes per 90","Dribbles per 90",
        "Tackles per 90","Interceptions per 90","Progressive passes per 90"
    )

CFG = Config()
np.random.seed(CFG.RANDOM_STATE)

# =========================
# 🔧 Utilidades
# =========================
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
    df = df.copy()
    for col in [CFG.TARGET, "predicted_value", "delta_pred_real"]:
        if col in df.columns:
            df[col] = df[col].apply(format_millions)
    return df

def auto_pick_sheet(xls: pd.ExcelFile, filename: str, required={"Player","Team","Market value"}) -> str:
    cands = [s for s in xls.sheet_names if any(k in s.lower() for k in ("search","result","players","jugadores","sheet","datos"))]
    if cands: return cands[0]
    best, best_score = None, -1
    for s in xls.sheet_names:
        try:
            preview = pd.read_excel(filename, sheet_name=s, nrows=5)
            cols_norm = {str(c).strip() for c in preview.columns}
            score = len(required.intersection(cols_norm))
            if score > best_score: best, best_score = s, score
        except Exception:
            pass
    return best or xls.sheet_names[0]

@st.cache_data(show_spinner=False)
def read_any(file_bytes, filename: str, sheet_hint: Optional[str]) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        xls = pd.ExcelFile(file_bytes)
        chosen = sheet_hint or auto_pick_sheet(xls, filename)
        df = pd.read_excel(file_bytes, sheet_name=chosen)
        df["__source_file__"] = filename; df["__source_sheet__"] = chosen
    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(file_bytes)
        df["__source_file__"] = filename; df["__source_sheet__"] = ""
    else:
        raise ValueError("Formato no soportado. Usa .xlsx o .csv")
    return normalize_columns(df)

def detect_time_columns(df: pd.DataFrame):
    date_cols, season_cols = [], []
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in ("date","fecha","match")):
            try:
                pd.to_datetime(df[c], errors="raise"); date_cols.append(c)
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

def select_numeric(df: pd.DataFrame):
    if CFG.TARGET not in df.columns:
        raise ValueError(f"No se encontró la columna objetivo '{CFG.TARGET}'.")
    df2 = df[df[CFG.TARGET].notna()].copy()
    df2[CFG.TARGET] = pd.to_numeric(df2[CFG.TARGET], errors="coerce")
    num_cols = [c for c in df2.columns if (np.issubdtype(df2[c].dtype, np.number) and c != CFG.TARGET)]
    if not num_cols:
        raise ValueError("No hay columnas numéricas suficientes para entrenar.")
    X = df2[num_cols].fillna(0.0)
    y = df2[CFG.TARGET].astype(float)
    return df2, X, y, num_cols

# ====== Modelo ML rápido con early stopping
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
        self.model.fit(Xtr, ytr,
                       eval_set=[(Xte, yte)], eval_metric="mae",
                       verbose=False, early_stopping_rounds=50)
        y_pred = self.model.predict(Xte)
        return {"cv_mae": np.nan,
                "holdout_mae": float(mean_absolute_error(yte, y_pred)),
                "r2": float(r2_score(yte, y_pred))}
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

def plot_importances(model: XGBRegressor, X: pd.DataFrame, top_n: int = 20):
    fig, ax = plt.subplots(figsize=(8,5))
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    ax.barh(np.array(X.columns)[idx][::-1], np.array(importances)[idx][::-1])
    ax.set_title("Importancia de características (XGBoost)")
    plt.tight_layout()
    st.pyplot(fig)

# ====== Estilos de club & encaje (PCA)
class ClubFit:
    """
    fit_score: similitud coseno entre vector PCA del jugador y centroides del club destino
    por cada posición del jugador (dentro del timeframe).
    signing_score = 0.55*fit + 0.35*z(predicted_value) - 0.10*z(Market value)
    """
    def __init__(self, df_full: pd.DataFrame, spaces: Dict[str, pd.DataFrame]):
        self.df = df_full.copy()
        self.latent = spaces["pca"]
        self.centroids = self._compute_centroids()
    @staticmethod
    def _norm_positions(pos_str: str) -> List[str]:
        if pd.isna(pos_str): return []
        return [p.strip() for p in str(pos_str).split(",") if p.strip()]
    def _compute_centroids(self) -> Dict[tuple, np.ndarray]:
        d = {}
        if "Team" not in self.df.columns or CFG.POS_COL not in self.df.columns:
            return d
        for club, g in self.df.groupby("Team"):
            pos_map = {}
            for i, row in g.iterrows():
                for p in self._norm_positions(row.get(CFG.POS_COL, "")):
                    pos_map.setdefault(p, []).append(i)
            for p, idxs in pos_map.items():
                if idxs: d[(club, p)] = self.latent.loc[idxs].mean(axis=0).values
        return d
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a,b)/(na*nb))
    def fit_score(self, idx: int, target_club: str) -> float:
        if ("Team" not in self.df.columns) or (CFG.POS_COL not in self.df.columns) or (target_club not in self.df["Team"].unique()):
            return 0.0
        vec = self.latent.loc[idx].values
        positions = self._norm_positions(self.df.loc[idx].get(CFG.POS_COL, ""))
        sims = []
        for p in positions:
            key = (target_club, p)
            if key in self.centroids:
                sims.append(self._cosine(vec, self.centroids[key]))
        return float(np.mean(sims)) if sims else 0.0
    def _with_scores(self, target_club: str) -> pd.DataFrame:
        df = self.df.copy()
        if "predicted_value" not in df or CFG.TARGET not in df:
            raise ValueError("Faltan columnas 'predicted_value' o target.")
        df["fit_score"] = [self.fit_score(i, target_club) for i in df.index]
        for col in ["predicted_value", CFG.TARGET]:
            mu, sd = df[col].mean(), df[col].std() + 1e-9
            df[f"{col}_z"] = (df[col]-mu)/sd
        df["signing_score"] = 0.55*df["fit_score"] + 0.35*df["predicted_value_z"] - 0.10*df[f"{CFG.TARGET}_z"]
        return df
    def rank_signings(self, target_club: str, top_k: int = CFG.TOP_K_RECS) -> pd.DataFrame:
        df = self._with_scores(target_club)
        df = df[df["Team"] != target_club]  # excluir jugadores del propio club
        cols = ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","fit_score","signing_score"]
        cols = [c for c in cols if c in df.columns]
        return df.sort_values("signing_score", ascending=False).head(top_k)[cols]
    def eval_player_in_club(self, player_name: str, target_club: str) -> pd.DataFrame:
        df_scored = self._with_scores(target_club)
        board = df_scored[df_scored["Team"] != target_club].sort_values("signing_score", ascending=False).reset_index(drop=True)
        cand = df_scored[df_scored["Player"].str.lower()==str(player_name).lower()]
        if cand.empty:
            raise ValueError(f"Jugador '{player_name}' no encontrado en el timeframe.")
        out_cols = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value"] if c in df_scored.columns]
        out = cand[out_cols].copy()
        out["target_club"] = target_club
        out["fit_score"] = float(cand.iloc[0]["fit_score"]) if "fit_score" in cand.columns else 0.0
        mask = board["Player"].str.lower()==str(player_name).lower()
        out["rank_for_target"] = int(board.index[mask][0]) + 1 if mask.any() else -1
        return out
    def best_destinations_for_player(self, player_name: str, top_k: int = 10) -> pd.DataFrame:
        """Ranking de clubes destino para un jugador según fit y signing_score."""
        if "Player" not in self.df.columns or "Team" not in self.df.columns:
            raise ValueError("Faltan columnas Player/Team.")
        cand = self.df[self.df["Player"].str.lower()==str(player_name).lower()]
        if cand.empty:
            raise ValueError(f"Jugador '{player_name}' no encontrado.")
        i = cand.index[0]
        clubs = sorted(self.df["Team"].dropna().unique().tolist())
        rows = []
        tmp = self.df.copy()
        # Aseguramos z-scores globales para costos/valor (para gamma)
        for col in ["predicted_value", CFG.TARGET]:
            mu, sd = tmp[col].mean(), tmp[col].std() + 1e-9
            tmp[f"{col}_z"] = (tmp[col]-mu)/sd
        for club in clubs:
            if club == cand.iloc[0]["Team"]:  # opcional: permitir considerar su propio club
                pass
            vec = self.latent.loc[i].values
            fit = 0.0
            pos = ClubFit._norm_positions(str(cand.iloc[0].get(CFG.POS_COL,"")))
            sims = []
            for p in pos:
                key = (club, p)
                if key in self.centroids:
                    sims.append(self._cosine(vec, self.centroids[key]))
            fit = float(np.mean(sims)) if sims else 0.0
            pv_z = float(tmp.loc[i,"predicted_value_z"]) if "predicted_value_z" in tmp.columns else 0.0
            mv_z = float(tmp.loc[i,f"{CFG.TARGET}_z"]) if f"{CFG.TARGET}_z" in tmp.columns else 0.0
            signing = 0.55*fit + 0.35*pv_z - 0.10*mv_z
            rows.append({"Club": club, "fit_score": fit, "signing_score": signing})
        res = pd.DataFrame(rows).sort_values(["signing_score","fit_score"], ascending=False)
        return res.head(top_k)

# ====== TF-IDF RAG helpers
def build_docs_for_rag(df_ctx: pd.DataFrame) -> List[str]:
    docs = []
    for _, r in df_ctx.iterrows():
        lines = []
        for c in CFG.ID_COLS:
            if c in r: lines.append(f"{c}: {r[c]}")
        for k in ["predicted_value","fit_score","signing_score","Goals per 90","Assists per 90","xG per 90","xA per 90"]:
            if k in df_ctx.columns and pd.notna(r.get(k, np.nan)):
                lines.append(f"{k}: {r[k]}")
        if CFG.TARGET in r:
            lines.append(f"{CFG.TARGET}: {r.get(CFG.TARGET, np.nan)}")
        docs.append("\n".join(lines))
    return docs

def retrieve_tfidf(docs: List[str], query: str, k: int = 6) -> str:
    vect = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    Xtf = vect.fit_transform(docs + [query])
    sims = cosine_similarity(Xtf[-1], Xtf[:-1]).ravel()
    top_idx = np.argsort(sims)[::-1][:k]
    return "\n\n---\n\n".join([docs[i] for i in top_idx])

def llm_summary(prompt: str) -> str:
    if not LLM_OK:
        return "LLM no disponible (faltan dependencias)."
    key = DEFAULT_OPENAI_KEY  # 🔐 usando la key embebida a petición
    os.environ["OPENAI_API_KEY"] = key
    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    msgs = [
        SystemMessage(content="Eres analista senior de scouting. Sé concreto, accionable y claro."),
        HumanMessage(content=prompt)
    ]
    return chat.invoke(msgs).content

# ====== Radar moderno (matplotlib polar)
def plot_radar(player_row: pd.Series, team_ref_df: pd.DataFrame, title: str):
    stats = [s for s in CFG.RADAR_STATS if s in team_ref_df.columns]
    if not stats:
        st.caption("No hay columnas ‘per 90’ suficientes para el radar.")
        return
    ref = team_ref_df[stats].astype(float)
    # Normalizamos 0–1 con percentiles robustos
    p05, p95 = ref.quantile(0.05), ref.quantile(0.95)
    def norm_series(x): return np.clip((x - p05) / (p95 - p05 + 1e-9), 0, 1)
    ref_n = ref.apply(norm_series, axis=0)
    # Perfil del equipo (mediana por posición del jugador si aplica)
    team_profile = ref_n.median(axis=0)
    # Jugador (si faltan valores, los imputamos con mediana)
    pj = player_row.reindex(stats).astype(float)
    pj = pj.fillna(ref[stats].median())
    pj_n = norm_series(pj)

    labels = stats
    values_team = team_profile.values.tolist()
    values_player = pj_n.values.tolist()
    # Cierre del círculo
    labels += labels[:1]
    values_team += values_team[:1]
    values_player += values_player[:1]

    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(6.2,6.2))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values_team, linewidth=2, linestyle='dashed')
    ax.fill(angles, values_team, alpha=0.08)
    ax.plot(angles, values_player, linewidth=2)
    ax.fill(angles, values_player, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title(title, fontweight="bold")
    st.pyplot(fig)

# =========================
# 🧭 Sidebar (inputs)
# =========================
with st.sidebar:
    st.header("① Carga de archivos")
    files = st.file_uploader("Sube uno o varios (.xlsx / .csv)", type=["xlsx","csv"], accept_multiple_files=True)
    sheet_hint = st.text_input("Hoja Excel (opcional)")

    st.header("② Timeframe")
    st.caption("Se aplica antes de entrenar y calcular fit/ranking.")

    st.header("③ Parámetros")
    top_k = st.number_input("Top K (ranking)", min_value=5, max_value=200, value=CFG.TOP_K_RECS, step=1)
    fast_mode = st.checkbox("Modo rápido (recomendado)", value=True)

st.title("⚽ Scouting Tool PRO")
st.caption("Multi-archivo · Timeframe · ML · Fit por club · Ranking · RAG · Libro de pases")

if not files:
    st.info("Sube al menos un archivo para comenzar.")
    st.stop()

# =========================
# 📥 Carga & vista previa
# =========================
frames = []
for f in files:
    try:
        df_tmp = read_any(f, f.name, sheet_hint or None)
        frames.append(df_tmp)
    except Exception as e:
        st.error(f"Error leyendo {f.name}: {e}")
        st.stop()

df_raw = pd.concat(frames, ignore_index=True, sort=False)
date_cols, season_cols = detect_time_columns(df_raw)

st.subheader("Vista previa y timeframe")
st.dataframe(df_raw.head(15), use_container_width=True)

c1, c2 = st.columns(2)
date_col = season_col = None
date_range = seasons_selected = None
with c1:
    chosen_date = st.selectbox("Columna de fecha (opcional)", ["(ninguna)"]+date_cols)
    if chosen_date != "(ninguna)":
        series_dt = pd.to_datetime(df_raw[chosen_date], errors="coerce").dropna()
        if series_dt.empty:
            st.warning("La columna elegida no contiene fechas válidas.")
        else:
            min_dt, max_dt = series_dt.min(), series_dt.max()
            date_range = st.slider("Rango de fechas",
                                   min_value=min_dt.to_pydatetime(),
                                   max_value=max_dt.to_pydatetime(),
                                   value=(min_dt.to_pydatetime(), max_dt.to_pydatetime()))
            date_col = chosen_date
with c2:
    chosen_season = st.selectbox("Columna de temporada/año (opcional)", ["(ninguna)"]+season_cols)
    if chosen_season != "(ninguna)":
        uniq = sorted([x for x in df_raw[chosen_season].dropna().unique().tolist()])
        seasons_selected = st.multiselect("Temporadas/Años", options=uniq, default=uniq)
        season_col = chosen_season

st.caption("Todo se calcula solo con datos dentro del timeframe seleccionado.")

# =========================
# ✂️ Aplicar timeframe & ML
# =========================
df_f = apply_timeframe(df_raw, date_col, date_range, season_col, seasons_selected)
if df_f.empty:
    st.warning("El filtro dejó el dataset vacío. Ajusta el timeframe.")
    st.stop()

try:
    df_model, X, y, num_cols = select_numeric(df_f)
except Exception as e:
    st.error(f"Error preparando datos: {e}")
    st.stop()

@st.cache_resource(show_spinner=False)
def train_pipeline(X: pd.DataFrame, y: pd.Series, fast: bool):
    status = st.empty(); prog = st.progress(0)
    status.info("Inicializando modelo…"); prog.progress(8)
    model = MarketValueModel(fast_mode=fast)
    status.info("Entrenando (early stopping)…"); prog.progress(55)
    metrics = model.train(X, y)
    status.info("Generando predicciones…"); prog.progress(80)
    preds = model.predict(X)
    status.info("Calculando PCA (estilos de club)…")
    scaler = StandardScaler(); Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    pca = PCA(n_components=CFG.PCA_COMPONENTS, random_state=CFG.RANDOM_STATE)
    latent = pd.DataFrame(pca.fit_transform(Xs), index=X.index)
    prog.progress(100); status.success("Entrenamiento finalizado.")
    return model, metrics, preds, {"scaled": Xs, "pca": latent}

st.subheader("Entrenamiento")
model, metrics, y_hat, spaces = train_pipeline(df_for_model, X, y, fast_mode)

df_pred = df_model.copy()
df_pred["predicted_value"] = y_hat
df_pred["delta_pred_real"] = df_pred["predicted_value"] - df_pred[CFG.TARGET]

m1, m2, m3 = st.columns(3)
m1.markdown(f"<div class='metric-card tight'><b>CV MAE</b><br>{'—' if np.isnan(metrics['cv_mae']) else f'{metrics['cv_mae']:,.0f}'}</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card tight'><b>Holdout MAE</b><br>{metrics['holdout_mae']:,.0f}</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card tight'><b>R²</b><br>{metrics['r2']:.3f}</div>", unsafe_allow_html=True)

st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

# Instancia de encaje
fitter = ClubFit(df_pred, spaces)

# =========================
# 📊 Tabs principales
# =========================
tab_rank, tab_fit, tab_dest, tab_book, tab_pred, tab_help = st.tabs([
    "🏟 Mejores incorporaciones", "🎯 Encaje jugador→club", "🧭 Mejores destinos por jugador",
    "📒 Libro de pases", "📈 Predicciones", "📘 Glosario"
])

# 1) Mejores incorporaciones + resumen LLM + radar
with tab_rank:
    st.subheader("Ranking de mejores incorporaciones (excluye jugadores del club)")
    clubs = sorted(df_pred["Team"].dropna().unique().tolist()) if "Team" in df_pred.columns else []
    if not clubs:
        st.error("Falta la columna 'Team'.")
    else:
        colA, colB = st.columns([1,3])
        with colA:
            club_sel = st.selectbox("Club destino", options=clubs)
            k = st.slider("Top K", 5, 60, CFG.TOP_K_RECS)
            run = st.button("Calcular ranking")
        with colB:
            if club_sel and run:
                board = fitter.rank_signings(club_sel, top_k=k)
                st.dataframe(display_money(board), use_container_width=True)
                st.download_button(f"Descargar ranking {club_sel} (CSV)",
                                   data=board.to_csv(index=False).encode("utf-8"),
                                   file_name=f"ranking_{club_sel}.csv", mime="text/csv")
                st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

                # Resumen por jugador + radar (si hay stats)
                for _, row in board.iterrows():
                    player = row["Player"]; team_from = row["Team"]
                    st.markdown(f"**{player}** <span class='badge'>desde {team_from} → {club_sel}</span>", unsafe_allow_html=True)
                    # Radar: comparamos jugador vs perfil del club (mismos jugadores del club_sel por posición si existen)
                    try:
                        pos = str(row.get(CFG.POS_COL,""))
                        club_ref = df_pred[(df_pred["Team"]==club_sel)]
                        if pos:
                            # si hay posición, usamos jugadores del club en esa posición
                            club_ref = club_ref[club_ref[CFG.POS_COL].fillna("").str.contains(pos.split(",")[0].strip(), na=False)]
                        if club_ref.empty:
                            club_ref = df_pred[df_pred["Team"]==club_sel]
                        player_row = df_pred[df_pred["Player"]==player].iloc[0]
                        plot_radar(player_row, club_ref, f"Radar — {player} vs {club_sel}")
                    except Exception:
                        st.caption("Radar no disponible para este jugador.")

                    # Resumen LLM (por qué sería buena incorporación)
                    ctx_cols = ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","fit_score","signing_score"]
                    # Añadimos fit/signing del board (ya calculado)
                    joined = df_pred.merge(board[["Player","fit_score","signing_score"]], on="Player", how="left")
                    docs = build_docs_for_rag(joined)
                    context = retrieve_tfidf(docs, f"Mejor incorporación para {club_sel}", k=6)
                    prompt = f"""Contexto:\n{context}\n\nEscribe un resumen ejecutivo (80-120 palabras) explicando por qué {player} sería una buena incorporación para {club_sel}. Considera fit_score, signing_score, edad, rol/posición, y relación valor estimado vs valor actual. Menciona 1 riesgo a vigilar."""
                    summary = llm_summary(prompt)
                    st.markdown(f"<div class='small-note'>{summary}</div>", unsafe_allow_html=True)
                    st.markdown("<hr class='hr'/>", unsafe_allow_html=True)

# 2) Encaje jugador→club + pros/cons
with tab_fit:
    st.subheader("Encaje de un jugador en un club (con pros & cons)")
    if not {"Player","Team"}.issubset(df_pred.columns):
        st.error("Se requieren columnas 'Player' y 'Team'.")
    else:
        players = sorted(df_pred["Player"].dropna().unique().tolist())
        clubs = sorted(df_pred["Team"].dropna().unique().tolist())
        c1, c2, c3 = st.columns([2,2,1])
        with c1: p_sel = st.selectbox("Jugador", options=players)
        with c2: club_t = st.selectbox("Club destino", options=clubs)
        with c3: go = st.button("Evaluar encaje")
        if go:
            try:
                res = fitter.eval_player_in_club(p_sel, club_t)
                st.dataframe(display_money(res), use_container_width=True)
                # Radar: jugador vs club destino
                try:
                    pos = str(res.iloc[0].get(CFG.POS_COL,""))
                    club_ref = df_pred[(df_pred["Team"]==club_t)]
                    if pos:
                        club_ref = club_ref[club_ref[CFG.POS_COL].fillna("").str.contains(pos.split(",")[0].strip(), na=False)]
                    if club_ref.empty:
                        club_ref = df_pred[df_pred["Team"]==club_t]
                    player_row = df_pred[df_pred["Player"]==p_sel].iloc[0]
                    plot_radar(player_row, club_ref, f"Radar — {p_sel} vs {club_t}")
                except Exception:
                    st.caption("Radar no disponible.")
                # Pros/Cons LLM
                docs = build_docs_for_rag(df_pred)
                context = retrieve_tfidf(docs, f"Encaje de {p_sel} en {club_t}", k=6)
                prompt = f"""Contexto:\n{context}\n\nAnaliza si {p_sel} es buena incorporación para {club_t}. 
Redacta: 
- 3 PROS claros (encaje táctico, edad, valor vs costo, métricas). 
- 2 CONDICIONES o RIESGOS. 
- Conclusión de 2 líneas con recomendación y nivel de prioridad (Alta/Media/Baja)."""
                summary = llm_summary(prompt)
                st.markdown(f"<div class='small-note'>{summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

# 3) Mejores destinos para un jugador (ranking de clubes) + resumen
with tab_dest:
    st.subheader("Mejores destinos para un jugador (según estilo del club)")
    if not {"Player","Team"}.issubset(df_pred.columns):
        st.error("Se requieren columnas 'Player' y 'Team'.")
    else:
        players = sorted(df_pred["Player"].dropna().unique().tolist())
        p2 = st.selectbox("Jugador", options=players, key="dest_player")
        k2 = st.slider("Top K clubes", 5, 40, 10)
        run2 = st.button("Calcular destinos")
        if run2:
            try:
                dests = fitter.best_destinations_for_player(p2, top_k=k2)
                st.dataframe(dests, use_container_width=True)
                # Resumen LLM
                ctx = build_docs_for_rag(df_pred)
                context = retrieve_tfidf(ctx, f"Destinos ideales para {p2}", k=6)
                prompt = f"""Contexto:\n{context}\n\nResume en 80-120 palabras los mejores destinos para {p2}, 
enfatizando 2-3 clubes con mayor fit_score y buen balance costo/beneficio. Señala rol esperado y un riesgo por club."""
                st.markdown(f"<div class='small-note'>{llm_summary(prompt)}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

# 4) Libro de pases (shortlist) — filtros y export
with tab_book:
    st.subheader("Libro de pases — Shortlist y filtros")
    if not {"Player","Team"}.issubset(df_pred.columns):
        st.error("Se requieren columnas 'Player' y 'Team'.")
    else:
        colf1, colf2, colf3, colf4 = st.columns(4)
        with colf1:
            pos_opts = sorted({p.strip() for s in df_pred[CFG.POS_COL].fillna("").tolist() for p in str(s).split(",") if p.strip()})
            pos_sel = st.multiselect("Posiciones", options=pos_opts, default=[])
        with colf2:
            age_min, age_max = int(df_pred["Age"].min() if "Age" in df_pred.columns else 15), int(df_pred["Age"].max() if "Age" in df_pred.columns else 45)
            age_rng = st.slider("Rango de edad", min_value=age_min, max_value=age_max, value=(age_min, age_max))
        with colf3:
            budget_m = st.number_input("Presupuesto máx. (M€)", min_value=0.0, value=50.0, step=1.0)
        with colf4:
            sort_by = st.selectbox("Ordenar por", ["signing_score","fit_score","predicted_value","Market value"])

        board = df_pred.copy()
        # Fit y signing si no existen (garantizamos)
        if "fit_score" not in board.columns or "signing_score" not in board.columns:
            # construimos un “club genérico” con centroides globales por posición para tener un orden objetivo
            pseudo_club = "GLOBAL"
            # centroides globales: agregamos Team="GLOBAL"
            tmp = board.copy(); tmp["Team"] = pseudo_club
            fitter_global = ClubFit(tmp, spaces)
            board["fit_score"] = [fitter_global.fit_score(i, pseudo_club) for i in board.index]
            for col in ["predicted_value", CFG.TARGET]:
                mu, sd = board[col].mean(), board[col].std()+1e-9
                board[f"{col}_z"] = (board[col]-mu)/sd
            board["signing_score"] = 0.55*board["fit_score"] + 0.35*board["predicted_value_z"] - 0.10*board[f"{CFG.TARGET}_z"]

        if pos_sel:
            board = board[board[CFG.POS_COL].fillna("").apply(lambda s: any(p in s for p in pos_sel))]
        if "Age" in board.columns:
            board = board[(board["Age"]>=age_rng[0]) & (board["Age"]<=age_rng[1])]
        # presupuesto sobre valor de mercado actual:
        board = board[board[CFG.TARGET] <= (budget_m * 1_000_000)]

        cols_show = ["Player","Team",CFG.POS_COL,"Age",CFG.TARGET,"predicted_value","fit_score","signing_score"]
        cols_show = [c for c in cols_show if c in board.columns]
        board = board.sort_values(sort_by, ascending=(sort_by in ["predicted_value","Market value"])).reset_index(drop=True)

        st.dataframe(display_money(board[cols_show].head(200)), use_container_width=True)
        st.download_button("Exportar shortlist (CSV)",
                           data=board[cols_show].to_csv(index=False).encode("utf-8"),
                           file_name="libro_de_pases_shortlist.csv",
                           mime="text/csv")

# 5) Predicciones (tabla y importancias)
with tab_pred:
    st.subheader("Predicciones de valor de mercado (display en millones)")
    cols_show = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","delta_pred_real"] if c in df_pred.columns]
    st.dataframe(display_money(df_pred[cols_show].head(500)), use_container_width=True)
    st.download_button("Descargar predicciones (CSV)",
                       data=df_pred[cols_show].to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")
    st.markdown("**Importancia del modelo**")
    plot_importances(model.model, X, top_n=20)

# 6) Glosario
with tab_help:
    st.subheader("Glosario y criterios")
    st.markdown("""
- **Team within selected timeframe**: todos los cálculos usan sólo registros dentro del rango de fechas y/o temporadas seleccionadas.  
- **Market value**: valor de mercado real del dataset.  
- **predicted_value**: estimación del modelo (XGBoost).  
- **delta_pred_real**: `predicted_value - Market value` (positivo → posible infravaloración).  
- **fit_score (0–1)**: similitud del vector PCA del jugador con los centroides del club destino por posición.  
- **signing_score**: `0.55*fit + 0.35*z(predicted_value) - 0.10*z(Market value)`; prioriza encaje e impacto esperado penalizando costo.  
- **rank_for_target**: posición del jugador en el ranking del club (1 = mejor), excluyendo jugadores del propio club.  
- **Libro de pases**: shortlist filtrable por posición/edad/presupuesto con exportación.  
- **Radares**: comparan al jugador (percentil normalizado) contra el perfil del club.  
- **RAG**: resúmenes y argumentos con LLM usando contexto real de tu dataset (TF-IDF).
""")
st.success("Listo. Explora ranking, encaje, destinos y arma tu libro de pases. 🚀")
