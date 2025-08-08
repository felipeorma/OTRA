# app.py
# ===============================================================
# ⚽ Scouting Tool — Streamlit (multi-archivo, timeframe, ML, encaje, ranking, RAG)
# ===============================================================
# Cómo usar:
# 1) Instala deps: `pip install -r requirements.txt`
# 2) Ejecuta: `streamlit run app.py`
# ---------------------------------------------------------------

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

# RAG (cargado sólo si hay API Key)
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# -------------------------
# Config & estilo
# -------------------------
st.set_page_config(page_title="Scouting Tool", layout="wide")
st.markdown(
    """
    <style>
    .small-note { color: #6b7280; font-size: 0.9em; }
    .metric-card { background: #0f172a10; border-radius: 12px; padding: 10px 14px; }
    </style>
    """,
    unsafe_allow_html=True
)

@dataclass
class Config:
    TARGET: str = "Market value"
    ID_COLS: Tuple[str, ...] = ("Player", "Team", "Position", "Age", "Contract expires")
    POS_COL: str = "Position"
    RANDOM_STATE: int = 42
    TOP_K_RECS: int = 20
    SHAP_SAMPLE: int = 250

CFG = Config()
np.random.seed(CFG.RANDOM_STATE)

# -------------------------
# Utilidades
# -------------------------
def format_millions(v):
    """Formatea números a millones (3.45 M)."""
    try:
        return f"{float(v)/1_000_000:.2f} M"
    except Exception:
        return v

def format_values_for_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in [CFG.TARGET, "predicted_value", "delta_pred_real"]:
        if col in out.columns:
            out[col] = out[col].apply(format_millions)
    return out

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = str(c).replace("\n", " ").replace("\r", " ")
        c2 = " ".join(c2.split())
        new_cols.append(c2)
    df.columns = new_cols
    return df

def _auto_pick_sheet(xls: pd.ExcelFile, filename: str, required={"Player","Team","Market value"}) -> str:
    cands = [s for s in xls.sheet_names if any(k in s.lower() for k in ("search","result","players","jugadores"))]
    if cands:
        return cands[0]
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
def read_any(file_bytes, filename: str, sheet_hint: Optional[str]) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        xls = pd.ExcelFile(file_bytes)
        chosen = sheet_hint or _auto_pick_sheet(xls, filename)
        df = pd.read_excel(file_bytes, sheet_name=chosen)
        df["__source_file__"] = filename
        df["__source_sheet__"] = chosen
    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(file_bytes)
        df["__source_file__"] = filename
        df["__source_sheet__"] = ""
    else:
        raise ValueError("Formato no soportado. Usa .xlsx o .csv")
    df = _normalize_columns(df)
    return df

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

def apply_time_filter(df: pd.DataFrame, date_col: Optional[str], date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
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
    df2 = df[df[CFG.TARGET].notna()].copy()
    df2[CFG.TARGET] = pd.to_numeric(df2[CFG.TARGET], errors="coerce")
    num_cols = [c for c in df2.columns if (np.issubdtype(df2[c].dtype, np.number) and c != CFG.TARGET)]
    if not num_cols:
        raise ValueError("No hay columnas numéricas suficientes para entrenar.")
    X = df2[num_cols].fillna(0.0)
    y = df2[CFG.TARGET].astype(float)
    return df2, X, y, num_cols

class LatentSpaces:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=12, random_state=CFG.RANDOM_STATE)
    def fit_transform(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        Xs = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        pca_latent = pd.DataFrame(self.pca.fit_transform(Xs), index=X.index)
        return {"scaled": Xs, "pca": pca_latent}

class MarketValueModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=650, learning_rate=0.06, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            random_state=CFG.RANDOM_STATE
        )
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        kf = KFold(n_splits=5, shuffle=True, random_state=CFG.RANDOM_STATE)
        cv_mae = -1 * cross_val_score(self.model, X, y, scoring="neg_mean_absolute_error", cv=kf).mean()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=CFG.RANDOM_STATE)
        self.model.fit(Xtr, ytr)
        y_pred = self.model.predict(Xte)
        return {
            "cv_mae": float(cv_mae),
            "holdout_mae": float(mean_absolute_error(yte, y_pred)),
            "r2": float(r2_score(yte, y_pred))
        }
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

def shap_plot(model: XGBRegressor, X: pd.DataFrame, sample: int = 250):
    try:
        idx = np.random.choice(X.index, size=min(sample, len(X)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.loc[idx])
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(sv, X.loc[idx], plot_type="bar", show=False, color=None)
        st.pyplot(fig)
    except Exception as e:
        st.info(f"SHAP no disponible: {e}")

class ClubFit:
    """
    Fit-score = similitud coseno entre el vector PCA del jugador y
    los centroides del club destino por cada posición del jugador,
    calculados SOLO con el equipo en el timeframe seleccionado.
    signing_score = 0.55*fit + 0.35*z(predicted) - 0.10*z(real)
    rank_for_target: ranking (1 = mejor) del jugador en el club destino,
    calculado EXCLUYENDO jugadores que ya pertenecen a ese club.
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
                if len(idxs) > 0:
                    d[(club, p)] = self.latent.loc[idxs].mean(axis=0).values
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

    def rank_signings(self, target_club: str, top_k: int = 20) -> pd.DataFrame:
        df = self._with_scores(target_club)
        df = df[df["Team"] != target_club]  # ⛔ excluir jugadores del club destino
        cols = ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","fit_score","signing_score"]
        cols = [c for c in cols if c in df.columns]
        return df.sort_values("signing_score", ascending=False).head(top_k)[cols]

    def eval_player_in_club(self, player_name: str, target_club: str) -> pd.DataFrame:
        df_scored = self._with_scores(target_club)
        board = df_scored[df_scored["Team"] != target_club].sort_values("signing_score", ascending=False).reset_index(drop=True)
        cand = df_scored[df_scored["Player"].str.lower()==str(player_name).lower()]
        if cand.empty:
            raise ValueError(f"Jugador '{player_name}' no encontrado en el timeframe filtrado.")
        out_cols = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value"] if c in df_scored.columns]
        out = cand[out_cols].copy()
        out["target_club"] = target_club
        out["fit_score"] = float(cand.iloc[0]["fit_score"]) if "fit_score" in cand.columns else 0.0
        mask = board["Player"].str.lower()==str(player_name).lower()
        out["rank_for_target"] = int(board.index[mask][0]) + 1 if mask.any() else -1
        return out

# -------------------------
# Sidebar: carga y ajustes
# -------------------------
with st.sidebar:
    st.header("① Carga de archivos")
    uploaded_files = st.file_uploader(
        "Sube uno o varios archivos (.xlsx o .csv)", type=["xlsx","csv"], accept_multiple_files=True
    )
    sheet_hint = st.text_input("Nombre de hoja (opcional, para Excel). Deja vacío para autodetectar.", value="")

    st.header("② Filtro timeframe")
    st.caption("Se aplica ANTES de entrenar el modelo y calcular encaje/ranking.")
    # place-holders (se llenan luego de cargar datos)

    st.header("③ RAG (opcional)")
    api_key = st.text_input("OPENAI_API_KEY", type="password", help="Pégala para habilitar el chat contextual.")

    st.header("④ Parámetros")
    top_k = st.number_input("Top K (ranking)", min_value=5, max_value=200, value=CFG.TOP_K_RECS, step=1)
    show_shap = st.checkbox("Mostrar SHAP (importancia de variables)", value=True)

st.title("⚽ Scouting Tool")

if not uploaded_files:
    st.info("Sube al menos un archivo para comenzar.")
    st.stop()

# -------------------------
# Unir archivos
# -------------------------
dfs = []
for f in uploaded_files:
    try:
        df_tmp = read_any(f, f.name, sheet_hint or None)
        dfs.append(df_tmp)
    except Exception as e:
        st.error(f"Error leyendo {f.name}: {e}")
        st.stop()

df_raw = pd.concat(dfs, ignore_index=True, sort=False)
date_cols, season_cols = detect_time_columns(df_raw)

# -------------------------
# Controles timeframe (en main)
# -------------------------
st.subheader("Vista previa y selección de timeframe")
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

st.caption("Todo lo que sigue se calculará **dentro del timeframe** elegido.")

# -------------------------
# Aplicar timeframe
# -------------------------
df_f = apply_time_filter(df_raw, date_col, date_range, season_col, seasons_selected)
if df_f.empty:
    st.warning("El filtro de timeframe dejó el dataset vacío. Ajusta los filtros.")
    st.stop()

# -------------------------
# Entrenamiento (cacheado)
# -------------------------
@st.cache_resource(show_spinner=True)
def train_pipeline(df_in: pd.DataFrame):
    df_for_model, X, y, num_cols = select_numeric(df_in)
    model = MarketValueModel()
    metrics = model.train(X, y)
    y_hat = model.predict(X)
    spaces = LatentSpaces().fit_transform(X)
    return df_for_model, X, y, y_hat, model, metrics, spaces

df_for_model, X, y, y_hat, model, metrics, spaces = train_pipeline(df_f)

# Resultados base
df_pred = df_for_model.copy()
df_pred["predicted_value"] = y_hat
df_pred["delta_pred_real"] = df_pred["predicted_value"] - df_pred[CFG.TARGET]
fitter = ClubFit(df_pred, spaces)

# -------------------------
# Métricas del modelo
# -------------------------
st.markdown("### Desempeño del modelo (dentro del timeframe)")
m1, m2, m3 = st.columns(3)
m1.markdown(f"<div class='metric-card'><b>CV MAE</b><br>{metrics['cv_mae']:,.0f}</div>", unsafe_allow_html=True)
m2.markdown(f"<div class='metric-card'><b>Holdout MAE</b><br>{metrics['holdout_mae']:,.0f}</div>", unsafe_allow_html=True)
m3.markdown(f"<div class='metric-card'><b>R²</b><br>{metrics['r2']:.3f}</div>", unsafe_allow_html=True)

if show_shap:
    st.markdown("**Importancia de variables (SHAP)**")
    shap_plot(model.model, X, sample=min(CFG.SHAP_SAMPLE, len(X)))

# -------------------------
# Tabs funcionales
# -------------------------
tab_pred, tab_fit, tab_rank, tab_rag, tab_help = st.tabs(
    ["Predicciones", "Encaje jugador→club", "Mejores incorporaciones", "Chat (RAG)", "Glosario / Ayuda"]
)

with tab_pred:
    st.subheader("Predicciones de valor de mercado")
    cols_show = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","delta_pred_real"] if c in df_pred.columns]
    st.dataframe(format_values_for_display(df_pred[cols_show].head(500)), use_container_width=True)
    st.caption("Se muestran 500 filas. Usa la descarga para ver completo.")
    st.download_button("Descargar CSV con predicciones", data=df_pred[cols_show].to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")

with tab_fit:
    st.subheader("Encaje de un jugador en un club (EXCLUYE jugadores del club en el ranking)")
    if ("Player" not in df_pred.columns) or ("Team" not in df_pred.columns):
        st.error("Necesito columnas 'Player' y 'Team' para esta función.")
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
                st.dataframe(format_values_for_display(res), use_container_width=True)
                st.markdown(
                    """
                    <div class='small-note'>
                    <b>fit_score</b>: 0–1, similitud del jugador con el perfil del club en sus posiciones.<br>
                    <b>rank_for_target</b>: posición del jugador en el ranking para ese club (1 = mejor), calculado excluyendo jugadores que ya están en el club destino.
                    </div>
                    """, unsafe_allow_html=True
                )
            except Exception as e:
                st.error(str(e))

with tab_rank:
    st.subheader("Ranking de mejores incorporaciones (dentro del timeframe)")
    if "Team" not in df_pred.columns:
        st.error("Necesito columna 'Team' para esta función.")
    else:
        clubs = sorted(df_pred["Team"].dropna().unique().tolist())
        target_club_r = st.selectbox("Club destino", options=clubs, key="rank_club")
        if st.button("Calcular ranking"):
            try:
                board = fitter.rank_signings(target_club_r, top_k=int(top_k))
                st.dataframe(format_values_for_display(board), use_container_width=True)
                st.caption("El ranking excluye a los jugadores que ya pertenecen al club destino en el periodo seleccionado.")
                st.download_button("Descargar ranking CSV", data=board.to_csv(index=False).encode("utf-8"),
                                   file_name=f"ranking_{target_club_r}.csv", mime="text/csv")
            except Exception as e:
                st.error(str(e))

with tab_rag:
    st.subheader("Chat RAG sobre el dataset (opcional)")
    if not LANGCHAIN_AVAILABLE:
        st.info("Faltan librerías de LangChain/FAISS. Revisa requirements.txt")
    else:
        q_default = "Dame 5 posibles fichajes con alto fit y buen precio para {club}"
        question = st.text_input("Pregunta", value=q_default)
        club_for_ctx = st.selectbox("Club para el contexto (opcional)", options=sorted(df_pred["Team"].dropna().unique().tolist()) if "Team" in df_pred.columns else [])
        if st.button("Preguntar"):
            if not api_key:
                st.warning("Pega tu OPENAI_API_KEY en la barra lateral para habilitar el chat.")
            else:
                try:
                    os.environ["OPENAI_API_KEY"] = api_key
                    df_ctx = df_pred.copy()
                    if club_for_ctx:
                        tmp = fitter.rank_signings(club_for_ctx, top_k=len(df_ctx))
                        if "Player" in df_ctx.columns:
                            df_ctx = df_ctx.merge(tmp[["Player","fit_score","signing_score"]], on="Player", how="left")
                    # Construir corpus
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
                    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
                    pieces = []
                    for d in docs:
                        pieces.extend(splitter.split_text(d))
                    store = FAISS.from_texts(pieces, OpenAIEmbeddings())
                    chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                    ctx = question.replace("{club}", club_for_ctx) if club_for_ctx else question
                    sims = store.similarity_search(ctx, k=6)
                    context_text = "\n\n---\n\n".join([s.page_content for s in sims])
                    msgs = [
                        SystemMessage(content="Eres analista de scouting. Responde solo con datos del contexto y sé accionable."),
                        HumanMessage(content=f"Contexto:\n{context_text}\n\nPregunta: {ctx}")
                    ]
                    answer = chat.invoke(msgs).content
                    st.write(answer)
                except Exception as e:
                    st.error(f"RAG error: {e}")

with tab_help:
    st.subheader("Glosario y criterios de la herramienta")
    st.markdown(
        """
        - **Team within selected timeframe**: todo se calcula usando sólo los partidos / registros dentro del **rango de fechas** y/o **temporadas** que selecciones.  
        - **Market value**: valor de mercado real del dataset que subiste.  
        - **predicted_value**: valor estimado por nuestro modelo (XGBoost) entrenado con variables numéricas del timeframe.  
        - **delta_pred_real**: `predicted_value - Market value`. Positivo → el modelo cree que el jugador está **infravalorado**.  
        - **fit_score (0–1)**: similitud del jugador con el **perfil** del club destino (centroides por posición en un espacio PCA).  
        - **signing_score**: métrica compuesta para ranking: `0.55*fit + 0.35*z(predicted_value) - 0.10*z(Market value)`  
          - Favorece **encaje** y **calidad esperada** (predicted), penaliza **costo** (market value alto).  
        - **rank_for_target**: posición del jugador en el ranking **para ese club** (1 = mejor), **excluyendo** a los jugadores que ya están en ese club dentro del timeframe.  
        - **Top K**: número de candidatos que quieres ver en el ranking (no es una métrica, sólo un parámetro de corte).  
        - **SHAP**: explica qué variables empujan más las predicciones del modelo.  
        - **RAG**: chat que responde con base en el contenido real de tu dataset (necesita tu `OPENAI_API_KEY`).  
        """.strip()
    )
    st.markdown(
        """
        **Sugerencias:**  
        - Verifica que las columnas `Player`, `Team`, `Position` y `Market value` existan y estén limpias.  
        - Si tienes métricas por 90' (xG, xA, etc.), el modelo y el fit-score suelen mejorar.  
        - Puedes subir varios archivos (p.ej., temporadas distintas) y filtrar con timeframe para estudiar ventanas específicas.  
        """.strip()
    )

st.success("Listo. Carga archivos, filtra el timeframe, y explora predicciones, encaje y ranking.")
