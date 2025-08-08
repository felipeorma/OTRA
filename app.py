# streamlit_app.py
# ===============================================================
# SCOUTING TOOL (Streamlit, solo XLSX/CSV local)
# - Auto-detección de hoja (XLSX)
# - Filtro por "Team within selected timeframe" (fecha/temporada)
# - ML (XGBoost) para valor de mercado + SHAP
# - Encaje por club (fit_score) dentro del timeframe
# - Ranking de mejores incorporaciones (EXCLUYE jugadores del club destino)
# - RAG opcional (pega tu OPENAI_API_KEY en el UI)
# - Valores monetarios redondeados a millones
# ===============================================================
# Cómo correr:
# 1) pip install streamlit numpy==1.26.4 pandas==2.2.2 scipy==1.11.4 scikit-learn==1.4.2 xgboost==2.0.3 shap==0.45.1
#    pip install langchain==0.2.16 langchain-community faiss-cpu tiktoken openpyxl==3.1.5
# 2) streamlit run streamlit_app.py
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

# RAG (se carga solo si el usuario pega su API key)
try:
    from langchain.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# ============ Config básica ============
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

# ============ Utilidades ============
def format_millions(v):
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

def _auto_pick_sheet(xls: pd.ExcelFile, fpath: str, required={"Player","Team","Market value"}) -> str:
    # 1) nombre por heurística
    cands = [s for s in xls.sheet_names if any(k in s.lower() for k in ("search","result","players","jugadores"))]
    if cands:
        return cands[0]
    # 2) scoring por columnas
    best, best_score = None, -1
    for s in xls.sheet_names:
        try:
            preview = pd.read_excel(fpath, sheet_name=s, nrows=5)
            cols_norm = {str(c).strip() for c in preview.columns}
            score = len(required.intersection(cols_norm))
            if score > best_score:
                best, best_score = s, score
        except Exception:
            pass
    return best or xls.sheet_names[0]

@st.cache_data(show_spinner=False)
def load_file(file_bytes, filename: str, sheet_name: Optional[str]) -> pd.DataFrame:
    if filename.lower().endswith(".xlsx"):
        xls = pd.ExcelFile(file_bytes)
        chosen = sheet_name or _auto_pick_sheet(xls, filename)
        df = pd.read_excel(file_bytes, sheet_name=chosen)
    elif filename.lower().endswith(".csv"):
        df = pd.read_csv(file_bytes)
    else:
        raise ValueError("Formato no soportado. Usa .xlsx o .csv")
    df = _normalize_columns(df)
    return df

def detect_time_columns(df: pd.DataFrame):
    """Detecta columnas de tiempo: fechas y temporadas."""
    # Fechas reales
    date_cols = []
    for c in df.columns:
        if "date" in c.lower() or "match" in c.lower() or "fecha" in c.lower():
            try:
                pd.to_datetime(df[c], errors="raise")
                date_cols.append(c)
            except Exception:
                pass
        elif np.issubdtype(df[c].dtype, np.datetime64):
            date_cols.append(c)

    # Temporadas / años
    season_cols = [c for c in df.columns if any(k in c.lower() for k in ("season","temporada","year","año"))]
    return date_cols, season_cols

def apply_time_filter(df: pd.DataFrame, date_col: Optional[str], date_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]],
                      season_col: Optional[str], seasons_selected: Optional[List]) -> pd.DataFrame:
    out = df.copy()
    if date_col and date_col in out.columns and date_range:
        out = out.copy()
        out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
        out = out[(out[date_col] >= date_range[0]) & (out[date_col] <= date_range[1])]
    if season_col and season_col in out.columns and seasons_selected:
        out = out[out[season_col].isin(seasons_selected)]
    return out.reset_index(drop=True)

def select_numeric(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
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
    calculado EXCLUYENDO los jugadores que ya pertenecen a ese club.
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
        # Excluir jugadores que YA están en el club destino en el timeframe filtrado
        df = df[df["Team"] != target_club]
        cols = ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","fit_score","signing_score"]
        cols = [c for c in cols if c in df.columns]
        return df.sort_values("signing_score", ascending=False).head(top_k)[cols]

    def eval_player_in_club(self, player_name: str, target_club: str) -> pd.DataFrame:
        df_scored = self._with_scores(target_club)
        # Ranking se calcula EXCLUYENDO jugadores del club destino
        board = df_scored[df_scored["Team"] != target_club].sort_values("signing_score", ascending=False).reset_index(drop=True)
        # Encontrar jugador
        cand = df_scored[df_scored["Player"].str.lower()==str(player_name).lower()]
        if cand.empty:
            raise ValueError(f"Jugador '{player_name}' no encontrado en el timeframe filtrado.")
        out_cols = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value"] if c in df_scored.columns]
        out = cand[out_cols].copy()
        out["target_club"] = target_club
        out["fit_score"] = float(cand.iloc[0]["fit_score"]) if "fit_score" in cand.columns else 0.0
        # Rank for target (1 = mejor) en el board sin jugadores del club
        mask = board["Player"].str.lower()==str(player_name).lower()
        out["rank_for_target"] = int(board.index[mask][0]) + 1 if mask.any() else -1
        return out

# ============ UI ============
st.set_page_config(page_title="Scouting Tool", layout="wide")
st.title("⚽ Scouting Tool — Streamlit")

with st.sidebar:
    st.header("1) Subir archivo")
    file = st.file_uploader("XLSX o CSV con jugadores (una liga).", type=["xlsx","csv"])
    sheet_input = None
    if file is not None and file.name.lower().endswith(".xlsx"):
        # dar opción (opcional) de escribir el nombre exacto de hoja
        sheet_input = st.text_input("Nombre de hoja (opcional). Si vacío, autodetecto.", value="")

    st.header("2) Filtro timeframe")
    date_col = season_col = None
    date_range = seasons_selected = None

    st.caption("Aplica el filtro ANTES de entrenar y calcular encajes/rankings.")

    st.header("3) RAG (opcional)")
    api_key = st.text_input("OPENAI_API_KEY (empieza con 'sk-...')", type="password", help="Pégala aquí para habilitar el chat con contexto del dataset.")

    st.header("4) Parámetros")
    top_k = st.number_input("Top K (ranking)", min_value=5, max_value=200, value=CFG.TOP_K_RECS, step=1)
    show_shap = st.checkbox("Mostrar SHAP (importancia de variables)", value=True)

main_tab, pred_tab, fit_tab, rank_tab, rag_tab = st.tabs(["Inicio", "Predicciones", "Encaje jugador→club", "Mejores incorporaciones", "Chat (RAG)"])

if not file:
    st.info("Sube un archivo para empezar.")
    st.stop()

# ====== Carga inicial ======
try:
    df_raw = load_file(file, file.name, sheet_input if sheet_input else None)
except Exception as e:
    st.error(f"Error al cargar: {e}")
    st.stop()

# Detectar columnas de tiempo
date_cols, season_cols = detect_time_columns(df_raw)

with main_tab:
    st.subheader("Vista previa de datos")
    st.write(df_raw.head(10))

    # Controles timeframe
    st.markdown("### Filtro de periodo (Team within selected timeframe)")
    col1, col2 = st.columns(2)
    with col1:
        chosen_date_col = st.selectbox("Columna de fecha (opcional)", options=["(ninguna)"]+date_cols, index=0)
        if chosen_date_col != "(ninguna)":
            series_dt = pd.to_datetime(df_raw[chosen_date_col], errors="coerce").dropna()
            if not series_dt.empty:
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

    st.caption("Usaré estos filtros para definir el **plantel y estilo del club** dentro del periodo seleccionado.")

# Aplicar timeframe
df_f = apply_time_filter(df_raw, date_col, date_range, season_col, seasons_selected)
if df_f.empty:
    st.warning("El filtro de timeframe dejó el dataset vacío. Ajusta los filtros.")
    st.stop()

# ============ Entrenamiento y features (sobre timeframe filtrado) ============
try:
    df_for_model, X, y, num_cols = select_numeric(df_f)
except Exception as e:
    st.error(f"Error preparando datos: {e}")
    st.stop()

@st.cache_resource(show_spinner=True)
def train_model(X, y):
    m = MarketValueModel()
    metrics = m.train(X, y)
    y_hat = m.predict(X)
    return m, metrics, y_hat

with main_tab:
    st.subheader("Entrenamiento del modelo (dentro del timeframe)")
    model, metrics, y_hat = train_model(X, y)
    st.write({
        "CV MAE": f"{metrics['cv_mae']:,.0f}",
        "Holdout MAE": f"{metrics['holdout_mae']:,.0f}",
        "R2": f"{metrics['r2']:.3f}"
    })
    if show_shap:
        st.markdown("**Importancia de variables (SHAP)**")
        shap_plot(model.model, X, sample=min(CFG.SHAP_SAMPLE, len(X)))

# Predicciones y espacios latentes
df_pred = df_for_model.copy()
df_pred["predicted_value"] = y_hat
df_pred["delta_pred_real"] = df_pred["predicted_value"] - df_pred[CFG.TARGET]

@st.cache_resource(show_spinner=False)
def build_spaces(X_df):
    return LatentSpaces().fit_transform(X_df)

spaces = build_spaces(X)
fitter = ClubFit(df_pred, spaces)

# ============ Tabs funcionales ============
with pred_tab:
    st.subheader("Predicciones de valor de mercado")
    cols_show = [c for c in ["Player","Team","Position","Age",CFG.TARGET,"predicted_value","delta_pred_real"] if c in df_pred.columns]
    st.dataframe(format_values_for_display(df_pred[cols_show].head(200)), use_container_width=True)
    st.caption("Se muestran 200 filas (usa descarga para ver todo).")
    st.download_button("Descargar CSV con predicciones", data=df_pred[cols_show].to_csv(index=False).encode("utf-8"),
                       file_name="predicciones.csv", mime="text/csv")

with fit_tab:
    st.subheader("Encaje de un jugador en un club (dentro del timeframe)")
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
                st.markdown("""
                **Notas**  
                - `fit_score`: similitud 0–1 con el perfil del club en sus posiciones.  
                - `rank_for_target`: ranking del jugador para ese club (1 = mejor), **excluyendo** jugadores que ya pertenecen al club destino dentro del timeframe.  
                """)
            except Exception as e:
                st.error(str(e))

with rank_tab:
    st.subheader("Ranking de mejores incorporaciones (EXCLUYE jugadores del club)")
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

with rag_tab:
    st.subheader("Chat RAG sobre el dataset (opcional)")
    if not LANGCHAIN_AVAILABLE:
        st.info("Librerías de LangChain no instaladas. Instálalas para usar RAG.")
    else:
        q_default = "Dame 5 posibles fichajes con alto fit y buen precio para {club}"
        question = st.text_input("Pregunta", value=q_default)
        club_for_ctx = st.selectbox("Club para el contexto (opcional)", options=sorted(df_pred["Team"].dropna().unique().tolist()) if "Team" in df_pred.columns else [])
        if st.button("Preguntar"):
            if not api_key:
                st.warning("Pega tu OPENAI_API_KEY en la barra lateral para habilitar el chat.")
            else:
                try:
                    df_ctx = df_pred.copy()
                    # Añadimos fit/signing para el club (si se seleccionó) — excluye jugadores del club en ranking
                    if club_for_ctx:
                        tmp = fitter.rank_signings(club_for_ctx, top_k=len(df_ctx))
                        if "Player" in df_ctx.columns:
                            df_ctx = df_ctx.merge(tmp[["Player","fit_score","signing_score"]], on="Player", how="left")
                    os.environ["OPENAI_API_KEY"] = api_key
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

st.success("Listo. Usa los tabs para trabajar. Recuerda: todo se calcula **dentro del timeframe** seleccionado, y el ranking **excluye** jugadores del club destino.")
