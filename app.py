import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# ==============================
# CONFIGURACIÓN INICIAL
# ==============================
st.set_page_config(page_title="Scouting Tool Avanzada", layout="wide")

TARGET = "Market value"
ID_COLS = ["Player", "Team", "Position", "Age", "Contract expires"]

# ==============================
# FUNCIONES AUXILIARES
# ==============================
def load_multiple_files(uploaded_files):
    dfs = []
    for file in uploaded_files:
        try:
            df = pd.read_excel(file)
            dfs.append(df)
        except Exception as e:
            st.error(f"Error leyendo {file.name}: {e}")
    if dfs:
        df_full = pd.concat(dfs, ignore_index=True)
        df_full = df_full[df_full[TARGET].notna()].copy()
        df_full[TARGET] = pd.to_numeric(df_full[TARGET], errors="coerce")
        return df_full
    return pd.DataFrame()

def select_numeric(df):
    num_cols = [c for c in df.columns if (np.issubdtype(df[c].dtype, np.number) and c != TARGET)]
    X = df[num_cols].fillna(0.0)
    y = df[TARGET].astype(float)
    return X, y, num_cols

class LatentSpaces:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=12, random_state=42)

    def fit_transform(self, X):
        Xs = pd.DataFrame(self.scaler.fit_transform(X), columns=X.columns, index=X.index)
        pca_latent = pd.DataFrame(self.pca.fit_transform(Xs), index=X.index)
        return {"scaled": Xs, "pca": pca_latent}

class MarketValueModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=650, learning_rate=0.06, max_depth=8,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            random_state=42
        )

    def train(self, X, y):
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -1 * cross_val_score(self.model, X, y, scoring="neg_mean_absolute_error", cv=kf).mean()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(Xtr, ytr)
        y_pred = self.model.predict(Xte)
        return {
            "cv_mae": float(cv_mae),
            "holdout_mae": float(mean_absolute_error(yte, y_pred)),
            "r2": float(r2_score(yte, y_pred))
        }

    def predict(self, X):
        return self.model.predict(X)

def plot_shap_importance(model, X, sample=250):
    try:
        idx = np.random.choice(X.index, size=min(sample, len(X)), replace=False)
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X.loc[idx])
        shap.summary_plot(sv, X.loc[idx], plot_type="bar", show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"No se pudo generar gráfico SHAP: {e}")

class ClubFit:
    def __init__(self, df_full, spaces):
        self.df = df_full.copy()
        self.latent = spaces["pca"]
        self.centroids = self._compute_centroids()

    def _norm_positions(self, pos_str):
        if pd.isna(pos_str): return []
        return [p.strip() for p in str(pos_str).split(",") if p.strip()]

    def _compute_centroids(self):
        d = {}
        for club, g in self.df.groupby("Team"):
            pos_map = {}
            for i, row in g.iterrows():
                for p in self._norm_positions(row.get("Position", "")):
                    pos_map.setdefault(p, []).append(i)
            for p, idxs in pos_map.items():
                d[(club, p)] = self.latent.loc[idxs].mean(axis=0).values
        return d

    def _cosine(self, a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0: return 0.0
        return float(np.dot(a, b) / (na * nb))

    def fit_score(self, idx, target_club):
        row = self.df.loc[idx]
        vec = self.latent.loc[idx].values
        positions = self._norm_positions(row.get("Position", ""))
        sims = []
        for p in positions:
            k = (target_club, p)
            if k in self.centroids:
                sims.append(self._cosine(vec, self.centroids[k]))
        return float(np.mean(sims)) if sims else 0.0

    def rank_signings(self, target_club, top_k=20, alpha=0.55, beta=0.35, gamma=0.10):
        df = self.df.copy()
        df = df[df["Team"] != target_club]  # excluir jugadores que ya están en el club
        df["fit_score"] = [self.fit_score(i, target_club) for i in df.index]
        for col in ["predicted_value", TARGET]:
            mu, sd = df[col].mean(), df[col].std() + 1e-9
            df[f"{col}_z"] = (df[col] - mu) / sd
        df["signing_score"] = alpha * df["fit_score"] + beta * df["predicted_value_z"] - gamma * df[f"{TARGET}_z"]
        return df.sort_values("signing_score", ascending=False).head(top_k)

# ==============================
# INTERFAZ STREAMLIT
# ==============================
st.title("📊 Scouting Tool Avanzada")
st.markdown("""
Esta herramienta permite:
- **Predecir valor de mercado** con Machine Learning.
- **Calcular encaje de jugadores** en un club objetivo.
- **Generar ranking de mejores fichajes** excluyendo los jugadores que ya están en el club.
- **Explicar cada métrica** usada.
""")

uploaded_files = st.file_uploader("📂 Cargar archivos XLSX", type=["xlsx"], accept_multiple_files=True)

if uploaded_files:
    df = load_multiple_files(uploaded_files)
    X, y, num_cols = select_numeric(df)

    m = MarketValueModel()
    metrics = m.train(X, y)
    df["predicted_value"] = m.predict(X).round(2)
    df["delta_pred_real"] = df["predicted_value"] - df[TARGET]

    st.success(f"Modelo entrenado | CV MAE: {metrics['cv_mae']:.2f} | Holdout MAE: {metrics['holdout_mae']:.2f} | R²: {metrics['r2']:.3f}")

    spaces = LatentSpaces().fit_transform(X)
    fitter = ClubFit(df, spaces)

    tab1, tab2, tab3 = st.tabs(["📈 Predicciones", "🏟 Ranking Fichajes", "ℹ Explicación de métricas"])

    with tab1:
        st.dataframe(df[[*ID_COLS, TARGET, "predicted_value", "delta_pred_real"]])
        plot_shap_importance(m.model, X)

    with tab2:
        club = st.selectbox("Selecciona club destino", sorted(df["Team"].dropna().unique()))
        top_k = st.slider("Número de fichajes sugeridos", 5, 50, 20)
        if club:
            ranking = fitter.rank_signings(club, top_k=top_k)
            st.dataframe(ranking[[*ID_COLS, TARGET, "predicted_value", "fit_score", "signing_score"]])

    with tab3:
        st.markdown("""
        **Market value:** Valor de mercado actual del jugador.
        **predicted_value:** Valor estimado por el modelo ML.
        **delta_pred_real:** Diferencia entre el valor estimado y el real.
        **fit_score:** Qué tan bien encaja el jugador en el club objetivo (0-1).
        **signing_score:** Ranking final ponderado que combina fit, valor estimado y valor real.
        """)
else:
    st.info("Sube al menos un archivo XLSX para comenzar.")
