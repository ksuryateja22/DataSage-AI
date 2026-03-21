import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pandasql as ps
from scipy import stats
from scipy.stats import skew, ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DataSage AI Pro", layout="wide", page_icon="🧪")

st.markdown("""
<style>
.main { background-color: #0e1117; }
.stMetric { background-color: #1e2130; padding: 12px; border-radius: 10px; border: 1px solid #3e4259; }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION ----------------
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛡️ DataSage AI Pro")
    st.write("KLEF CSE Edition")
    menu = st.radio("Modules", [
        "📁 Data Integration",
        "🧹 Stat-Cleaning",
        "🔬 Inference Lab",
        "📉 Risk Analytics",
        "🤖 ML Engine",
        "🔍 SQL Workspace"
    ])

# ---------------- 1. DATA INTEGRATION ----------------
if menu == "📁 Data Integration":
    st.title("📂 Data Integration")
    files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

    if files:
        dfs = {f.name: pd.read_csv(f) for f in files}

        for name, df in dfs.items():
            st.subheader(name)
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            st.dataframe(df)

        # If only 1 file → auto select
        if len(dfs) == 1:
            st.session_state.merged_df = list(dfs.values())[0]
            st.success("Single dataset loaded.")

        # If 2+ files → merge option
        if len(dfs) >= 2:
            st.subheader("🔗 Merge Datasets")
            keys = list(dfs.keys())
            t1 = st.selectbox("Dataset A", keys)
            t2 = st.selectbox("Dataset B", keys)
            join_col = st.text_input("Join Column")

            if st.button("Merge"):
                try:
                    st.session_state.merged_df = pd.merge(dfs[t1], dfs[t2], on=join_col, how="inner")
                    st.success("Datasets merged successfully.")
                except Exception as e:
                    st.error(e)

# ---------------- 2. STAT CLEANING ----------------
elif menu == "🧹 Stat-Cleaning":
    st.title("🧹 Statistical Cleaning")
    df = st.session_state.merged_df

    if df is not None:
        num_cols = df.select_dtypes(include=np.number).columns
        col = st.selectbox("Select Column", num_cols)

        skew_val = skew(df[col].dropna())
        st.metric("Skewness", f"{skew_val:.2f}")

        if abs(skew_val) > 0.5:
            st.warning("High skew → Median preferred")
        else:
            st.success("Normal distribution → Mean is fine")

        method = st.radio("Method", ["Mean", "Median", "Drop Nulls"])

        if st.button("Apply"):
            if method == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "Median":
                df[col] = df[col].fillna(df[col].median())
            else:
                df = df.dropna(subset=[col])

            st.session_state.merged_df = df
            st.success("Cleaning applied")

    else:
        st.warning("Upload data first")

# ---------------- 3. INFERENCE ----------------
elif menu == "🔬 Inference Lab":
    st.title("🔬 Hypothesis Testing")
    df = st.session_state.merged_df

    if df is not None:
        cols = df.select_dtypes(include=np.number).columns
        c1, c2 = st.columns(2)

        v1 = c1.selectbox("Variable 1", cols)
        v2 = c2.selectbox("Variable 2", cols)

        if st.button("Run T-Test"):
            t_stat, p_val = ttest_ind(df[v1].dropna(), df[v2].dropna())

            st.write(f"P-value: {p_val:.4f}")

            if p_val < 0.05:
                st.success("Statistically Significant")
            else:
                st.error("Not Significant")

    else:
        st.warning("Upload data first")

# ---------------- 4. RISK ANALYTICS ----------------
elif menu == "📉 Risk Analytics":
    st.title("📉 Outlier Detection")
    df = st.session_state.merged_df

    if df is not None:
        cols = df.select_dtypes(include=np.number).columns
        col = st.selectbox("Column", cols)

        z = np.abs(stats.zscore(df[col].dropna()))
        outliers = df.iloc[np.where(z > 3)]

        c1, c2 = st.columns(2)
        c1.metric("Outliers", len(outliers))
        c2.metric("Risk", "High" if len(outliers) > 5 else "Low")

        if not outliers.empty:
            st.dataframe(outliers)

    else:
        st.warning("Upload data first")

# ---------------- 5. ML ENGINE ----------------
elif menu == "🤖 ML Engine":
    st.title("🤖 Machine Learning")
    df = st.session_state.merged_df

    if df is not None:
        df = df.dropna()
        cols = df.select_dtypes(include=np.number).columns

        target = st.selectbox("Target", cols)
        feats = st.multiselect("Features", [c for c in cols if c != target])

        if st.button("Train Model") and feats:
            X_train, X_test, y_train, y_test = train_test_split(
                df[feats], df[target], test_size=0.2, random_state=42
            )

            model = LinearRegression().fit(X_train, y_train)
            preds = model.predict(X_test)

            st.metric("R² Score", f"{r2_score(y_test, preds):.2f}")

            fig = px.scatter(x=y_test, y=preds,
                             labels={"x": "Actual", "y": "Predicted"},
                             title="Actual vs Predicted")
            st.plotly_chart(fig)

    else:
        st.warning("Upload data first")

# ---------------- 6. SQL ----------------
elif menu == "🔍 SQL Workspace":
    st.title("🔍 SQL Lab")
    df = st.session_state.merged_df

    if df is not None:
        query = st.text_area("SQL Query", "SELECT * FROM df LIMIT 10")

        if st.button("Run"):
            try:
                result = ps.sqldf(query, {"df": df})
                st.dataframe(result)
            except Exception as e:
                st.error(e)

    else:
        st.warning("Upload data first")
