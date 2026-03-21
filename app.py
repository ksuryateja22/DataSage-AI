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
st.set_page_config(page_title="DataBoss AI Pro", layout="wide", page_icon="🧪")

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
    st.title("🛡️ DataBoss AI Pro")
    st.write("Analyst Edition")
    menu = st.radio("Modules", [
        "📁 Data Integration",
        "🧹 Stat-Cleaning",
        "📊 Visualization",
        "🔬 Inference Lab",
        "📉 Risk Analytics",
        "🤖 ML Engine",
        "🔍 SQL Workspace"
    ])

# ---------------- GLOBAL DATA VIEW ----------------
if st.session_state.merged_df is not None:
    st.subheader("📌 Current Dataset")
    df_preview = st.session_state.merged_df
    st.write(f"Rows: {df_preview.shape[0]}, Columns: {df_preview.shape[1]}")
    st.dataframe(df_preview)

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

        if len(dfs) == 1:
            st.session_state.merged_df = list(dfs.values())[0]
            st.success("Single dataset loaded.")

        if len(dfs) >= 2:
            st.subheader("🔗 Merge Datasets")

            keys = list(dfs.keys())
            t1 = st.selectbox("Dataset A", keys)
            t2 = st.selectbox("Dataset B", keys)

            join_type = st.selectbox(
                "Join Type",
                ["inner", "left", "right", "outer", "cross"]
            )

            if join_type != "cross":
                common_cols = list(set(dfs[t1].columns).intersection(set(dfs[t2].columns)))

                if common_cols:
                    join_col = st.selectbox("Join Column", common_cols)

                    if st.button("Merge"):
                        try:
                            merged = pd.merge(
                                dfs[t1],
                                dfs[t2],
                                on=join_col,
                                how=join_type
                            )
                            st.session_state.merged_df = merged.reset_index(drop=True)
                            st.success(f"{join_type.upper()} JOIN applied successfully.")
                        except Exception as e:
                            st.error(e)
                else:
                    st.error("No common columns found for join")

            else:
                if st.button("Merge (Cross Join)"):
                    try:
                        merged = dfs[t1].merge(dfs[t2], how="cross")
                        st.session_state.merged_df = merged.reset_index(drop=True)
                        st.success("CROSS JOIN applied successfully.")
                    except Exception as e:
                        st.error(e)

# ---------------- 2. STAT CLEANING ----------------
elif menu == "🧹 Stat-Cleaning":
    st.title("🧹 Statistical Cleaning")
    df = st.session_state.merged_df

    if df is not None and not df.empty:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            col = st.selectbox("Select Column", num_cols)

            if col in df.columns:
                skew_val = skew(df[col].dropna())
                st.metric("Skewness", f"{skew_val:.2f}")

                if abs(skew_val) > 0.5:
                    st.warning("High skew → Median preferred")
                else:
                    st.success("Normal distribution → Mean is fine")

                method = st.radio("Method", ["Mean", "Median", "Drop Nulls", "Remove Duplicates"])

                if st.button("Apply"):
                    if method == "Mean":
                        df[col] = df[col].fillna(df[col].mean())

                    elif method == "Median":
                        df[col] = df[col].fillna(df[col].median())

                    elif method == "Drop Nulls":
                        df = df.dropna(subset=[col])

                    elif method == "Remove Duplicates":
                        before = df.shape[0]
                        df = df.drop_duplicates()
                        after = df.shape[0]
                        st.success(f"Removed {before - after} duplicate rows")

                    st.session_state.merged_df = df.reset_index(drop=True)
                    st.success("Cleaning applied")
        else:
            st.warning("No numeric columns found")
    else:
        st.warning("Upload data first")

# ---------------- 3. VISUALIZATION ----------------
elif menu == "📊 Visualization":
    st.title("📊 Data Visualization")
    df = st.session_state.merged_df

    if df is not None and not df.empty:
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Histogram"])
        x = st.selectbox("X-axis", cols)
        y = st.selectbox("Y-axis", num_cols) if num_cols else None

        if st.button("Generate Chart"):
            try:
                if chart_type == "Bar":
                    fig = px.bar(df, x=x, y=y)
                elif chart_type == "Line":
                    fig = px.line(df, x=x, y=y)
                elif chart_type == "Scatter":
                    fig = px.scatter(df, x=x, y=y)
                else:
                    fig = px.histogram(df, x=x)

                st.plotly_chart(fig, width='stretch')
            except Exception as e:
                st.error(e)
    else:
        st.warning("Upload data first")

# ---------------- 4. INFERENCE ----------------
elif menu == "🔬 Inference Lab":
    st.title("🔬 Hypothesis Testing")
    df = st.session_state.merged_df

    if df is not None and not df.empty:
        cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(cols) >= 2:
            v1 = st.selectbox("Variable 1", cols)
            v2 = st.selectbox("Variable 2", cols)

            if st.button("Run T-Test"):
                try:
                    t_stat, p_val = ttest_ind(df[v1].dropna(), df[v2].dropna())
                    st.write(f"P-value: {p_val:.4f}")

                    if p_val < 0.05:
                        st.success("Statistically Significant")
                    else:
                        st.error("Not Significant")
                except Exception as e:
                    st.error(e)
        else:
            st.warning("Need at least 2 numeric columns")
    else:
        st.warning("Upload data first")

# ---------------- 5. RISK ANALYTICS ----------------
elif menu == "📉 Risk Analytics":
    st.title("📉 Outlier Detection")
    df = st.session_state.merged_df

    if df is not None and not df.empty:
        cols = df.select_dtypes(include=np.number).columns.tolist()

        if cols:
            col = st.selectbox("Column", cols)

            clean_series = df[col].dropna().reset_index(drop=True)
            z = np.abs(stats.zscore(clean_series))

            outliers = clean_series[z > 3]

            st.metric("Outliers", len(outliers))
            st.metric("Risk", "High" if len(outliers) > 5 else "Low")

            if not outliers.empty:
                st.dataframe(outliers.to_frame(name=col))
        else:
            st.warning("No numeric columns found")
    else:
        st.warning("Upload data first")

# ---------------- 6. ML ENGINE ----------------
elif menu == "🤖 ML Engine":
    st.title("🤖 Machine Learning")
    df = st.session_state.merged_df

    if df is not None and not df.empty:
        df = df.dropna().reset_index(drop=True)
        cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(cols) >= 2:
            target = st.selectbox("Target", cols)
            feats = st.multiselect("Features", [c for c in cols if c != target])

            if st.button("Train Model") and feats:
                try:
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

                except Exception as e:
                    st.error(e)
        else:
            st.warning("Need at least 2 numeric columns")
    else:
        st.warning("Upload data first")

# ---------------- 7. SQL ----------------
elif menu == "🔍 SQL Workspace":
    st.title("🔍 SQL Lab")
    df = st.session_state.merged_df

    if df is not None and not df.empty:
        query = st.text_area("SQL Query", "SELECT * FROM df LIMIT 10")

        if st.button("Run"):
            try:
                result = ps.sqldf(query, {"df": df})
                st.dataframe(result)
            except Exception as e:
                st.error(e)
    else:
        st.warning("Upload data first")
