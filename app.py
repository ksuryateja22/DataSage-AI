import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pandasql as ps
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="DataSage AI Pro", layout="wide", page_icon="🚀")

# ---------------- SESSION STATE ----------------
if "dfs" not in st.session_state:
    st.session_state.dfs = {}

if "merged_df" not in st.session_state:
    st.session_state.merged_df = None

if "figures" not in st.session_state:
    st.session_state.figures = []

# ---------------- HELPER FUNCTION ----------------
def get_active_df():
    if st.session_state.merged_df is not None:
        return st.session_state.merged_df
    elif st.session_state.dfs:
        return list(st.session_state.dfs.values())[0]
    else:
        return None

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛡️ DataSage AI Pro")

    menu = st.radio(
        "Navigation",
        [
            "Home & Upload",
            "Data Cleaning",
            "Visualization",
            "AI Insights",
            "SQL Lab",
            "ML Prediction",
            "Export PDF",
            "AI Prompt / Dynamic Joins",
        ],
    )

    st.markdown("---")
    st.info("Surya | 2026 Edition")

# ---------------- HOME & UPLOAD ----------------
if menu == "Home & Upload":

    st.title("🚀 Multi-CSV Upload & Management")

    uploaded_files = st.file_uploader(
        "Upload CSV files (1–5)", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            st.session_state.dfs[file.name] = pd.read_csv(file)

        st.success(f"{len(uploaded_files)} file(s) loaded successfully!")

    if st.session_state.dfs:

        st.subheader("Uploaded Dataset Preview")

        for name, df in st.session_state.dfs.items():
            st.write(f"**{name}** | Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            st.dataframe(df, use_container_width=True)

        if len(st.session_state.dfs) >= 2:

            st.subheader("🔗 Merge Datasets")

            df_names = list(st.session_state.dfs.keys())

            df1_name = st.selectbox("Select First Dataset", df_names)
            df2_name = st.selectbox("Select Second Dataset", df_names)

            on_col = st.text_input("Enter Column to Join On")

            join_type = st.selectbox(
                "Join Type", ["inner", "left", "right", "outer"]
            )

            if st.button("Merge Datasets"):

                try:
                    st.session_state.merged_df = pd.merge(
                        st.session_state.dfs[df1_name],
                        st.session_state.dfs[df2_name],
                        on=on_col,
                        how=join_type,
                    )

                    st.success("Datasets merged successfully!")

                    st.dataframe(
                        st.session_state.merged_df,
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(e)

# ---------------- DATA CLEANING ----------------
elif menu == "Data Cleaning":

    st.title("🧹 Professional Data Cleaning")

    df = get_active_df()

    if df is not None:

        tab1, tab2 = st.tabs(["Handle Null Values", "Remove Duplicates"])

        with tab1:

            st.write("Missing Values per Column")

            st.write(df.isnull().sum()[df.isnull().sum() > 0])

            method = st.selectbox("Fill Method", ["Mean", "Median", "Drop Rows"])

            if st.button("Apply Null Handling"):

                if method == "Mean":
                    df = df.fillna(df.mean(numeric_only=True))

                elif method == "Median":
                    df = df.fillna(df.median(numeric_only=True))

                else:
                    df = df.dropna()

                st.session_state.merged_df = df

                st.success(f"Missing values handled using {method}")

        with tab2:

            dupes = df.duplicated().sum()

            st.write(f"Found {dupes} duplicate rows")

            if st.button("Remove Duplicates"):

                df = df.drop_duplicates()

                st.session_state.merged_df = df

                st.success("Duplicates removed")

        st.subheader("Cleaned Dataset Preview")

        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Cleaned CSV",
            csv,
            "cleaned_data.csv",
            "text/csv",
        )

    else:
        st.warning("Upload dataset first")

# ---------------- VISUALIZATION ----------------
elif menu == "Visualization":

    st.title("📊 Interactive Visualization")

    df = get_active_df()

    if df is not None:

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        all_cols = df.columns.tolist()

        chart_type = st.selectbox(
            "Chart Type",
            [
                "Bar Chart",
                "Histogram",
                "Scatter Plot",
                "Line Chart",
                "Correlation Heatmap",
            ],
        )

        fig = None

        if chart_type == "Bar Chart":

            x = st.selectbox("X Axis", all_cols)
            y = st.selectbox("Y Axis", num_cols)

            fig = px.bar(df, x=x, y=y, template="plotly_white")

        elif chart_type == "Histogram":

            col = st.selectbox("Column", num_cols)

            fig = px.histogram(df, x=col)

        elif chart_type == "Scatter Plot":

            x = st.selectbox("X Axis", num_cols)
            y = st.selectbox("Y Axis", num_cols)

            fig = px.scatter(df, x=x, y=y)

        elif chart_type == "Line Chart":

            y = st.selectbox("Column", num_cols)

            fig = px.line(df, y=y)

        elif chart_type == "Correlation Heatmap":

            fig = px.imshow(
                df[num_cols].corr(),
                text_auto=True,
                color_continuous_scale="RdBu_r",
            )

        if fig:

            st.plotly_chart(fig, use_container_width=True)

            if st.button("Save Chart"):
                st.session_state.figures.append(fig)
                st.success("Chart saved for PDF report")

        if st.button("Clear Saved Charts"):
            st.session_state.figures = []

# ---------------- AI INSIGHTS ----------------
elif menu == "AI Insights":

    st.title("💡 Automated Data Insights")

    df = get_active_df()

    if df is not None:

        if st.button("Generate Insights"):

            for col in df.columns:

                with st.expander(f"{col}"):

                    st.write(df[col].describe())

                    if pd.api.types.is_numeric_dtype(df[col]):

                        skew = df[col].skew()

                        if skew > 1:
                            st.warning("Highly Right Skewed")

                        elif skew < -1:
                            st.warning("Highly Left Skewed")

                        else:
                            st.success("Approximately Normal Distribution")

# ---------------- SQL LAB ----------------
elif menu == "SQL Lab":

    st.title("🔍 SQL Query Workspace")

    df = get_active_df()

    if df is not None:

        query = st.text_area("Write SQL Query", "SELECT * FROM df LIMIT 5")

        if st.button("Run SQL"):

            try:
                res = ps.sqldf(query, {"df": df})

                st.dataframe(res, use_container_width=True)

            except Exception as e:
                st.error(e)

# ---------------- ML PREDICTION ----------------
elif menu == "ML Prediction":

    st.title("🤖 ML Prediction Engine")

    df = get_active_df()

    if df is not None:

        df = df.dropna()

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:

            target = st.selectbox("Target Column", num_cols)

            features = st.multiselect(
                "Feature Columns",
                [c for c in num_cols if c != target],
            )

            if st.button("Train Model") and features:

                model = LinearRegression()

                model.fit(df[features], df[target])

                score = model.score(df[features], df[target])

                st.metric("R² Score", f"{score:.2f}")

                st.success("Model trained successfully")

# ---------------- EXPORT PDF ----------------
elif menu == "Export PDF":

    st.title("📄 Export PDF Report")

    if st.session_state.figures:

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font("Arial", size=14)

        pdf.cell(0, 10, "DataSage AI Report", ln=True)

        for fig in st.session_state.figures:

            img_bytes = fig.to_image(format="png")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:

                tmp.write(img_bytes)

                pdf.image(tmp.name, x=10, w=pdf.w - 20)

        file = "DataSage_AI_Report.pdf"

        pdf.output(file)

        with open(file, "rb") as f:

            st.download_button(
                "📥 Download PDF",
                f,
                file,
                mime="application/pdf",
            )

    else:
        st.warning("Save charts first from Visualization tab")

# ---------------- AI PROMPT ----------------
elif menu == "AI Prompt / Dynamic Joins":

    st.title("🤖 Natural Language Data Queries")

    df = get_active_df()

    if df is not None:

        query = st.text_area(
            "Ask a question about your dataset",
            "top 5 values",
        )

        if st.button("Run Query"):

            q = query.lower()

            if "top" in q:

                n = int("".join(filter(str.isdigit, q)) or 5)

                col = df.select_dtypes(include=np.number).columns[0]

                st.dataframe(df.nlargest(n, col))

            elif "mean" in q:

                st.dataframe(
                    df.mean(numeric_only=True).to_frame("Mean")
                )

            elif "max" in q:

                st.dataframe(
                    df.max(numeric_only=True).to_frame("Max")
                )

            elif "min" in q:

                st.dataframe(
                    df.min(numeric_only=True).to_frame("Min")
                )

            else:

                st.info("Query not recognized")

    else:
        st.warning("Upload dataset first")
