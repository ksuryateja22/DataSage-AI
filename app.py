import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pandasql as ps
from sklearn.linear_model import LinearRegression
from fpdf import FPDF

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

    st.title("🚀 Multi CSV Upload")

    uploaded_files = st.file_uploader(
        "Upload CSV files", type="csv", accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            st.session_state.dfs[file.name] = pd.read_csv(file)

        st.success(f"{len(uploaded_files)} dataset(s) uploaded")

    if st.session_state.dfs:

        st.subheader("Dataset Preview")

        for name, df in st.session_state.dfs.items():

            st.write(f"**{name}**")

            st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

            st.dataframe(df, use_container_width=True)

        if len(st.session_state.dfs) >= 2:

            st.subheader("Merge Datasets")

            df_names = list(st.session_state.dfs.keys())

            df1 = st.selectbox("Dataset 1", df_names)
            df2 = st.selectbox("Dataset 2", df_names)

            join_col = st.text_input("Join Column")

            join_type = st.selectbox(
                "Join Type", ["inner", "left", "right", "outer"]
            )

            if st.button("Merge"):

                try:

                    st.session_state.merged_df = pd.merge(
                        st.session_state.dfs[df1],
                        st.session_state.dfs[df2],
                        on=join_col,
                        how=join_type,
                    )

                    st.success("Datasets merged")

                    st.dataframe(
                        st.session_state.merged_df,
                        use_container_width=True,
                    )

                except Exception as e:

                    st.error(e)

# ---------------- DATA CLEANING ----------------
elif menu == "Data Cleaning":

    st.title("Data Cleaning")

    df = get_active_df()

    if df is not None:

        st.write("Missing Values")

        st.write(df.isnull().sum())

        method = st.selectbox(
            "Handle Missing Values", ["Mean", "Median", "Drop Rows"]
        )

        if st.button("Apply Cleaning"):

            if method == "Mean":

                df = df.fillna(df.mean(numeric_only=True))

            elif method == "Median":

                df = df.fillna(df.median(numeric_only=True))

            else:

                df = df.dropna()

            st.session_state.merged_df = df

            st.success("Cleaning Applied")

        st.write("Duplicate Rows:", df.duplicated().sum())

        if st.button("Remove Duplicates"):

            df = df.drop_duplicates()

            st.session_state.merged_df = df

            st.success("Duplicates Removed")

        st.dataframe(df, use_container_width=True)

    else:

        st.warning("Upload dataset first")

# ---------------- VISUALIZATION ----------------
elif menu == "Visualization":

    st.title("Visualization")

    df = get_active_df()

    if df is not None:

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        all_cols = df.columns.tolist()

        chart = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Histogram", "Scatter Plot", "Line Chart"],
        )

        fig = None

        if chart == "Bar Chart":

            x = st.selectbox("X Axis", all_cols)

            y = st.selectbox("Y Axis", num_cols)

            fig = px.bar(df, x=x, y=y)

        elif chart == "Histogram":

            col = st.selectbox("Column", num_cols)

            fig = px.histogram(df, x=col)

        elif chart == "Scatter Plot":

            x = st.selectbox("X Axis", num_cols)

            y = st.selectbox("Y Axis", num_cols)

            fig = px.scatter(df, x=x, y=y)

        elif chart == "Line Chart":

            y = st.selectbox("Column", num_cols)

            fig = px.line(df, y=y)

        if fig:

            st.plotly_chart(fig, use_container_width=True)

            if st.button("Save Chart"):

                st.session_state.figures.append(fig)

                st.success("Chart saved")

# ---------------- AI INSIGHTS ----------------
elif menu == "AI Insights":

    st.title("AI Insights")

    df = get_active_df()

    if df is not None:

        if st.button("Generate Insights"):

            for col in df.columns:

                with st.expander(col):

                    st.write(df[col].describe())

                    if pd.api.types.is_numeric_dtype(df[col]):

                        skew = df[col].skew()

                        if skew > 1:

                            st.warning("Right Skewed")

                        elif skew < -1:

                            st.warning("Left Skewed")

                        else:

                            st.success("Normal Distribution")

# ---------------- SQL LAB ----------------
elif menu == "SQL Lab":

    st.title("SQL Query Lab")

    df = get_active_df()

    if df is not None:

        query = st.text_area(
            "SQL Query", "SELECT * FROM df LIMIT 5"
        )

        if st.button("Run SQL"):

            try:

                result = ps.sqldf(query, {"df": df})

                st.dataframe(result, use_container_width=True)

            except Exception as e:

                st.error(e)

# ---------------- ML PREDICTION ----------------
elif menu == "ML Prediction":

    st.title("ML Prediction")

    df = get_active_df()

    if df is not None:

        df = df.dropna()

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) >= 2:

            target = st.selectbox("Target", num_cols)

            features = st.multiselect(
                "Features", [c for c in num_cols if c != target]
            )

            if st.button("Train Model") and features:

                model = LinearRegression()

                model.fit(df[features], df[target])

                score = model.score(df[features], df[target])

                st.metric("R² Score", f"{score:.2f}")

                st.success("Model trained")

# ---------------- EXPORT PDF ----------------
elif menu == "Export PDF":

    st.title("Export PDF")

    df = get_active_df()

    if df is not None:

        if st.button("Generate Report"):

            pdf = FPDF()

            pdf.add_page()

            pdf.set_font("Arial", size=16)

            pdf.cell(0, 10, "DataSage AI Report", ln=True)

            pdf.set_font("Arial", size=10)

            pdf.cell(0, 10, f"Rows: {df.shape[0]}", ln=True)

            pdf.cell(0, 10, f"Columns: {df.shape[1]}", ln=True)

            pdf.ln(10)

            pdf.cell(0, 10, "Columns:", ln=True)

            for col in df.columns:

                pdf.cell(0, 8, col, ln=True)

            file = "datasage_report.pdf"

            pdf.output(file)

            with open(file, "rb") as f:

                st.download_button(
                    "Download PDF",
                    f,
                    file,
                    mime="application/pdf",
                )

# ---------------- AI QUERY ----------------
elif menu == "AI Prompt / Dynamic Joins":

    st.title("Natural Language Query")

    df = get_active_df()

    if df is not None:

        q = st.text_area("Ask question", "top 5 rows")

        if st.button("Run Query"):

            query = q.lower()

            if "top" in query:

                n = int("".join(filter(str.isdigit, query)) or 5)

                st.dataframe(df.head(n))

            elif "mean" in query:

                st.dataframe(
                    df.mean(numeric_only=True).to_frame("Mean")
                )

            elif "max" in query:

                st.dataframe(
                    df.max(numeric_only=True).to_frame("Max")
                )

            elif "min" in query:

                st.dataframe(
                    df.min(numeric_only=True).to_frame("Min")
                )

            else:

                st.info("Query not recognized")

    else:

        st.warning("Upload dataset first")
