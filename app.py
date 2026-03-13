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
if 'dfs' not in st.session_state:
    st.session_state.dfs = {}
if 'merged_df' not in st.session_state:
    st.session_state.merged_df = None
if 'figures' not in st.session_state:
    st.session_state.figures = []

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🛡️ DataSage AI Pro")
    menu = st.radio("Navigation", ["Home & Upload", "Data Cleaning", "Visualization", "AI Insights", 
                                   "SQL Lab", "ML Prediction", "Export PDF", "AI Prompt / Dynamic Joins"])
    st.markdown("---")
    st.info("surya | 2026 Edition")

# ---------------- 1. HOME & UPLOAD ----------------
if menu == "Home & Upload":
    st.title("🚀 Multi-CSV Upload & Management")
    uploaded_files = st.file_uploader("Upload CSV files (1–5)", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        for file in uploaded_files:
            st.session_state.dfs[file.name] = pd.read_csv(file)
        st.success(f"{len(uploaded_files)} file(s) loaded successfully!")

    if st.session_state.dfs:
        st.subheader("Uploaded Datasets Preview")
        for name, df in st.session_state.dfs.items():
            st.write(f"**{name}** - Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            st.dataframe(df, width='stretch')

        if len(st.session_state.dfs) >= 2:
            st.subheader("🔗 Merge Datasets")
            df_names = list(st.session_state.dfs.keys())
            df1_name = st.selectbox("Select First Dataset", df_names)
            df2_name = st.selectbox("Select Second Dataset", df_names)
            on_col = st.text_input("Enter Column to Join On")
            join_type = st.selectbox("Join Type", ["inner", "left", "right", "outer"])
            if st.button("Merge Datasets"):
                try:
                    st.session_state.merged_df = pd.merge(
                        st.session_state.dfs[df1_name],
                        st.session_state.dfs[df2_name],
                        on=on_col, how=join_type
                    )
                    st.success("Datasets merged successfully!")
                    st.dataframe(st.session_state.merged_df, width='stretch')
                except Exception as e:
                    st.error(f"Merge Error: {e}")

# ---------------- 2. DATA CLEANING ----------------
elif menu == "Data Cleaning":
    st.title("🧹 Professional Data Cleaning")
    df = st.session_state.merged_df or (list(st.session_state.dfs.values())[0] if st.session_state.dfs else None)
    if df is not None:
        tab1, tab2 = st.tabs(["Handle Null Values", "Remove Duplicates"])
        
        with tab1:
            st.write("Missing values per column:")
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
                st.success(f"Missing values handled with {method}")
        
        with tab2:
            dupes = df.duplicated().sum()
            st.write(f"Found {dupes} duplicate rows")
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.session_state.merged_df = df
                st.success("Duplicates removed")
        
        st.subheader("Cleaned Dataset Preview")
        st.dataframe(df, width='stretch')
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Cleaned CSV", csv, "cleaned_data.csv", "text/csv")
    else:
        st.warning("Upload or merge a dataset first")

# ---------------- 3. VISUALIZATION ----------------
elif menu == "Visualization":
    st.title("📊 Interactive Visualization")
    df = st.session_state.merged_df or (list(st.session_state.dfs.values())[0] if st.session_state.dfs else None)
    if df is not None:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        all_cols = df.columns.tolist()
        chart_type = st.selectbox("Select Chart Type", ["Bar Chart", "Histogram", "Scatter Plot", "Line Chart", "Correlation Heatmap"])
        
        fig = None
        if chart_type == "Bar Chart":
            x_col = st.selectbox("X-Axis", all_cols)
            y_col = st.selectbox("Y-Axis", num_cols)
            fig = px.bar(df, x=x_col, y=y_col, template="plotly_white")
        elif chart_type == "Histogram":
            col = st.selectbox("Column", num_cols)
            fig = px.histogram(df, x=col, nbins=30, template="plotly_white")
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("X-Axis", num_cols)
            y_col = st.selectbox("Y-Axis", num_cols)
            fig = px.scatter(df, x=x_col, y=y_col, template="plotly_white")
        elif chart_type == "Line Chart":
            y_col = st.selectbox("Column", num_cols)
            fig = px.line(df, y=y_col, template="plotly_white")
        elif chart_type == "Correlation Heatmap":
            fig = px.imshow(df[num_cols].corr(), text_auto=True, color_continuous_scale='RdBu_r')

        if fig:
            st.plotly_chart(fig, width='stretch')
            st.session_state.figures.append(fig)

# ---------------- 4. AI INSIGHTS ----------------
elif menu == "AI Insights":
    st.title("💡 AI-Generated Insights")
    df = st.session_state.merged_df or (list(st.session_state.dfs.values())[0] if st.session_state.dfs else None)
    if df is not None:
        n_rows = st.number_input("Number of top/last rows to show", min_value=3, max_value=20, value=5)
        num_cols = df.columns
        if st.button("Generate Insights"):
            for col in num_cols:
                with st.expander(f"Column: {col}"):
                    st.write(df[col].describe())
                    # Top / Last
                    if pd.api.types.is_numeric_dtype(df[col]):
                        st.write(f"Top {n_rows} values:")
                        st.dataframe(df.nlargest(n_rows, col)[[col]])
                        st.write(f"Last {n_rows} values:")
                        st.dataframe(df.nsmallest(n_rows, col)[[col]])
                    else:
                        st.write(f"Top {n_rows} values:")
                        st.dataframe(df.sort_values(col, ascending=False).head(n_rows)[[col]])
                        st.write(f"Last {n_rows} values:")
                        st.dataframe(df.sort_values(col, ascending=True).head(n_rows)[[col]])
                    # Skewness
                    if pd.api.types.is_numeric_dtype(df[col]):
                        skewness = df[col].skew()
                        if skewness > 1:
                            st.warning("Highly right-skewed")
                        elif skewness < -1:
                            st.warning("Highly left-skewed")
                        else:
                            st.success("Relatively normal distribution")

# ---------------- 5. SQL LAB ----------------
elif menu == "SQL Lab":
    st.title("🔍 SQL Query Workspace")
    table_options = list(st.session_state.dfs.keys())
    if st.session_state.merged_df is not None:
        table_options.append("merged_df")
    table_name = st.selectbox("Select Table", table_options)
    
    df = st.session_state.dfs.get(table_name) if table_name != "merged_df" else st.session_state.merged_df
    query = st.text_area("Write SQL (use table name as selected above)", "SELECT * FROM df LIMIT 5")
    
    if st.button("Run SQL"):
        try:
            res = ps.sqldf(query, { "df": df })
            st.dataframe(res, width='stretch')
        except Exception as e:
            st.error(f"SQL Error: {e}")

# ---------------- 6. ML PREDICTION ----------------
elif menu == "ML Prediction":
    st.title("🤖 ML Prediction Engine")
    df = st.session_state.merged_df or (list(st.session_state.dfs.values())[0] if st.session_state.dfs else None)
    if df is not None:
        df = df.dropna()
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            target = st.selectbox("Target Column", num_cols)
            features = st.multiselect("Feature Columns", [c for c in num_cols if c != target])
            if st.button("Train Model") and features:
                model = LinearRegression().fit(df[features], df[target])
                st.metric("R² Score", f"{model.score(df[features], df[target]):.2f}")
                st.success("Model trained successfully!")
        else:
            st.error("Need at least 2 numeric columns for prediction")

# ---------------- 7. EXPORT PDF ----------------
elif menu == "Export PDF":
    st.title("📄 Export PDF Report")
    if st.session_state.figures:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, "DataSage AI Report", ln=True)
        pdf.cell(0, 10, f"Generated {pd.Timestamp.now()}", ln=True)

        for fig in st.session_state.figures:
            st.plotly_chart(fig, use_container_width=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                tmpfile.write(img_bytes)
                tmpfile_path = tmpfile.name
            pdf.image(tmpfile_path, x=10, w=pdf.w - 20)

        pdf_file = "DataSage_AI_Report.pdf"
        pdf.output(pdf_file)
        st.success("PDF generated successfully!")
        with open(pdf_file, "rb") as f:
            st.download_button("📥 Download PDF Report", f, pdf_file)
    else:
        st.warning("Generate at least one chart in Visualization tab first")

# ---------------- 8. AI PROMPT / DYNAMIC JOINS ----------------
elif menu == "AI Prompt / Dynamic Joins":
    st.title("🤖 AI-Powered Queries & Dynamic Joins")
    df_names = list(st.session_state.dfs.keys())
    df_main = st.session_state.merged_df or (list(st.session_state.dfs.values())[0] if st.session_state.dfs else None)
    
    if df_main is not None:
        user_query = st.text_area("Type your query (e.g., 'Top 5 sales', 'Join df1 and df2 on column X')", height=100)
        if st.button("Run AI Query"):
            try:
                query = user_query.lower()
                
                # ----------------- TOP / LAST -----------------
                if "top" in query or "last" in query:
                    n = int(''.join(filter(str.isdigit, query))) or 5
                    col_candidates = [c for c in df_main.columns if c.lower() in query]
                    col = col_candidates[0] if col_candidates else df_main.columns[0]
                    
                    if pd.api.types.is_numeric_dtype(df_main[col]):
                        if "top" in query:
                            st.dataframe(df_main.nlargest(n, col))
                        else:
                            st.dataframe(df_main.nsmallest(n, col))
                    else:
                        if "top" in query:
                            st.dataframe(df_main.sort_values(col, ascending=False).head(n))
                        else:
                            st.dataframe(df_main.sort_values(col, ascending=True).head(n))
                
                # ----------------- MEAN / MAX / MIN -----------------
                elif any(k in query for k in ["mean","max","min"]):
                    if "mean" in query:
                        st.dataframe(df_main.mean(numeric_only=True).to_frame("Mean"))
                    if "max" in query:
                        st.dataframe(df_main.max(numeric_only=True).to_frame("Max"))
                    if "min" in query:
                        st.dataframe(df_main.min(numeric_only=True).to_frame("Min"))
                
                # ----------------- DYNAMIC JOIN -----------------
                elif "join" in query and len(st.session_state.dfs) >= 2:
                    words = query.split()
                    df1_name = next((name for name in df_names if name.lower() in words), df_names[0])
                    df2_name = next((name for name in df_names if name.lower() in words and name != df1_name), df_names[1])
                    on_cols = [word for word in words if word in st.session_state.dfs[df1_name].columns and word in st.session_state.dfs[df2_name].columns]
                    on_col = on_cols[0] if on_cols else st.session_state.dfs[df1_name].columns[0]
                    st.session_state.merged_df = pd.merge(
                        st.session_state.dfs[df1_name],
                        st.session_state.dfs[df2_name],
                        on=on_col,
                        how="inner"
                    )
                    st.success(f"Datasets merged dynamically on {on_col}")
                    st.dataframe(st.session_state.merged_df, width='stretch')
                else:
                    st.info("Query not recognized. Try keywords: top, last, mean, max, min, join.")
            except Exception as e:
                st.error(f"Error running AI query: {e}")
    else:
        st.warning("Upload at least one dataset first.")
