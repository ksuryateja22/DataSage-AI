import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pandasql as ps
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="DataSage AI Pro", layout="wide")

# ---------------- SESSION STORAGE ----------------

if "dataframes" not in st.session_state:
    st.session_state.dataframes = {}

if "merged_df" not in st.session_state:
    st.session_state.merged_df = None

if "charts" not in st.session_state:
    st.session_state.charts = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚀 DataSage AI Pro")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Upload Data",
        "Data Cleaning",
        "Visualization",
        "AI Insights",
        "SQL Lab",
        "ML Prediction",
        "AI Query",
        "Chat with AI"
    ],
)

# ---------------- HELPER ----------------

def get_df():
    if st.session_state.merged_df is not None:
        return st.session_state.merged_df
    elif st.session_state.dataframes:
        return list(st.session_state.dataframes.values())[0]
    else:
        return None


# ---------------- UPLOAD ----------------

if menu == "Upload Data":

    st.title("📂 Upload CSV Dataset")

    files = st.file_uploader(
        "Upload CSV Files", type=["csv"], accept_multiple_files=True
    )

    if files:

        for file in files:
            df = pd.read_csv(file)
            st.session_state.dataframes[file.name] = df

        st.success("Files uploaded successfully")

    for name, df in st.session_state.dataframes.items():

        st.subheader(name)

        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")

        st.dataframe(df)


# ---------------- DATA CLEANING ----------------

elif menu == "Data Cleaning":

    st.title("🧹 Data Cleaning")

    df = get_df()

    if df is not None:

        st.write("Missing Values")

        st.write(df.isnull().sum())

        option = st.selectbox(
            "Handle Missing Values",
            ["Fill Mean", "Fill Median", "Drop Rows"],
        )

        if st.button("Apply Cleaning"):

            if option == "Fill Mean":
                df = df.fillna(df.mean(numeric_only=True))

            elif option == "Fill Median":
                df = df.fillna(df.median(numeric_only=True))

            else:
                df = df.dropna()

            st.session_state.merged_df = df

            st.success("Cleaning Applied")

        st.write("Duplicate rows:", df.duplicated().sum())

        if st.button("Remove Duplicates"):

            df = df.drop_duplicates()

            st.session_state.merged_df = df

            st.success("Duplicates Removed")

        st.dataframe(df)

    else:

        st.warning("Upload dataset first")


# ---------------- VISUALIZATION ----------------

elif menu == "Visualization":

    st.title("📊 Data Visualization")

    df = get_df()

    if df is not None:

        num_cols = df.select_dtypes(include=np.number).columns
        all_cols = df.columns

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

            x = st.selectbox("X", num_cols)
            y = st.selectbox("Y", num_cols)

            fig = px.scatter(df, x=x, y=y)

        elif chart == "Line Chart":

            col = st.selectbox("Column", num_cols)

            fig = px.line(df, y=col)

        if fig:

            st.plotly_chart(fig, use_container_width=True)

    else:

        st.warning("Upload dataset first")


# ---------------- AI INSIGHTS ----------------

elif menu == "AI Insights":

    st.title("🧠 AI Insights")

    df = get_df()

    if df is not None:

        if st.button("Generate Insights"):

            st.write(df.describe())

            for col in df.select_dtypes(include=np.number).columns:

                skew = df[col].skew()

                if skew > 1:
                    st.warning(f"{col} is right skewed")

                elif skew < -1:
                    st.warning(f"{col} is left skewed")

                else:
                    st.success(f"{col} is normally distributed")

    else:

        st.warning("Upload dataset first")


# ---------------- SQL LAB ----------------

elif menu == "SQL Lab":

    st.title("🗄️ SQL Query Lab")

    df = get_df()

    if df is not None:

        query = st.text_area(
            "Write SQL Query",
            "SELECT * FROM df LIMIT 10",
        )

        if st.button("Run Query"):

            try:

                result = ps.sqldf(query, {"df": df})

                st.dataframe(result)

            except Exception as e:

                st.error(e)

    else:

        st.warning("Upload dataset first")


# ---------------- ML PREDICTION ----------------

elif menu == "ML Prediction":

    st.title("🤖 Machine Learning Prediction")

    df = get_df()

    if df is not None:

        df = df.dropna()

        num_cols = df.select_dtypes(include=np.number).columns

        if len(num_cols) >= 2:

            target = st.selectbox("Target Column", num_cols)

            features = st.multiselect(
                "Feature Columns",
                [c for c in num_cols if c != target],
            )

            if st.button("Train Model") and features:

                X = df[features]
                y = df[target]

                model = LinearRegression()

                model.fit(X, y)

                score = model.score(X, y)

                st.metric("Model R² Score", round(score, 2))

                st.success("Model trained successfully")

    else:

        st.warning("Upload dataset first")


# ---------------- AI QUERY ----------------

elif menu == "AI Query":

    st.title("🔍 Natural Language Query")

    df = get_df()

    if df is not None:

        question = st.text_input("Ask question about dataset")

        if st.button("Run Query"):

            q = question.lower()

            if "top" in q:

                n = int("".join(filter(str.isdigit, q)) or 5)

                st.dataframe(df.head(n))

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

                st.warning("Query not recognized")

    else:

        st.warning("Upload dataset first")


# ---------------- CHAT WITH AI ----------------

elif menu == "Chat with AI":

    st.title("🤖 Chat with Your Data")

    df = get_df()

    if df is not None:

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        user_input = st.text_input("Ask something about the dataset")

        if st.button("Ask AI"):

            question = user_input.lower()

            response = None

            if "rows" in question:
                response = f"The dataset has {df.shape[0]} rows."

            elif "columns" in question:
                response = f"The dataset has {df.shape[1]} columns."

            elif "top" in question:
                response = df.head()

            elif "summary" in question:
                response = df.describe()

            elif "mean" in question:
                response = df.mean(numeric_only=True)

            elif "max" in question:
                response = df.max(numeric_only=True)

            elif "min" in question:
                response = df.min(numeric_only=True)

            else:
                response = "Sorry, I don't understand that question yet."

            st.session_state.chat_history.append(("You", user_input))
            st.session_state.chat_history.append(("AI", response))

        for speaker, message in st.session_state.chat_history:

            if speaker == "You":
                st.write(f"🧑 **You:** {message}")
            else:
                st.write("🤖 **AI:**")
                st.write(message)

    else:

        st.warning("Upload dataset first")
