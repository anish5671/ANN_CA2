import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import plotly.express as px

st.set_page_config(page_title="ML Pipeline", layout="wide")

st.title("ML Auto Pipeline")

menu = st.sidebar.radio("Menu", ["Upload", "EDA", "Preprocess", "Model"])


# ---------- FUNCTIONS ----------

def preprocess(df):
    df = df.copy()

    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Missing")

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    for col in df.select_dtypes(include=['object', 'string']):
        if df[col].nunique() < 50:
            df[col] = df[col].astype('category').cat.codes
        else:
            df.drop(col, axis=1, inplace=True)

    return df


def detect_problem(y):
    if y.dtype in ['int64', 'float64'] and y.nunique() > 20:
        return "Regression"
    else:
        return "Classification"


def scale(X):
    try:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        return pd.DataFrame(X)
    except:
        return X


def get_models(problem):
    if problem == "Classification":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        return {
            "Random Forest": RandomForestClassifier(),
            "KNN": KNeighborsClassifier()
        }
    else:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        return {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression()
        }


def train(X_train, X_test, y_train, y_test, problem):
    models = get_models(problem)
    results = {}

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if problem == "Classification":
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_test, preds)
            else:
                from sklearn.metrics import r2_score
                score = r2_score(y_test, preds)

            results[name] = round(score, 3)

        except:
            results[name] = "Error"

    return results


# ---------- UPLOAD ----------

file = st.sidebar.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
else:
    df = None


# ---------- UI ----------

if menu == "Upload":
    if df is not None:
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
    else:
        st.warning("Upload file")


# ---------- EDA ----------
elif menu == "EDA":
    if df is not None:
        col = st.selectbox("Column", df.columns)

        # Histogram
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

        # ✅ Correlation Matrix
        num_df = df.select_dtypes(include=np.number)

        if num_df.shape[1] > 1:
            st.subheader("Correlation Matrix")
            corr = num_df.corr()

            fig2 = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig2)
        else:
            st.info("Not enough numeric columns")

    else:
        st.warning("Upload file")


# ---------- PREPROCESS ----------
elif menu == "Preprocess":
    if df is not None:
        if st.button("Run"):
            df = preprocess(df)
            st.success("Done")

        st.dataframe(df.head())

        st.download_button("Download", df.to_csv(index=False), "clean.csv")
    else:
        st.warning("Upload file")


# ---------- MODEL ----------
elif menu == "Model":
    if df is not None:
        df = preprocess(df)

        target = st.selectbox("Target", df.columns)

        X = df.drop(target, axis=1)
        y = df[target]

        X = scale(X)

        # ✅ NEW: manual + auto
        problem_choice = st.selectbox(
            "Select Problem Type",
            ["Auto", "Classification", "Regression"]
        )

        if problem_choice == "Auto":
            problem = detect_problem(y)
        else:
            problem = problem_choice

        st.write("Problem Type:", problem)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        if st.button("Train"):
            res = train(X_train, X_test, y_train, y_test, problem)

            st.write(res)

            fig = px.bar(x=list(res.keys()), y=list(res.values()))
            st.plotly_chart(fig)

    else:
        st.warning("Upload file")