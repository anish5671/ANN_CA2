import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import plotly.express as px

st.set_page_config(page_title="ML Pipeline", layout="wide")
st.title("🚀 ML Auto Pipeline")

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


# ---------- UPLOAD ----------

file = st.sidebar.file_uploader("Upload CSV")

if file:
    df = pd.read_csv(file)
else:
    df = None


# ---------- UI ----------

# UPLOAD
if menu == "Upload":
    if df is not None:
        st.dataframe(df.head())
        st.write("Shape:", df.shape)
    else:
        st.warning("Upload file")


# EDA
elif menu == "EDA":
    if df is not None:
        col = st.selectbox("Column", df.columns)

        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

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


# PREPROCESS
elif menu == "Preprocess":
    if df is not None:
        if st.button("Run Preprocessing"):
            df = preprocess(df)
            st.success("Preprocessing Done")

        st.dataframe(df.head())
        st.download_button("Download Clean Data", df.to_csv(index=False), "clean.csv")

    else:
        st.warning("Upload file")


# MODEL
elif menu == "Model":
    if df is not None:
        df = preprocess(df)

        target = st.selectbox("Target Column", df.columns)

        X = df.drop(target, axis=1)
        y = df[target]

        # ---------- FEATURE SELECTION (FIXED) ----------
        temp_df = df.copy()

        if temp_df[target].dtype == "object":
            temp_df[target] = LabelEncoder().fit_transform(temp_df[target].astype(str))

        for col in temp_df.columns:
            try:
                temp_df[col] = pd.to_numeric(temp_df[col])
            except:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col].astype(str))

        corr = temp_df.corr()

        if target in corr.columns:
            top_features = corr[target].abs().sort_values(ascending=False).index[1:6]
            if len(top_features) == 0:
                top_features = X.columns[:5]
        else:
            top_features = X.columns[:5]

        X = X[top_features]

        st.write("Selected Features:", list(top_features))

        # ---------- SCALING ----------
        X = scale(X)

        # ---------- PROBLEM TYPE ----------
        choice = st.selectbox("Problem Type", ["Auto", "Classification", "Regression"])

        if choice == "Auto":
            problem = detect_problem(y)
        else:
            problem = choice

        st.write("Problem:", problem)

        models = get_models(problem)

        X_train, X_test, y_train, y_test = train_test_split(X, y)

        # ---------- TRAIN ----------
        if st.button("Train Models"):
            results = {}

            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=5)
                    results[name] = round(scores.mean(), 3)
                except:
                    results[name] = "Error"

            st.write(results)

            fig = px.bar(x=list(results.keys()), y=list(results.values()))
            st.plotly_chart(fig)

    else:
        st.warning("Upload file")