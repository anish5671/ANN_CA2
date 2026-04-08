import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import plotly.express as px


# ---------------- UI ----------------
st.set_page_config(page_title="ML Pipeline", layout="wide")
st.title("🚀 Machine Learning Pipeline Dashboard")

menu = st.sidebar.radio("Navigation", [
    "Upload Data",
    "EDA",
    "Preprocessing",
    "Model Training"
])


# ---------------- Functions ----------------

def clean_data(df):
    df = df.drop_duplicates()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())

    return df


def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=np.number)

    Q1 = numeric_cols.quantile(0.25)
    Q3 = numeric_cols.quantile(0.75)
    IQR = Q3 - Q1

    mask = ~((numeric_cols < (Q1 - 1.5 * IQR)) |
             (numeric_cols > (Q3 + 1.5 * IQR))).any(axis=1)

    return df[mask]


def drop_text_columns(X):
    X = X.copy()
    for col in X.columns:
        if X[col].dtype == 'object' and X[col].nunique() > 50:
            X = X.drop(col, axis=1)
    return X


def encode_data(X):
    X = X.copy()
    for col in X.select_dtypes(include=['object']):
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    return X


def handle_missing(X):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns)


def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)


def train_models(X_train, X_test, y_train, y_test):
    models = {
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = acc

    return results


# ---------------- Upload ----------------

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = None


# ---------------- Upload Section ----------------

if menu == "Upload Data":

    st.header("Dataset Overview")

    if df is not None:
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

    else:
        st.info("Upload a CSV file.")


# ---------------- EDA ----------------

elif menu == "EDA":

    st.header("Exploratory Data Analysis")

    if df is not None:
        col = st.selectbox("Select column", df.columns)

        fig = px.histogram(df, x=col)
        st.plotly_chart(fig, use_container_width=True)

        if df.select_dtypes(include=np.number).shape[1] > 1:
            corr = df.corr(numeric_only=True)
            fig2 = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.warning("Upload data first")


# ---------------- Preprocessing ----------------

elif menu == "Preprocessing":

    st.header("Data Preprocessing")

    if df is not None:

        if st.button("Clean Data"):
            df = clean_data(df)
            st.success("Missing values handled & duplicates removed")

        if st.checkbox("Remove Outliers"):
            df = remove_outliers(df)
            st.success("Outliers removed")

        st.dataframe(df.head())

        st.download_button(
            "Download Processed Data",
            df.to_csv(index=False),
            file_name="processed_data.csv"
        )

    else:
        st.warning("Upload data first")


# ---------------- Model Training ----------------

elif menu == "Model Training":

    st.header("Train & Compare Models")

    if df is not None:

        target = st.selectbox("Select Target Column", df.columns)

        if target:

            X = df.drop(target, axis=1)
            y = df[target]

           
            X = drop_text_columns(X)
            X = encode_data(X)
            X = handle_missing(X)  
            X = scale_features(X)

            st.write("Remaining Missing Values:", X.isnull().sum().sum())

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if st.button("Train Models"):

                results = train_models(X_train, X_test, y_train, y_test)

                st.subheader("Model Performance")

                for model, score in results.items():
                    st.write(f"{model}: {round(score, 3)}")

                best_model = max(results, key=results.get)
                st.success(f"Best Model: {best_model}")

                fig = px.bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    title="Model Accuracy Comparison"
                )
                st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Upload data first")