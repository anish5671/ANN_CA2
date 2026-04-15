import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

import plotly.express as px

st.set_page_config(page_title="ML Pipeline", layout="wide")
st.title("🚀 ML Auto Pipeline")

menu = st.sidebar.radio("Menu", ["Upload", "EDA", "Preprocess", "Model", "Predict"])


# ---------- SESSION ----------
if "model" not in st.session_state:
    st.session_state.model = None

if "features" not in st.session_state:
    st.session_state.features = None

if "target" not in st.session_state:
    st.session_state.target = None


# ---------- FUNCTIONS ----------

def preprocess(df, target_col=None):
    df = df.copy()
    df = df.drop_duplicates()

    # CLEAN ₹ and commas
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = df[col].str.replace("₹", "", regex=False)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna("Missing")

    # Convert numeric-like
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    # Handle categorical (DO NOT DROP TARGET)
    for col in df.select_dtypes(include=['object', 'string']):
        if col == target_col:
            continue

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


# ---------- UPLOAD ----------
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

        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

        num_df = df.select_dtypes(include=np.number)

        if num_df.shape[1] > 1:
            st.subheader("Correlation Matrix")
            corr = num_df.corr()
            fig2 = px.imshow(corr, text_auto=True)
            st.plotly_chart(fig2)
    else:
        st.warning("Upload file")


# ---------- PREPROCESS ----------
elif menu == "Preprocess":
    if df is not None:
        target = st.selectbox("Select Target (optional)", ["None"] + list(df.columns))

        if target == "None":
            target = None

        if st.button("Run Preprocessing"):
            df_clean = preprocess(df, target)
            st.success("Preprocessing Done")
            st.dataframe(df_clean.head())
            st.download_button("Download Clean Data", df_clean.to_csv(index=False), "clean.csv")
        else:
            st.dataframe(df.head())

    else:
        st.warning("Upload file")


# ---------- MODEL ----------
elif menu == "Model":
    if df is not None:

        # SMART TARGET SUGGESTION
        possible_targets = [col for col in df.columns if col.lower() in ["price", "salary", "amount", "target"]]
        default_target = possible_targets[0] if possible_targets else df.columns[-1]

        target = st.selectbox("Select Target Column", df.columns, index=df.columns.get_loc(default_target))

        # SAVE TARGET
        st.session_state.target = target

        df = preprocess(df, target)

        X = df.drop(columns=[target], errors="ignore")
        y = df[target]

        # ---------- FEATURE SELECTION ----------
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
        else:
            top_features = X.columns[:5]

        X = X[top_features]

        # SAVE FEATURES
        st.session_state.features = list(top_features)

        st.write("Selected Features:", list(top_features))

        X = scale(X)

        choice = st.selectbox("Problem Type", ["Auto", "Classification", "Regression"])
        problem = detect_problem(y) if choice == "Auto" else choice

        models = get_models(problem)

        if st.button("Train Models"):
            results = {}
            trained_models = {}

            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X, y, cv=5)
                    model.fit(X, y)

                    results[name] = round(scores.mean(), 3)
                    trained_models[name] = model
                except:
                    results[name] = "Error"

            st.write(results)

            fig = px.bar(x=list(results.keys()), y=list(results.values()))
            st.plotly_chart(fig)

            best_model_name = max(results, key=lambda k: results[k] if results[k] != "Error" else -1)
            best_model = trained_models.get(best_model_name)

            st.success(f"Best Model: {best_model_name}")

            st.session_state.model = best_model

    else:
        st.warning("Upload file")


# ---------- PREDICT ----------
elif menu == "Predict":
    st.subheader("🔮 Prediction Panel")

    if st.session_state.model is None:
        st.warning("Train model first")
    else:
        st.write(f"Predicting for: {st.session_state.target}")

        input_data = {}

        for col in st.session_state.features:
            val = st.text_input(f"Enter {col}")
            input_data[col] = val

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])

                for col in input_df.columns:
                    try:
                        input_df[col] = pd.to_numeric(input_df[col])
                    except:
                        input_df[col] = 0

                prediction = st.session_state.model.predict(input_df)[0]

                st.success(f"Prediction: {prediction}")

            except Exception as e:
                st.error(e)