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

if "scaler" not in st.session_state:
    st.session_state.scaler = None

if "encoders" not in st.session_state:
    st.session_state.encoders = {}

if "bool_cols" not in st.session_state:
    st.session_state.bool_cols = []


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

    # Handle boolean columns (TRUE/FALSE strings) → 1/0
    for col in df.columns:
        if df[col].dtype == "object":
            unique_vals = df[col].astype(str).str.strip().str.upper().unique()
            if set(unique_vals).issubset({"TRUE", "FALSE", "MISSING"}):
                df[col] = df[col].astype(str).str.strip().str.upper().map(
                    {"TRUE": 1, "FALSE": 0, "MISSING": 0}
                )

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
        X_scaled = scaler.fit_transform(X)
        st.session_state.scaler = scaler
        return pd.DataFrame(X_scaled, columns=X.columns)
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


def smart_convert_input(val):
    val_str = str(val).strip().upper()

    if val_str in ["TRUE", "YES"]:
        return 1
    elif val_str in ["FALSE", "NO"]:
        return 0
    else:
        try:
            return float(val)
        except:
            return 0


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

        # Detect boolean columns BEFORE preprocessing (from raw df)
        bool_cols = []
        for col in df.columns:
            if df[col].dtype == "object":
                unique_vals = df[col].astype(str).str.strip().str.upper().unique()
                if set(unique_vals).issubset({"TRUE", "FALSE", "NAN", "MISSING"}):
                    bool_cols.append(col)
        st.session_state.bool_cols = bool_cols

        df_processed = preprocess(df, target)

        X_all = df_processed.drop(columns=[target], errors="ignore")
        y = df_processed[target]

        # ---------- FEATURE SELECTION (MANUAL with smart defaults) ----------
        temp_df = df_processed.copy()

        if temp_df[target].dtype == "object":
            temp_df[target] = LabelEncoder().fit_transform(temp_df[target].astype(str))

        for col in temp_df.columns:
            try:
                temp_df[col] = pd.to_numeric(temp_df[col])
            except:
                temp_df[col] = LabelEncoder().fit_transform(temp_df[col].astype(str))

        corr = temp_df.corr()

        # Compute top 5 by correlation as default suggestion
        if target in corr.columns:
            top_5_default = list(corr[target].abs().sort_values(ascending=False).index[1:6])
        else:
            top_5_default = list(X_all.columns[:5])

        # Keep only valid columns that exist in X_all
        top_5_default = [f for f in top_5_default if f in X_all.columns]

        st.subheader("🎯 Feature Selection")

        # Show correlation table for reference
        if target in corr.columns:
            corr_series = corr[target].abs().drop(labels=[target], errors="ignore").sort_values(ascending=False)
            with st.expander("📊 Feature Correlation with Target (for reference)"):
                corr_df = corr_series.reset_index()
                corr_df.columns = ["Feature", "Correlation (abs)"]
                corr_df["Correlation (abs)"] = corr_df["Correlation (abs)"].round(4)
                st.dataframe(corr_df, use_container_width=True)

        # Manual multiselect — default = top 5 by correlation
        selected_features = st.multiselect(
            "Select Features for Training (default = top 5 by correlation)",
            options=list(X_all.columns),
            default=top_5_default,
            help="You can add/remove features manually. Default selection is top 5 most correlated with target."
        )

        if len(selected_features) == 0:
            st.warning("⚠️ Please select at least 1 feature.")
            st.stop()

        # SAVE FEATURES
        st.session_state.features = selected_features
        st.info(f"✅ Using {len(selected_features)} features: {selected_features}")

        X = X_all[selected_features]
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
        st.write(f"Predicting for: **{st.session_state.target}**")

        input_data = {}

        for col in st.session_state.features:
            if col in st.session_state.bool_cols:
                val = st.text_input(f"Enter {col}", placeholder="TRUE or FALSE")
            else:
                val = st.text_input(f"Enter {col}")
            input_data[col] = val

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([input_data])

                for col in input_df.columns:
                    input_df[col] = smart_convert_input(input_df[col].iloc[0])

                if st.session_state.scaler is not None:
                    try:
                        input_scaled = st.session_state.scaler.transform(input_df)
                        input_df = pd.DataFrame(input_scaled, columns=input_df.columns)
                    except Exception as scale_err:
                        st.warning(f"Scaling skip: {scale_err}")

                prediction = st.session_state.model.predict(input_df)[0]

                if isinstance(prediction, float):
                    st.success(f"Prediction: {prediction:,.2f}")
                else:
                    st.success(f"Prediction: {prediction}")

            except Exception as e:
                st.error(f"Error: {e}")