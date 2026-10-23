"""
Student Performance Prediction
A beginner/intermediate machine-learning project using scikit-learn and Streamlit.

Run locally:
    python -m pip install -r requirements.txt
    streamlit run app.py
"""

from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "student_performance.csv"
TARGET = "final_score"

NUMERIC_FEATURES = [
    "study_hours",
    "attendance_percent",
    "previous_score",
    "assignments_completed",
]
CATEGORICAL_FEATURES = ["extracurricular", "internet_access"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(missing)}")
    return df


def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            min_samples_leaf=2,
        )),
    ])

    # Fixed seed makes the demonstration repeatable.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions) ** 0.5,
        "r2": r2_score(y_test, predictions),
        "test_actual": y_test.reset_index(drop=True),
        "test_predicted": pd.Series(predictions),
    }
    # Fit final model on all rows for the interactive prediction form.
    model.fit(X, y)
    return model, metrics


def score_label(score):
    if score >= 85:
        return "Excellent"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Needs improvement"
    return "At risk — consider extra academic support"


st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Performance Prediction")
st.write(
    "Explore a demonstration machine-learning model that estimates a student's "
    "final score from study habits and learning indicators."
)
st.info(
    "Educational demo only. The included dataset is synthetic, not real student "
    "records. Predictions are estimates and must not be used to grade, label, or "
    "make high-stakes decisions about students."
)

try:
    data = load_data()
    model, metrics = train_model(data)
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

with st.sidebar:
    st.header("About this project")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Rows in sample dataset:**", len(data))
    st.write("**Target:** Final score (0–100)")
    st.write("**Features:** Study hours, attendance, previous score, assignments, extracurricular activities, internet access")

tab_predict, tab_data, tab_metrics = st.tabs(
    ["Make a prediction", "Explore dataset", "Model evaluation"]
)

with tab_predict:
    st.subheader("Enter student details")
    left, right = st.columns(2)
    with left:
        study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 80, 1)
        previous_score = st.slider("Previous exam score", 0, 100, 65, 1)
    with right:
        assignments = st.slider("Assignments completed (%)", 0, 100, 75, 5)
        extracurricular = st.selectbox("Participates in extracurricular activities?", ["No", "Yes"])
        internet_access = st.selectbox("Has internet access for study?", ["Yes", "No"])

    if st.button("Predict final score", type="primary"):
        input_df = pd.DataFrame([{
            "study_hours": study_hours,
            "attendance_percent": attendance,
            "previous_score": previous_score,
            "assignments_completed": assignments,
            "extracurricular": extracurricular,
            "internet_access": internet_access,
        }])
        prediction = float(model.predict(input_df)[0])
        prediction = max(0.0, min(100.0, prediction))
        st.metric("Estimated final score", f"{prediction:.1f} / 100")
        st.write(f"**Interpretation:** {score_label(prediction)}")
        st.caption(
            "This is a model estimate based on synthetic training data, not a "
            "guarantee of actual performance."
        )

with tab_data:
    st.subheader("Sample dataset")
    st.dataframe(data, use_container_width=True, hide_index=True)
    st.subheader("Quick statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average final score", f"{data[TARGET].mean():.1f}")
    c2.metric("Average attendance", f"{data['attendance_percent'].mean():.1f}%")
    c3.metric("Average study hours", f"{data['study_hours'].mean():.1f}")
    st.bar_chart(data.groupby("internet_access")[TARGET].mean())

with tab_metrics:
    st.subheader("Held-out test set performance")
    st.write(
        "Metrics are calculated on a test split that was not used for the first "
        "training pass. The model is then retrained on all sample rows for the "
        "interactive prediction form."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.2f} points")
    c2.metric("RMSE", f"{metrics['rmse']:.2f} points")
    c3.metric("R²", f"{metrics['r2']:.3f}")
    chart_data = pd.DataFrame({
        "Actual score": metrics["test_actual"],
        "Predicted score": metrics["test_predicted"],
    })
    st.write("Actual vs predicted scores (test rows)")
    st.dataframe(chart_data.round(1), use_container_width=True, hide_index=True)
    st.caption(
        "A lower MAE/RMSE is better. R² is dataset-dependent; with a small "
        "synthetic dataset, these metrics are only for demonstrating the workflow."
    )
