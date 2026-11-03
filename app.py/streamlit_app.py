
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# PAGE CONFIGURATION
st.set_page_config(
    page_title="PowerLift AI",
    page_icon="🏋️",
    layout="wide"
)

# LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    saved_data = joblib.load("models/powerlift_model.pkl")
    return saved_data["model"], saved_data["features"]

model, features = load_model()

# CUSTOM DESIGN
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
h1 {
    color: #ffffff;
}
div[data-testid="stMetric"] {
    background-color: #202735;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.title("🏋️ PowerLift AI")
    st.write("Athlete Performance Prediction")
    st.divider()
    st.subheader("About the Project")
    st.write(
        "PowerLift AI uses machine learning to estimate "
        "an athlete's next recorded competition total "
        "from previous performance information."
    )
    st.divider()
    st.caption("Model: Random Forest")
    st.caption("Dataset: OpenPowerlifting")
    st.caption("Training data ends: January 2018")

# ACTUAL VS PREDICTED PERFORMANCE
st.header("Actual vs Predicted Performance")

try:
    results = pd.read_csv("outputs/sample_predictions.csv")

    st.write(
        "This chart compares actual competition totals "
        "with the totals predicted by the trained model."
    )

    # Use a sample to keep the chart easy to display
    chart_data = results.sample(
        n=min(1500, len(results)),
        random_state=42
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        chart_data["ActualNextTotalKg"],
        chart_data["PredictedNextTotalKg"],
        alpha=0.4
    )

    # Ideal prediction reference line
    lower = min(
        chart_data["ActualNextTotalKg"].min(),
        chart_data["PredictedNextTotalKg"].min()
    )

    upper = max(
        chart_data["ActualNextTotalKg"].max(),
        chart_data["PredictedNextTotalKg"].max()
    )

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--"
    )

    ax.set_xlabel("Actual Total (kg)")
    ax.set_ylabel("Predicted Total (kg)")
    ax.set_title("Actual vs Predicted Competition Totals")
    ax.grid(alpha=0.2)

    st.pyplot(fig)
    plt.close(fig)

except Exception as e:
    st.warning(f"Could not load prediction results: {e}")

# MAIN HEADING
st.title("🏋️ PowerLift AI")

# MODEL PERFORMANCE SECTION
st.header("Model Performance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Mean Absolute Error", "47.88 kg")

with metric2:
    st.metric("Root Mean Squared Error", "88.59 kg")

with metric3:
    st.metric("R² Score", "0.7422")

st.caption(
    "Metrics are from the historical held-out test dataset."
)

st.divider()

st.subheader("Athlete Performance Prediction System")

st.write(
    "Enter previous competition information to estimate "
    "the athlete's total lift at their next recorded competition."
)

st.divider()

# INPUT SECTION
st.header("Athlete Information")

col1, col2 = st.columns(2)

with col1:
    previous_total = st.number_input(
        "Previous Total Lift (kg)",
        min_value=1.0,
        value=500.0
    )

    previous_squat = st.number_input(
        "Previous Squat (kg)",
        min_value=0.0,
        value=180.0
    )

    previous_bench = st.number_input(
        "Previous Bench Press (kg)",
        min_value=0.0,
        value=120.0
    )

    previous_deadlift = st.number_input(
        "Previous Deadlift (kg)",
        min_value=0.0,
        value=200.0
    )

with col2:
    bodyweight = st.number_input(
        "Bodyweight (kg)",
        min_value=1.0,
        value=75.0
    )

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=100,
        value=25
    )

    days_since_previous = st.number_input(
        "Days Since Previous Competition",
        min_value=0,
        value=90
    )

st.divider()

# PREDICTION
if st.button("Generate Prediction", type="primary"):

    calculated_total = (
        previous_squat +
        previous_bench +
        previous_deadlift
    )

    if abs(calculated_total - previous_total) > 5:
        st.warning(
            "Please check your inputs. The three lifts should "
            "approximately equal the previous total."
        )

    else:
        input_data = pd.DataFrame([{
            "PreviousTotalKg": previous_total,
            "PreviousSquatKg": previous_squat,
            "PreviousBenchKg": previous_bench,
            "PreviousDeadliftKg": previous_deadlift,
            "BodyweightKg": bodyweight,
            "Age": age,
            "DaysSincePrevious": days_since_previous
        }])

        input_data = input_data[features]

        prediction = model.predict(input_data)[0]

        st.divider()
        st.header("Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Estimated Next Competition Total",
                f"{prediction:.2f} kg"
            )

        with result_col2:
            st.metric(
                "Previous Competition Total",
                f"{previous_total:.2f} kg"
            )

        st.success("Prediction generated successfully!")

        st.info(
            "This is an experimental estimate based on historical "
            "competition data. It is not a guaranteed future result."
        )

# FOOTER
st.divider()
st.caption(
    "PowerLift AI | Machine Learning and Data Science Project"
)
