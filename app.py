import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="RoadSense AI",
    page_icon="🚦",
    layout="wide"
)
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(37,99,235,0.18), transparent 30%),
        radial-gradient(circle at 90% 10%, rgba(124,58,237,0.14), transparent 28%),
        linear-gradient(135deg, #020617 0%, #0b1120 50%, #111827 100%);
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ALL MAIN TEXT */
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3,
[data-testid="stAppViewContainer"] p {
    color: #f8fafc;
}

/* FORM CARD */
[data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.18);
    border-radius: 22px;
    padding: 28px;
    box-shadow: 0 18px 45px rgba(0,0,0,0.28);
}

/* FORM SECTION TITLES */
[data-testid="stForm"] h3 {
    color: #f8fafc !important;
    font-size: 24px;
    font-weight: 750;
}

/* INPUT LABELS */
[data-testid="stWidgetLabel"] p {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
}

/* SELECT BOX */
[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
}

/* SELECT BOX TEXT */
[data-baseweb="select"] span {
    color: #0f172a !important;
}

/* NUMBER INPUT */
[data-testid="stNumberInput"] input {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

/* DIVIDERS */
hr {
    border-color: rgba(148,163,184,0.18) !important;
}

/* HERO */
.hero {
    padding: 38px;
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(30,64,175,0.75),
            rgba(76,29,149,0.65)
        );
    border: 1px solid rgba(147,197,253,0.20);
    box-shadow: 0 22px 55px rgba(0,0,0,0.35);
    margin-bottom: 28px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
}

.hero-subtitle {
    color: #e2e8f0;
    font-size: 17px;
    line-height: 1.7;
    margin-top: 14px;
    max-width: 1000px;
}

/* RESULT CARD */
.result-card {
    padding: 30px;
    border-radius: 22px;
    margin-top: 20px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}

.slight {
    background: linear-gradient(
        135deg,
        rgba(5,150,105,0.85),
        rgba(6,78,59,0.90)
    );
}

.serious {
    background: linear-gradient(
        135deg,
        rgba(217,119,6,0.88),
        rgba(120,53,15,0.92)
    );
}

.fatal {
    background: linear-gradient(
        135deg,
        rgba(220,38,38,0.88),
        rgba(127,29,29,0.94)
    );
}

.result-title {
    font-size: 30px;
    font-weight: 800;
    color: #ffffff;
}

.result-text {
    color: #f1f5f9;
    font-size: 15px;
    margin-top: 8px;
}

/* BUTTON */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    border-radius: 12px;
    height: 52px;
    font-weight: 700;
    font-size: 16px;
    background: linear-gradient(
        90deg,
        #2563eb,
        #7c3aed
    );
    color: white !important;
    border: none;
    transition: all 0.2s ease;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(59,130,246,0.35);
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background: rgba(15,23,42,0.85);
    border: 1px solid rgba(148,163,184,0.18);
    padding: 18px;
    border-radius: 16px;
}

[data-testid="stMetricLabel"] p {
    color: #cbd5e1 !important;
}

[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* EXPANDER */
[data-testid="stExpander"] {
    background: rgba(15,23,42,0.75);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px;
}

/* CAPTION */
[data-testid="stCaptionContainer"] {
    color: #94a3b8 !important;
}
[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.95) !important;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("accident_model.pkl")

model = load_model()

st.markdown("""
<div class="hero">
    <div class="hero-title">
        🚦 RoadSense AI
    </div>
    <div class="hero-subtitle">
        Intelligent traffic accident severity prediction
        powered by class-weighted XGBoost.
        Analyze driver, road, weather and collision factors
        to estimate accident severity.
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

day_map = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6
}

sex_map = {
    "Male": 0,
    "Female": 1
}

age_map = {
    "under 18": 0,
    "18-30": 1,
    "31-50": 2,
    "over 50": 3
}

Educational_map = {
    "Illiterate": 0,
    "Elementary school": 1,
    "Junior high school": 2,
    "High school": 3,
    "Above high school": 4
}

experience_map = {
    "Below 1yr": 0,
    "1-2yr": 1,
    "2-5yr": 2,
    "5-10yr": 3,
    "Above 10yr": 4
}

vehicle_map = {
    "Bicycle": 0,
    "Special vehicle": 1,
    "Motorcycle": 2,
    "Turbo": 3,
    "Automobile": 4,
    "Public (> 45 seats)": 5,
    "Lorry (41?100Q)": 6,
    "Public (13?45 seats)": 7,
    "Lorry (11?40Q)": 8,
    "Long lorry": 9,
    "Public (12 seats)": 10,
    "Taxi": 11,
    "Pick up upto 10Q": 12,
    "Stationwagen": 13,
    "Ridden horse": 14
}

Area_map = {
    "Residential areas": 0,
    "Office areas": 1,
    "Recreational areas": 10,
    "Industrial areas": 3,
    "Church areas": 4,
    "Market areas": 5,
    "Rural village areas": 6,
    "Outside rural areas": 7,
    "Hospital areas": 8,
    "School areas": 9,
    "Rural village areasOffice areas": 9
}

Junction_map = {
    "No junction": 0,
    "Y Shape": 1,
    "Crossing": 2,
    "O Shape": 3,
    "Unknown": 4,
    "T Shape": 5,
    "X Shape": 6
}

light_map = {
    "Daylight": 0,
    "Darkness - lights lit": 1,
    "Darkness - no lighting": 2,
    "Darkness - lights unlit:": 3
}

weather_map = {
    "Normal": 0,
    "Raining": 1,
    "Raining and Windy": 2,
    "Cloudy": 3,
    "Windy": 4,
    "Snow": 5,
    "Fog or mist": 6
}

Collision_map = {
    "Collision with roadside-parked vehicles": 0,
    "Vehicle with vehicle collision": 1,
    "Collision with roadside objects": 2,
    "Collision with animals": 3,
    "Rollover": 4,
    "Fall from vehicles": 5,
    "Collision with pedestrians": 6,
    "With Train": 7
}

cause_map = {
    "Moving Backward": 0,
    "Overtaking": 1,
    "Changing lane to the left": 2,
    "Changing lane to the right": 3,
    "Driving under the influence of drugs": 4,
    "No distancing": 5,
    "Driving carelessly": 6,
    "Driving at high speed": 7,
    "Overloading": 8,
    "Other": 9,
    "Improper parking": 10,
    "No priority to vehicle": 11,
    "No priority to pedestrian": 12,
    "Fatigue": 13,
    "Drunk driving": 14,
    "Bad Road": 15,
    "Bad weather": 16
}

severity_map = {
    0: "Slight Injury",
    1: "Serious Injury",
    2: "Fatal Injury"
}

with st.form("accident_form"):
    st.subheader("Driver Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        day = st.selectbox("Day Of The Week", list(day_map.keys()))
        sex = st.selectbox("Sex", list(sex_map.keys()))

    with col2:
        age = st.selectbox("Age", list(age_map.keys()))
        Educational = st.selectbox(
            "Educational level",
            list(Educational_map.keys())
        )

    with col3:
        experience = st.selectbox(
            "Driving experience",
            list(experience_map.keys())
        )
        vehicle = st.selectbox(
            "Vehicle type",
            list(vehicle_map.keys())
        )

    st.divider()
    st.subheader("Road and Environmental Conditions")

    col4, col5, col6 = st.columns(3)

    with col4:
        Area = st.selectbox(
            "Area type",
            list(Area_map.keys())
        )
        Junction = st.selectbox(
            "Junction type",
            list(Junction_map.keys())
        )

    with col5:
        light = st.selectbox(
            "Light conditions",
            list(light_map.keys())
        )
        weather = st.selectbox(
            "Weather conditions",
            list(weather_map.keys())
        )

    with col6:
        Collision = st.selectbox(
            "Type of collision",
            list(Collision_map.keys())
        )
        casulties = st.number_input(
            "Number of casualties",
            min_value=0,
            max_value=20,
            value=1,
            step=1
        )

    st.divider()

    cause = st.selectbox(
        "Cause of accident",
        list(cause_map.keys())
    )

    submitted = st.form_submit_button(
        "Predict Accident Severity",
        use_container_width=True
    )

feature_order = [
    "Day_of_week",
    "Sex_of_driver",
    "Age_band_of_driver",
    "Educational_level",
    "Driving_experience",
    "Type_of_vehicle",
    "Area_accident_occured",
    "Types_of_Junction",
    "Light_conditions",
    "Weather_conditions",
    "Type_of_collision",
    "Number_of_casualties",
    "Cause_of_accident"
]

if submitted:
    input_data = pd.DataFrame([{
        "Day_of_week": day_map[day],
        "Sex_of_driver": sex_map[sex],
        "Age_band_of_driver": age_map[age],
        "Educational_level": Educational_map[Educational],
        "Driving_experience": experience_map[experience],
        "Type_of_vehicle": vehicle_map[vehicle],
        "Area_accident_occured": Area_map[Area],
        "Types_of_Junction": Junction_map[Junction],
        "Light_conditions": light_map[light],
        "Weather_conditions": weather_map[weather],
        "Type_of_collision": Collision_map[Collision],
        "Number_of_casualties": casulties,
        "Cause_of_accident": cause_map[cause]
    }])

    input_data = input_data[feature_order]

    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]
    result = severity_map[prediction]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 0:
        st.markdown(f"""
        <div class="result-card slight">
            <div class="result-title">
                🟢 {result}
            </div>
            <div class="result-text">
                The model predicts a lower severity
                accident outcome.
            </div>
        </div>
        """, unsafe_allow_html=True)

    elif prediction == 1:
        st.markdown(f"""
        <div class="result-card serious">
            <div class="result-title">
                🟠 {result}
            </div>
            <div class="result-text">
                The model predicts a serious
                injury accident outcome.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div class="result-card fatal">
            <div class="result-title">
                🔴 {result}
            </div>
            <div class="result-text">
                The model predicts a high-risk
                fatal accident outcome.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Prediction Probabilities")

    probability_df = pd.DataFrame({
        "Severity": [
            "Slight Injury",
            "Serious Injury",
            "Fatal Injury"
        ],
        "Probability": probabilities
    })

    st.bar_chart(
        probability_df.set_index("Severity")
    )

    col7, col8, col9 = st.columns(3)

    with col7:
        st.metric(
            "Slight Injury",
            f"{probabilities[0] * 100:.1f}%"
        )

    with col8:
        st.metric(
            "Serious Injury",
            f"{probabilities[1] * 100:.1f}%"
        )

    with col9:
        st.metric(
            "Fatal Injury",
            f"{probabilities[2] * 100:.1f}%"
        )

    with st.expander("View Model Input"):
        st.dataframe(
            input_data,
            use_container_width=True
        )

st.divider()

st.caption(
    "Machine Learning project using class-weighted XGBoost "
    "for multiclass traffic accident severity prediction."
)