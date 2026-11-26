
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

:root {
    --navy: #142B49;
    --teal: #159D9A;
    --blue: #4776D0;
    --lavender: #8977D8;
    --text: #233B56;
}

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* MAIN BACKGROUND */
.stApp {
    background:
        radial-gradient(circle at 10% 5%, #D9F2F5 0%, transparent 30%),
        radial-gradient(circle at 90% 10%, #E6E1FF 0%, transparent 32%),
        linear-gradient(135deg, #EDF5FC, #F4F0FF, #E6F6F4);
    color: var(--text);
}

.block-container {
    max-width: 1450px;
    padding: 2rem 2rem 3rem;
}

/* HEADINGS */
h1, h2, h3, h4 {
    color: #193B60 !important;
    font-weight: 700 !important;
}

/* MAIN HEADER */
.main-header {
    background: linear-gradient(120deg, #153B61, #167F91, #6659B9);
    border: 2px solid rgba(255,255,255,.55);
    padding: 2.5rem 1.5rem;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 12px 35px rgba(43,85,130,.22);
}

.main-header h1 {
    color: #FFFFFF !important;
    font-size: clamp(1.8rem, 4vw, 2.8rem);
    margin: 0;
    font-weight: 800 !important;
}

.main-header p {
    color: #DDF8FA !important;
    font-size: 1rem;
    margin-top: .7rem;
}

/* METRIC CARDS */
.metric-card {
    background: linear-gradient(145deg, #FFFFFF, #E8F5FC);
    border: 2px solid #B9DCEB;
    border-top: 5px solid #159D9A;
    border-radius: 19px;
    padding: 1.3rem;
    min-height: 150px;
    text-align: center;
    box-shadow: 0 7px 20px rgba(39,80,125,.10);
    transition: .3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: #6CBEC5;
    box-shadow: 0 12px 25px rgba(39,80,125,.18);
}

.metric-card h3 {
    color: #167F91 !important;
    font-size: 2.2rem;
    margin: .3rem 0;
}

.metric-card p {
    color: #526B83 !important;
    font-size: .9rem;
}

/* FORMS */
[data-testid="stForm"] {
    background: linear-gradient(145deg, #FFFFFF, #F0F7FD);
    border: 2px solid #C4DDEC;
    border-radius: 20px;
    padding: 1.5rem;
    box-shadow: 0 8px 24px rgba(42,80,120,.09);
}

/* INPUT LABELS */
[data-testid="stWidgetLabel"] p,
[data-testid="stCheckbox"] label,
[data-testid="stSlider"] label {
    color: #284B6A !important;
    font-weight: 600 !important;
}

/* INPUT FIELDS */
div[data-baseweb="select"] > div,
div[data-testid="stTextInput"] input {
    background: #FFFFFF !important;
    border: 1.5px solid #B9D5E7 !important;
    border-radius: 11px !important;
    color: #193B60 !important;
}

div[data-baseweb="select"] > div:hover {
    border-color: #159D9A !important;
}

/* BUTTONS */
.stButton > button,
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(110deg, #159D9A, #4776D0);
    color: white !important;
    border: none;
    border-radius: 13px;
    padding: .8rem 1.5rem;
    font-size: 1.05rem;
    font-weight: 700;
    width: 100%;
    min-height: 48px;
    box-shadow: 0 5px 15px rgba(40,120,160,.20);
    transition: .3s ease;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(110deg, #117C80, #6258B8);
    transform: translateY(-3px);
    box-shadow: 0 9px 22px rgba(40,100,160,.30);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg, #102D4A 0%, #164C66 55%, #225E78 100%
    ) !important;
    border-right: 3px solid #55C9C0;
    min-width: 280px;
}

[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
    font-size: 1.75rem !important;
    font-weight: 800 !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #91E2D9 !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #E2F1FA !important;
    font-size: 1rem !important;
}

/* SIDEBAR NAVIGATION */
[data-testid="stSidebar"] [role="radiogroup"] {
    gap: 12px;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,.08);
    border: 1.5px solid rgba(180,225,240,.25);
    border-radius: 13px;
    padding: 13px 12px;
    min-height: 48px;
    transition: .25s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: rgba(85,201,192,.20);
    border-color: #7DDED5;
    transform: translateX(3px);
}

[data-testid="stSidebar"] [role="radiogroup"] label p {
    color: #FFFFFF !important;
    font-size: 1.02rem !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(110deg, #159D9A, #4776D0);
    border: 2px solid #A5F0E5;
    box-shadow: 0 5px 15px rgba(0,0,0,.20);
}

/* SIDEBAR ABOUT BOX */
.sidebar-about {
    background: rgba(255,255,255,.10);
    border: 1.5px solid rgba(180,230,240,.30);
    border-radius: 15px;
    padding: 15px;
    margin-top: 15px;
    color: #E5F4FA;
    font-size: .95rem;
    line-height: 1.8;
}

/* TIP CARDS */
.tip-card {
    background: linear-gradient(110deg, #FFFFFF, #E4F6F4);
    border: 1.5px solid #A8DCD5;
    border-left: 5px solid #159D9A;
    padding: 1rem 1.3rem;
    border-radius: 12px;
    margin: .6rem 0;
    color: #28556A;
    box-shadow: 0 4px 12px rgba(30,100,110,.07);
}

/* RISK RESULTS */
.risk-low {
    background: linear-gradient(120deg,#D5F5E5,#A6E8C6);
    color: #12613C;
    border: 2px solid #63C99A;
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 800;
}

.risk-medium {
    background: linear-gradient(120deg,#FFF2C9,#FFE09A);
    color: #805500;
    border: 2px solid #EAC05B;
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 800;
}

.risk-high {
    background: linear-gradient(120deg,#FFE0E4,#FFBBC6);
    color: #9B263E;
    border: 2px solid #E98B9A;
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.5rem;
    font-weight: 800;
}

/* STEP BADGE */
.step-badge {
    background: linear-gradient(135deg,#159D9A,#6659B9);
    color: white;
    border-radius: 50%;
    width: 32px;
    height: 32px;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    margin-right: 8px;
}

/* METRICS AND TABLES */
[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1.5px solid #C5DDEB;
    border-radius: 14px;
    padding: 1rem;
}

[data-testid="stMetricLabel"] {
    color: #526B83 !important;
}

[data-testid="stMetricValue"] {
    color: #167F91 !important;
}

[data-testid="stDataFrame"] {
    border: 2px solid #C3DCEB;
    border-radius: 13px;
    overflow: hidden;
}

hr {
    border-color: #C5DCEB !important;
}

/* RESPONSIVE DESIGN */
@media(max-width: 900px) {
    .block-container {
        padding: 1.2rem 1rem;
    }

    .metric-card {
        min-height: 125px;
        padding: 1rem .6rem;
    }

    .metric-card h3 {
        font-size: 1.7rem;
    }
}

@media(max-width: 600px) {
    .block-container {
        padding: .7rem .6rem;
    }

    .main-header {
        padding: 1.5rem .8rem;
        border-radius: 16px;
    }

    .main-header h1 {
        font-size: 1.7rem;
    }

    [data-testid="stSidebar"] {
        min-width: 0;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.15rem !important;
    }
}
</style>
""", unsafe_allow_html=True)


# ─── Generate Synthetic Training Data ──────────────────────────────────────────
@st.cache_data
def generate_lifestyle_data(n=2000):
    np.random.seed(42)
    data = {
        "sleep_hours": np.random.uniform(4, 10, n),
        "physical_activity": np.random.choice([0, 1, 2, 3], n),
        "diet_type": np.random.choice([0, 1, 2], n),
        "bmi": np.random.uniform(16, 40, n),
        "stress_level": np.random.randint(1, 11, n),
        "smoking": np.random.choice([0, 1], n),
        "alcohol": np.random.choice([0, 1, 2], n),
        "family_history": np.random.choice([0, 1], n),
        "age": np.random.randint(18, 75, n),
        "water_intake": np.random.uniform(0.5, 4, n),
    }
    df = pd.DataFrame(data)

    score = (
        (df["bmi"] > 25).astype(int) * 2 +
        (df["bmi"] > 30).astype(int) * 2 +
        (df["stress_level"] > 6).astype(int) * 2 +
        (df["sleep_hours"] < 6).astype(int) * 2 +
        df["smoking"] * 3 +
        (df["alcohol"] == 2).astype(int) * 2 +
        df["family_history"] * 2 +
        (df["physical_activity"] == 0).astype(int) * 2 +
        (df["diet_type"] == 0).astype(int) * 2 +
        (df["age"] > 50).astype(int) +
        (df["water_intake"] < 1.5).astype(int)
    )

    df["risk_label"] = pd.cut(
        score,
        bins=[-1, 5, 10, 100],
        labels=[0, 1, 2]
    ).astype(int)

    return df


@st.cache_data
def generate_symptom_data(n=2000):
    np.random.seed(99)

    diseases = [
        "Diabetes",
        "Hypertension",
        "Obesity",
        "Heart Disease",
        "Asthma"
    ]

    templates = {
        "Diabetes": [1,0,1,0,0,1,1,0,0,1],
        "Hypertension": [0,1,0,1,0,0,1,1,0,0],
        "Obesity": [0,0,1,0,1,0,1,0,1,1],
        "Heart Disease": [1,1,0,1,0,1,0,1,0,0],
        "Asthma": [0,0,0,0,1,0,0,0,0,0],
    }

    symptoms = [
        "fatigue",
        "headache",
        "frequent_urination",
        "chest_pain",
        "shortness_of_breath",
        "blurred_vision",
        "weight_gain",
        "dizziness",
        "joint_pain",
        "increased_thirst"
    ]

    rows, labels = [], []

    for _ in range(n):
        d = np.random.choice(diseases)
        base = np.array(templates[d])
        noise = np.random.randint(0, 2, len(base)) * 0.3
        row = np.clip(base + noise, 0, 1).round().astype(int)

        rows.append(row)
        labels.append(diseases.index(d))

    df = pd.DataFrame(rows, columns=symptoms)
    df["disease"] = labels

    return df, symptoms, diseases


# ─── Train Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_lifestyle_models():
    df = generate_lifestyle_data()

    features = [
        "sleep_hours", "physical_activity", "diet_type", "bmi",
        "stress_level", "smoking", "alcohol", "family_history",
        "age", "water_intake"
    ]

    X = df[features]
    y = df["risk_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
    }

    trained, accuracies = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        accuracies[name] = round(
            accuracy_score(y_test, model.predict(X_test)) * 100, 2
        )

    best = max(accuracies, key=accuracies.get)

    return trained, accuracies, best, features


@st.cache_resource
def train_disease_models():
    df, symptoms, diseases = generate_symptom_data()

    X = df[symptoms]
    y = df["disease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
    }

    trained, accuracies = {}, {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        accuracies[name] = round(
            accuracy_score(y_test, model.predict(X_test)) * 100, 2
        )

    best = max(accuracies, key=accuracies.get)

    return trained, accuracies, best, symptoms, diseases


# ─── Recommendations ──────────────────────────────────────────────────────────
RECOMMENDATIONS = {
    "Low": [
        "✅ Great job! Maintain your current healthy lifestyle.",
        "💧 Keep drinking 2–3 litres of water daily.",
        "🏃 Continue regular physical activity (30 min/day).",
        "😴 Maintain 7–8 hours of quality sleep.",
        "🥗 Keep eating a balanced diet with fruits & vegetables.",
    ],

    "Medium": [
        "⚠️ Your risk is moderate — lifestyle changes needed now!",
        "🏋️ Increase physical activity to at least 150 min/week.",
        "🍔 Reduce junk food, processed sugar & saturated fats.",
        "🧘 Practice stress management (yoga, meditation, deep breathing).",
        "💊 Get regular health check-ups every 6 months.",
        "🚭 If you smoke or drink, start reducing immediately.",
        "💧 Drink at least 2.5 litres of water daily.",
    ],

    "High": [
        "🚨 HIGH RISK DETECTED — Please consult a doctor immediately!",
        "🏥 Schedule a comprehensive health check-up this week.",
        "🍬 Strictly control sugar, salt, and fat intake.",
        "🚭 Stop smoking and alcohol consumption completely.",
        "💊 Follow prescribed medications if any.",
        "🏋️ Start light exercise under medical supervision.",
        "👨‍👩‍👧 Inform family members — hereditary risks may affect them too.",
        "📱 Use a health monitoring app to track vitals daily.",
    ],
}


DISEASE_TIPS = {
    "Diabetes": [
        "Monitor blood sugar regularly",
        "Reduce refined carbs & sugars",
        "Exercise daily for 30 min",
        "Maintain healthy BMI"
    ],

    "Hypertension": [
        "Reduce salt intake",
        "Avoid caffeine & alcohol",
        "Practice deep breathing exercises",
        "Monitor BP regularly"
    ],

    "Obesity": [
        "Create a calorie deficit diet",
        "Walk 10,000 steps daily",
        "Avoid late-night eating",
        "Stay hydrated"
    ],

    "Heart Disease": [
        "Take prescribed medication strictly",
        "Avoid physical overexertion",
        "Eat heart-healthy foods (oats, fish)",
        "Quit smoking immediately"
    ],

    "Asthma": [
        "Avoid allergens & pollutants",
        "Always carry inhaler",
        "Practice breathing exercises",
        "Avoid cold air exposure"
    ],
}


# ─── Load Models ──────────────────────────────────────────────────────────────
lf_models, lf_acc, lf_best, lf_features = train_lifestyle_models()

ds_models, ds_acc, ds_best, symptom_cols, disease_names = train_disease_models()


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("### 🩺 Navigation")

    page = st.radio(
        "",
        [
            "🏠 Home",
            "📊 Phase 1: Risk Prediction",
            "🔬 Phase 2: Disease Prediction",
            "📈 Model Performance"
        ]
    )

    st.markdown("---")

    st.markdown("### ℹ️ About")

    st.markdown("""
    <div class="sidebar-about">
        This system is for <b>educational purposes only</b>.<br>
        It does <b>NOT replace</b> professional medical advice.<br><br>
        <b>Team:</b> Vartika Singh<br>
        <b>United University | BCA 3rd Year</b>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Home":

    st.markdown("""
    <div class='main-header'>
        <h1>🩺 AI Health Predictor</h1>
        <p>AI-Powered Lifestyle-Based Disease Risk & Early Disease Prediction System</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    for col, emoji, val, label in zip(
        [c1, c2, c3, c4],
        ["🤖", "📊", "🏥", "✅"],
        ["4", "2000+", "5", "4"],
        [
            "ML Models",
            "Training Samples",
            "Diseases Covered",
            "Risk Levels"
        ]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <p style='font-size:2rem;margin:0'>{emoji}</p>
            <h3>{val}</h3>
            <p>{label}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("## 🔄 How It Works")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### <span class='step-badge'>1</span> Enter Lifestyle Data

        Fill in your daily habits — sleep, diet, BMI, stress, activity level etc.

        ### <span class='step-badge'>2</span> Get Risk Score

        Our ML models analyze your lifestyle and predict your disease risk as **Low / Medium / High**.

        ### <span class='step-badge'>3</span> Symptom Analysis *(if needed)*

        If your risk is Medium or High, enter your symptoms for specific disease prediction.

        ### <span class='step-badge'>4</span> Personalized Recommendations

        Get AI-powered health tips tailored to your risk level and predicted condition.
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        ### 🎯 Diseases Covered

        | Disease | Key Risk Factors |
        |---|---|
        | 🩸 Diabetes | Sugar, BMI, Fatigue |
        | ❤️ Heart Disease | Stress, Smoking, Chest Pain |
        | 📈 Hypertension | Salt, Alcohol, Headache |
        | ⚖️ Obesity | Diet, Activity, BMI |
        | 🌬️ Asthma | Allergens, Breathing Issues |

        ### 🔬 Algorithms Used

        - Random Forest ✅ *(Best Performer)*
        - Logistic Regression
        - Decision Tree
        - k-Nearest Neighbors
        """)

    st.info(
        "👈 **Start by clicking 'Phase 1: Risk Prediction' in the sidebar!**"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PHASE 1
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Phase 1: Risk Prediction":

    st.markdown("""
    <div class='main-header'>
        <h1>📊 Phase 1: Lifestyle Risk Prediction</h1>
        <p>Enter your lifestyle details to get your disease risk score</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("lifestyle_form"):

        st.markdown("### 👤 Personal Information")

        c1, c2 = st.columns(2)

        with c1:
            age = st.slider("🎂 Age", 18, 80, 25)
            bmi = st.slider(
                "⚖️ BMI (Body Mass Index)",
                15.0, 45.0, 22.0, 0.1
            )

        with c2:
            name = st.text_input("📝 Your Name (optional)", "")
            gender = st.selectbox(
                "🧑 Gender",
                ["Male", "Female", "Other"]
            )

        st.markdown("### 🌙 Daily Habits")

        c1, c2, c3 = st.columns(3)

        with c1:
            sleep = st.slider(
                "😴 Sleep (hours/day)",
                3.0, 12.0, 7.0, 0.5
            )

            stress = st.slider(
                "😰 Stress Level (1–10)",
                1, 10, 4
            )

        with c2:
            activity_lbl = st.selectbox(
                "🏃 Physical Activity",
                ["None", "Light", "Moderate", "Heavy"]
            )

            activity = [
                "None", "Light", "Moderate", "Heavy"
            ].index(activity_lbl)

            water = st.slider(
                "💧 Water Intake (litres/day)",
                0.5, 5.0, 2.0, 0.25
            )

        with c3:
            diet_lbl = st.selectbox(
                "🥗 Diet Type",
                ["Unhealthy", "Average", "Healthy"]
            )

            diet = [
                "Unhealthy", "Average", "Healthy"
            ].index(diet_lbl)

        st.markdown("### ⚠️ Risk Factors")

        c1, c2, c3 = st.columns(3)

        with c1:
            smoking = st.checkbox("🚬 Do you smoke?")

        with c2:
            alcohol_lbl = st.selectbox(
                "🍺 Alcohol Consumption",
                ["None", "Occasional", "Regular"]
            )

            alcohol = [
                "None", "Occasional", "Regular"
            ].index(alcohol_lbl)

        with c3:
            family = st.checkbox(
                "👨‍👩‍👧 Family history of chronic disease?"
            )

        model_choice = st.selectbox(
            "🤖 Choose ML Algorithm",
            list(lf_models.keys()),
            index=0
        )

        submitted = st.form_submit_button("🔍 Predict My Risk")

    if submitted:

        inp = np.array([[
            sleep,
            activity,
            diet,
            bmi,
            stress,
            int(smoking),
            alcohol,
            int(family),
            age,
            water
        ]])

        model = lf_models[model_choice]

        pred = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]

        labels = ["Low", "Medium", "High"]
        risk = labels[pred]

        st.markdown("---")

        greeting = f"**{name},** your" if name else "Your"

        st.markdown(f"### {greeting} Result")

        colors = {
            "Low": "risk-low",
            "Medium": "risk-medium",
            "High": "risk-high"
        }

        emojis = {
            "Low": "✅",
            "Medium": "⚠️",
            "High": "🚨"
        }

        st.markdown(
            f"<div class='{colors[risk]}'>{emojis[risk]} Disease Risk: {risk.upper()}</div>",
            unsafe_allow_html=True
        )

        st.markdown("#### 📊 Confidence Scores")

        cols = st.columns(3)

        for i, (lbl, prob) in enumerate(zip(labels, proba)):
            cols[i].metric(
                f"{lbl} Risk",
                f"{prob * 100:.1f}%"
            )

        # Bar chart
        fig, ax = plt.subplots(
            figsize=(6, 3),
            facecolor="#F0F7FC"
        )

        ax.set_facecolor("#F0F7FC")

        bar_colors = ["#159D9A", "#E7B84B", "#E87588"]

        bars = ax.bar(
            labels,
            proba * 100,
            color=bar_colors,
            width=0.5,
            edgecolor="none"
        )

        for bar, p in zip(bars, proba * 100):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{p:.1f}%",
                ha="center",
                color="#193B60",
                fontsize=11,
                fontweight="bold"
            )

        ax.set_ylabel("Confidence %", color="#193B60")
        ax.tick_params(colors="#193B60")
        ax.set_ylim(0, 110)

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)
        plt.close()

        st.markdown("#### 💡 Personalized Recommendations")

        for tip in RECOMMENDATIONS[risk]:
            st.markdown(
                f"<div class='tip-card'>{tip}</div>",
                unsafe_allow_html=True
            )

        if risk in ["Medium", "High"]:
            st.warning(
                "👉 **Go to Phase 2: Disease Prediction** in the sidebar for detailed symptom analysis!"
            )

        st.session_state["risk"] = risk


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PHASE 2
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Phase 2: Disease Prediction":

    st.markdown("""
    <div class='main-header'>
        <h1>🔬 Phase 2: Disease Prediction</h1>
        <p>Select your current symptoms for specific disease prediction</p>
    </div>
    """, unsafe_allow_html=True)

    risk = st.session_state.get("risk", None)

    if risk == "Low":
        st.success(
            "✅ Your Phase 1 risk was **Low**. Disease prediction is optional."
        )

    elif risk is None:
        st.info(
            "ℹ️ Complete Phase 1 first for best results. You can still use this page independently."
        )

    sym_labels = {
        "fatigue": "😴 Fatigue / Constant tiredness",
        "headache": "🤕 Frequent Headaches",
        "frequent_urination": "🚽 Frequent Urination",
        "chest_pain": "💔 Chest Pain / Tightness",
        "shortness_of_breath": "😮‍💨 Shortness of Breath",
        "blurred_vision": "👁️ Blurred Vision",
        "weight_gain": "⚖️ Unexplained Weight Gain",
        "dizziness": "💫 Dizziness / Lightheadedness",
        "joint_pain": "🦴 Joint Pain / Stiffness",
        "increased_thirst": "💧 Increased Thirst",
    }

    with st.form("symptom_form"):

        st.markdown("### 🤒 Select Symptoms You Are Experiencing")

        selected = {}

        c1, c2 = st.columns(2)

        for i, (key, label) in enumerate(sym_labels.items()):

            col = c1 if i % 2 == 0 else c2
            selected[key] = int(col.checkbox(label))

        model_choice = st.selectbox(
            "🤖 Algorithm",
            list(ds_models.keys()),
            index=0
        )

        submitted2 = st.form_submit_button("🔬 Predict Disease")

    if submitted2:

        inp = np.array([[
            selected[s] for s in symptom_cols
        ]])

        model = ds_models[model_choice]

        pred = model.predict(inp)[0]
        proba = model.predict_proba(inp)[0]

        disease = disease_names[pred]

        st.markdown("---")

        st.markdown(
            f"### 🏥 Predicted Condition: **{disease}**"
        )

        st.markdown(
            f"**Confidence:** `{proba[pred] * 100:.1f}%`"
        )

        # Probability chart
        fig, ax = plt.subplots(
            figsize=(8, 3),
            facecolor="#F0F7FC"
        )

        ax.set_facecolor("#F0F7FC")

        y_pos = range(len(disease_names))

        colors = [
            "#159D9A" if i == pred else "#91B8D8"
            for i in range(len(disease_names))
        ]

        ax.barh(
            list(y_pos),
            proba * 100,
            color=colors,
            edgecolor="none"
        )

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(
            disease_names,
            color="#193B60"
        )

        ax.set_xlabel("Confidence %", color="#193B60")
        ax.tick_params(axis="x", colors="#193B60")
        ax.set_xlim(0, 115)

        for i, p in enumerate(proba * 100):
            ax.text(
                p + 1,
                i,
                f"{p:.1f}%",
                va="center",
                color="#193B60",
                fontsize=9
            )

        for spine in ax.spines.values():
            spine.set_visible(False)

        st.pyplot(fig)
        plt.close()

        st.markdown("#### 💊 Disease-Specific Recommendations")

        for tip in DISEASE_TIPS.get(disease, []):
            st.markdown(
                f"<div class='tip-card'>💡 {tip}</div>",
                unsafe_allow_html=True
            )

        st.error(
            "⚕️ **DISCLAIMER:** This is an AI prediction for awareness only. "
            "Please consult a qualified doctor for proper diagnosis and treatment."
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Model Performance":

    st.markdown("""
    <div class='main-header'>
        <h1>📈 Model Performance Comparison</h1>
        <p>Accuracy comparison of all trained ML algorithms</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏥 Phase 1 — Lifestyle Risk Models")

    df_lf = pd.DataFrame({
        "Algorithm": list(lf_acc.keys()),
        "Accuracy (%)": list(lf_acc.values())
    }).sort_values(
        "Accuracy (%)",
        ascending=False
    )

    df_lf["Best?"] = df_lf["Algorithm"].apply(
        lambda x: "⭐ Best" if x == lf_best else ""
    )

    st.dataframe(
        df_lf.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(
        figsize=(8, 4),
        facecolor="#F0F7FC"
    )

    ax.set_facecolor("#F0F7FC")

    cols = [
        "#159D9A" if a == lf_best else "#91B8D8"
        for a in df_lf["Algorithm"]
    ]

    ax.bar(
        df_lf["Algorithm"],
        df_lf["Accuracy (%)"],
        color=cols,
        edgecolor="none"
    )

    ax.set_ylim(50, 105)
    ax.set_ylabel("Accuracy %", color="#193B60")
    ax.tick_params(colors="#193B60")

    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, v in enumerate(df_lf["Accuracy (%)"]):
        ax.text(
            i,
            v + 0.5,
            f"{v}%",
            ha="center",
            color="#193B60",
            fontweight="bold"
        )

    ax.set_title(
        "Phase 1 — Risk Prediction Accuracy",
        color="#193B60",
        fontsize=13
    )

    st.pyplot(fig)
    plt.close()

    st.markdown("### 🔬 Phase 2 — Disease Prediction Models")

    df_ds = pd.DataFrame({
        "Algorithm": list(ds_acc.keys()),
        "Accuracy (%)": list(ds_acc.values())
    }).sort_values(
        "Accuracy (%)",
        ascending=False
    )

    df_ds["Best?"] = df_ds["Algorithm"].apply(
        lambda x: "⭐ Best" if x == ds_best else ""
    )

    st.dataframe(
        df_ds.reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

    fig2, ax2 = plt.subplots(
        figsize=(7, 4),
        facecolor="#F0F7FC"
    )

    ax2.set_facecolor("#F0F7FC")

    cols2 = [
        "#159D9A" if a == ds_best else "#91B8D8"
        for a in df_ds["Algorithm"]
    ]

    ax2.bar(
        df_ds["Algorithm"],
        df_ds["Accuracy (%)"],
        color=cols2,
        edgecolor="none"
    )

    ax2.set_ylim(50, 105)
    ax2.set_ylabel("Accuracy %", color="#193B60")
    ax2.tick_params(colors="#193B60")

    for spine in ax2.spines.values():
        spine.set_visible(False)

    for i, v in enumerate(df_ds["Accuracy (%)"]):
        ax2.text(
            i,
            v + 0.5,
            f"{v}%",
            ha="center",
            color="#193B60",
            fontweight="bold"
        )

    ax2.set_title(
        "Phase 2 — Disease Prediction Accuracy",
        color="#193B60",
        fontsize=13
    )

    st.pyplot(fig2)
    plt.close()

    st.markdown("### 📝 Interpretation")

    st.markdown(f"""
    - **Best Phase 1 Model:** `{lf_best}` with **{lf_acc[lf_best]}%** accuracy
    - **Best Phase 2 Model:** `{ds_best}` with **{ds_acc[ds_best]}%** accuracy
    - Random Forest generally performs best due to its ensemble nature
    - All models trained on synthetic data — real dataset will improve accuracy further
    """)
