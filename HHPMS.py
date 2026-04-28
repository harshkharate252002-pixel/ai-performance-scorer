import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import re

st.set_page_config(page_title="AI PMS v2.0", layout="wide")

# ============================================================
# SENTIMENT ENGINE (same as your code)
# ============================================================

POSITIVE_WORDS = ["excellent","outstanding","exceptional","proactive","innovative",
                  "leadership","collaborative","efficient","achieved","delivered",
                  "exceeded","strong","great","good","mentor","champion"]

NEGATIVE_WORDS = ["poor","weak","failed","missed","delayed","unresponsive",
                  "problem","struggled","underperformed","inconsistent","error"]

NEUTRAL_WORDS = ["sometimes","average","moderate","adequate","satisfactory",
                 "acceptable","normal"]

def analyze_sentiment(text):
    tokens = re.sub(r"[^a-z\s]", " ", text.lower()).split()
    pos = [w for w in tokens if w in POSITIVE_WORDS]
    neg = [w for w in tokens if w in NEGATIVE_WORDS]
    neu = [w for w in tokens if w in NEUTRAL_WORDS]

    total = len(pos) + len(neg) + len(neu) or 1
    score = (len(pos) - 1.3*len(neg)) / total

    if score > 0.3:
        tone, delta = "positive", min(0.35, score*0.5)
    elif score < -0.1:
        tone, delta = "negative", max(-0.35, score*0.5)
    else:
        tone, delta = "neutral", 0

    return tone, round(delta,3), len(pos), len(neg), len(neu)

# ============================================================
# MODEL TRAINING
# ============================================================

X_train = np.array([
    [90,8.5,20,97,8.8,8.5,8.0,7.5],
    [70,6.0,5,85,6.2,6.0,6.5,4.0],
    [95,9.0,30,99,9.2,9.0,8.8,9.0],
    [60,5.5,2,78,5.0,5.5,6.0,2.0]
])

y_train = np.array([4.5,3.0,5.0,2.5])

model = LinearRegression()
model.fit(X_train, y_train)

# ============================================================
# UI HEADER
# ============================================================

st.title("📊 AI Performance Management System v2.0")
st.markdown("MBA HR Project | Predict Employee Performance using AI + NLP")

# ============================================================
# INPUT SECTION
# ============================================================

st.sidebar.header("📥 Employee Inputs")

name = st.sidebar.text_input("Employee Name")

goal = st.sidebar.slider("Goal Completion (%)", 0, 100, 75)
peer = st.sidebar.slider("Peer Feedback (1-10)", 1.0, 10.0, 6.0)
training = st.sidebar.slider("Training Hours", 0, 60, 10)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 90)
manager = st.sidebar.slider("Manager Rating", 1.0, 10.0, 7.0)
project = st.sidebar.slider("Project Score", 1.0, 10.0, 7.0)
self_score = st.sidebar.slider("Self Score", 1.0, 10.0, 7.0)
mobility = st.sidebar.slider("Mobility Score", 0.0, 10.0, 5.0)

feedback = st.text_area("📝 Manager Feedback")

# ============================================================
# PREDICTION
# ============================================================

if st.button("🚀 Predict Performance"):

    inputs = np.array([[goal, peer, training, attendance,
                        manager, project, self_score, mobility]])

    base = model.predict(inputs)[0]
    base = round(max(1, min(5, base)), 2)

    tone, delta, pos, neg, neu = analyze_sentiment(feedback)

    final = round(max(1, min(5, base + delta)), 2)

    # Tier
    if final >= 4.5:
        tier = "🏆 Outstanding"
    elif final >= 3.8:
        tier = "🔥 Above Average"
    elif final >= 3:
        tier = "⚖️ Average"
    elif final >= 2:
        tier = "⚠️ Below Average"
    else:
        tier = "❌ Needs Improvement"

    # ========================================================
    # OUTPUT SECTION
    # ========================================================

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Base Score", base)
        st.metric("NLP Adjustment", delta)
        st.metric("Final Score", final)

    with col2:
        st.success(f"Performance Tier: {tier}")
        st.info(f"Sentiment: {tone.upper()}")

    # ========================================================
    # CHARTS
    # ========================================================

    st.subheader("📊 Sentiment Breakdown")

    df = pd.DataFrame({
        "Type": ["Positive", "Negative", "Neutral"],
        "Count": [pos, neg, neu]
    })

    fig, ax = plt.subplots()
    ax.bar(df["Type"], df["Count"])
    st.pyplot(fig)

    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.subheader("📌 Recommendations")

    recs = []

    if goal < 70:
        recs.append("Improve goal completion planning.")
    if peer < 6:
        recs.append("Work on teamwork & collaboration.")
    if training < 8:
        recs.append("Increase training participation.")
    if attendance < 85:
        recs.append("Attendance needs improvement.")
    if manager < 5.5:
        recs.append("Manager suggests performance plan.")
    if self_score - manager > 2:
        recs.append("⚠️ Self-rating may be inflated.")

    if final >= 4.5:
        recs.append("Eligible for leadership programs.")

    if not recs:
        recs.append("Maintain performance and take stretch roles.")

    for r in recs:
        st.write("✔️", r)
