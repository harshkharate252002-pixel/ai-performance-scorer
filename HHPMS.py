import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Performance Scorer", layout="wide")

# --- MODEL TRAINING (Cached for performance) ---
@st.cache_data
def train_model():
    # Training Data
    X_train = np.array([
        [90, 8.5, 20], [70, 6.0, 5], [95, 9.0, 30], [60, 5.5, 2],
        [85, 7.5, 15], [75, 7.0, 10], [88, 8.0, 22], [55, 5.0, 1],
        [92, 8.8, 25], [78, 6.5, 12]
    ])
    y_train = np.array([4.5, 3.0, 5.0, 2.5, 4.0, 3.5, 4.2, 2.0, 4.8, 3.6])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Metrics
    y_pred = model.predict(X_train)
    r2 = r2_score(y_train, y_pred)
    return model, r2

model, r2_val = train_model()

# --- SIDEBAR: INPUTS ---
st.sidebar.header("Employee Input Data")
emp_name = st.sidebar.text_input("Employee Name", "Priya Sharma")
goal_pct = st.sidebar.slider("Goal Completion %", 0, 100, 85)
peer_score = st.sidebar.slider("Peer Feedback (1-10)", 1.0, 10.0, 7.5)
training_hrs = st.sidebar.number_input("Training Hours", 0, 100, 15)

# --- MAIN INTERFACE ---
st.title("📊 APS v1.0: AI Performance Scoring Model")
st.markdown("### MBA Project: AI in HR Performance Management")
st.divider()

# Create two columns for Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Prediction Results")
    
    # Prediction Logic
    raw_prediction = model.predict([[goal_pct, peer_score, training_hrs]])[0]
    final_rating = max(1.0, min(5.0, round(raw_prediction, 2)))
    
    # Display Score
    st.metric(label=f"Predicted Rating for {emp_name}", value=f"{final_rating} / 5.0")
    
    # Progress Bar visual
    st.progress(final_rating / 5.0)

    # Driver Analysis
    contributions = {
        'Goal Completion': model.coef_[0] * goal_pct,
        'Peer Feedback': model.coef_[1] * peer_score,
        'Training Hours': model.coef_[2] * training_hrs
    }
    top_driver = max(contributions, key=contributions.get)
    st.info(f"**Key Driver:** {top_driver} is contributing the most to this score.")

with col2:
    st.subheader("Model Diagnostics")
    diag_df = pd.DataFrame({
        "Metric": ["R-Squared", "Base Intercept", "Goal Weight", "Peer Weight", "Training Weight"],
        "Value": [
            f"{r2_val:.4f}", 
            f"{model.intercept_:.4f}", 
            f"{model.coef_[0]:.4f}", 
            f"{model.coef_[1]:.4f}", 
            f"{model.coef_[2]:.4f}"
        ]
    })
    st.table(diag_df)

# --- VISUALIZATION SECTION ---
st.divider()
st.subheader("Feature Impact Analysis")
impact_data = pd.DataFrame({
    'Factors': ['Goal Completion', 'Peer Feedback', 'Training Hours'],
    'Weight (Coefficient)': model.coef_
})
st.bar_chart(impact_data.set_index('Factors'))
