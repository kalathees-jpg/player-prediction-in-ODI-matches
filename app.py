# ==========================================
# AI Cricket Performance Predictor (Advanced)
# With Rolling Average Analytics
# ==========================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cricket AI Predictor", layout="wide")

st.title("🏏 AI-Based Cricket Performance Analytics")
st.markdown("### Rolling Average & Future Prediction System")

# Load model
model = joblib.load("player_performance_model.pkl")

# Load dataset
df = pd.read_csv("final_cricket_dataset.csv")

# Aggregate match-level stats
player_history = df.groupby(['match_id', 'batter']).agg({
    'runs_batter': 'sum',
    'over': 'count'
}).reset_index()

player_history.columns = ['match_id', 'player', 'runs_scored', 'balls_faced']

# Feature Engineering
player_history['strike_rate'] = (
    player_history['runs_scored'] / player_history['balls_faced']
) * 100

player_history['avg_runs_per_ball'] = (
    player_history['runs_scored'] / player_history['balls_faced']
)

player_history.fillna(0, inplace=True)

# Player Selection
players = sorted(player_history['player'].unique())
selected_player = st.selectbox("👤 Select Player", players)

player_data = player_history[player_history['player'] == selected_player].sort_values("match_id")

# -----------------------------------
# 📊 Historical + Rolling Averages
# -----------------------------------

st.subheader("📈 Performance Trend Analysis")

player_data['rolling_3'] = player_data['runs_scored'].rolling(window=3).mean()
player_data['rolling_5'] = player_data['runs_scored'].rolling(window=5).mean()

fig1, ax1 = plt.subplots()
ax1.plot(player_data['match_id'], player_data['runs_scored'], marker='o', label="Actual Runs")
ax1.plot(player_data['match_id'], player_data['rolling_3'], linestyle='--', label="3-Match Avg")
ax1.plot(player_data['match_id'], player_data['rolling_5'], linestyle=':', label="5-Match Avg")

ax1.set_xlabel("Match ID")
ax1.set_ylabel("Runs")
ax1.set_title(f"{selected_player} Performance Trend")
ax1.legend()

st.pyplot(fig1)

# -----------------------------------
# 🤖 Future Prediction
# -----------------------------------

st.subheader("🤖 AI Future Match Prediction")

if len(player_data) >= 5:

    recent_data = player_data.tail(5)

    avg_balls = recent_data['balls_faced'].mean()
    avg_sr = recent_data['strike_rate'].mean()
    avg_rpb = recent_data['avg_runs_per_ball'].mean()

    input_data = np.array([[avg_balls, avg_sr, avg_rpb]])
    prediction = model.predict(input_data)[0]

    st.success(f"🎯 Predicted Runs in Next Match: {round(prediction, 2)}")

    next_match_id = player_data['match_id'].max() + 1

    fig2, ax2 = plt.subplots()
    ax2.plot(player_data['match_id'], player_data['runs_scored'], marker='o', label="Past Runs")
    ax2.scatter(next_match_id, prediction, marker='X', s=200, label="Predicted Next Match")

    ax2.set_xlabel("Match ID")
    ax2.set_ylabel("Runs")
    ax2.set_title("Future Match Prediction")
    ax2.legend()

    st.pyplot(fig2)

    if prediction > 50:
        st.markdown("### 🔥 Excellent Form")
    elif prediction > 30:
        st.markdown("### 👍 Good Form")
    else:
        st.markdown("### ⚠️ Poor Form")

else:
    st.warning("Minimum 5 matches required for advanced prediction.")