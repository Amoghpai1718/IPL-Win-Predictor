import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------------------------
# 1. APP CONFIG
# --------------------------------------------------------------------
st.set_page_config(page_title="IPL AI Assistant", layout="wide")

# Load model and encoders safely
@st.cache_resource
def load_model():
    model = pickle.load(open("ipl_winner_model.pkl", "rb"))
    team_encoder = pickle.load(open("team_encoder.pkl", "rb"))
    toss_encoder = pickle.load(open("toss_decision_encoder.pkl", "rb"))
    venue_encoder = pickle.load(open("venue_encoder.pkl", "rb"))
    return model, team_encoder, toss_encoder, venue_encoder

model, team_encoder, toss_encoder, venue_encoder = load_model()

# Load match dataset for analysis
@st.cache_data
def load_data():
    df = pd.read_csv("all_matches.csv")
    # Standardize columns
    df.columns = df.columns.str.lower().str.strip()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "season" not in df.columns:
        if "year" in df.columns:
            df["season"] = df["year"]
        elif "date" in df.columns:
            df["season"] = df["date"].dt.year
        else:
            df["season"] = np.nan
    return df

matches = load_data()

# --------------------------------------------------------------------
# 2. GOOGLE GEMINI CHATBOT CONFIGURATION
# --------------------------------------------------------------------
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    model_gemini = None

# --------------------------------------------------------------------
# 3. STREAMLIT TABS
# --------------------------------------------------------------------
st.title("🏏 IPL AI Assistant (Predictor + Chatbot)")

tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "🤖 Ask IPL Chatbot"])

# --------------------------------------------------------------------
# TAB 1: PREDICT MATCH WINNER
# --------------------------------------------------------------------
with tab1:
    st.header("IPL Match Winner Predictor")

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox("Batting Team", sorted(matches["team1"].unique()))
        bowling_team = st.selectbox("Bowling Team", sorted(matches["team2"].unique()))
    with col2:
        toss_decision = st.selectbox("Toss Decision", toss_encoder.classes_)
        venue = st.selectbox("Venue", sorted(venue_encoder.classes_))

    if st.button("Predict Winner"):
        if batting_team == bowling_team:
            st.error("Batting and Bowling team cannot be the same.")
        else:
            input_data = pd.DataFrame({
                "team1": [batting_team],
                "team2": [bowling_team],
                "toss_decision": [toss_decision],
                "venue": [venue]
            })

            # Encode
            input_data["team1"] = team_encoder.transform(input_data["team1"])
            input_data["team2"] = team_encoder.transform(input_data["team2"])
            input_data["toss_decision"] = toss_encoder.transform(input_data["toss_decision"])
            input_data["venue"] = venue_encoder.transform(input_data["venue"])

            pred = model.predict(input_data)[0]
            winner = team_encoder.inverse_transform([pred])[0]

            st.success(f"🏆 Predicted Winner: **{winner}**")

            # ----------------- Detailed Analysis -----------------
            st.subheader("📊 Match Analysis Insights")

            h2h = matches[
                ((matches["team1"] == batting_team) & (matches["team2"] == bowling_team)) |
                ((matches["team1"] == bowling_team) & (matches["team2"] == batting_team))
            ]

            if h2h.empty:
                st.write("No head-to-head data available between these teams.")
            else:
                total_matches = len(h2h)
                wins = h2h["winner"].value_counts()
                st.write(f"Total Matches Played: **{total_matches}**")
                st.bar_chart(wins)

                # Season trend with auto-detection fix
                if "season" in h2h.columns:
                    season_col = "season"
                elif "year" in h2h.columns:
                    season_col = "year"
                else:
                    h2h["season_year"] = h2h["date"].dt.year
                    season_col = "season_year"

                win_trend = h2h.groupby(season_col)["winner"].value_counts().unstack().fillna(0)
                st.line_chart(win_trend)

                st.caption("Analysis includes head-to-head performance and season-wise trends.")

# --------------------------------------------------------------------
# TAB 2: CHATBOT
# --------------------------------------------------------------------
with tab2:
    st.header("IPL Chatbot (AI Assistant)")

    if not model_gemini:
        st.warning("Google API key not found. Chatbot is disabled.")
    else:
        user_input = st.text_input("Ask me anything about IPL, players, or stats:")
        if st.button("Ask"):
            if user_input.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    response = model_gemini.generate_content(user_input)
                    st.write("**Answer:**", response.text)

# --------------------------------------------------------------------
# END
# --------------------------------------------------------------------


