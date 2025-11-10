import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import google.generativeai as genai
from sklearn.preprocessing import LabelEncoder
import joblib  # Use joblib as it's often more robust for sklearn models

# --------------------------------------------------------------------
# 1. APP CONFIG
# --------------------------------------------------------------------
st.set_page_config(page_title="IPL AI Assistant", layout="wide")

# Load model and encoders safely
@st.cache_resource
def load_model():
    # Load all 4 .pkl files created by the training script
    try:
        model = joblib.load("ipl_winner_model.pkl")
        team_encoder = joblib.load("team_encoder.pkl")
        
        # --- FIX 1: Corrected the filename ---
        # The training script saved this as 'toss_encoder.pkl'
        toss_encoder = joblib.load("toss_encoder.pkl") 
        
        venue_encoder = joblib.load("venue_encoder.pkl")
        
        return model, team_encoder, toss_encoder, venue_encoder
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.error("Please ensure all .pkl files (ipl_winner_model.pkl, team_encoder.pkl, toss_encoder.pkl, venue_encoder.pkl) are in the same folder.")
        return None, None, None, None

model, team_encoder, toss_encoder, venue_encoder = load_model()

# Load match dataset for analysis
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("all_matches.csv")
        # Standardize columns
        df.columns = df.columns.str.lower().str.strip()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
        # Create a 'season' column as a fallback
        if "season" not in df.columns:
            if "year" in df.columns:
                df["season"] = df["year"]
            elif "date" in df.columns:
                df["season"] = df["date"].dt.year
            else:
                df["season"] = 2023 # A reasonable default if no date/year
        return df
    except FileNotFoundError as e:
        st.error(f"Error: {e}")
        st.error("Please ensure 'all_matches.csv' is in the same folder.")
        return pd.DataFrame() # Return empty df to avoid crashes

matches = load_data()

# --------------------------------------------------------------------
# 2. GOOGLE GEMINI CHATBOT CONFIGURATION
# --------------------------------------------------------------------

# --- FIX 2: Use st.secrets for deployment ---
# This will read the "GEMINI_API_KEY" from Streamlit Cloud secrets
api_key = st.secrets.get("GEMINI_API_KEY") 

if api_key:
    genai.configure(api_key=api_key)
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
else:
    model_gemini = None
    # Show a warning in the sidebar if the key is missing
    st.sidebar.warning("GEMINI_API_KEY not found in st.secrets. Chatbot will be disabled.")

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

    if model is None or matches.empty:
        st.error("Application cannot start. Missing model or data files.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            # Get team lists from the encoder to ensure consistency
            all_teams = sorted(list(team_encoder.classes_))
            batting_team = st.selectbox("Batting Team", all_teams)
            bowling_team = st.selectbox("Bowling Team", all_teams, index=1 if len(all_teams) > 1 else 0)
        
        with col2:
            toss_decision = st.selectbox("Toss Decision", sorted(list(toss_encoder.classes_)))
            venue = st.selectbox("Venue", sorted(list(venue_encoder.classes_)))

        if st.button("Predict Winner"):
            if batting_team == bowling_team:
                st.error("Batting and Bowling team cannot be the same.")
            else:
                try:
                    # --- FIX 3: Build the full feature set for the model ---
                    
                    # 1. Encode the inputs from the UI
                    team1_enc = team_encoder.transform([batting_team])[0]
                    team2_enc = team_encoder.transform([bowling_team])[0]
                    toss_decision_enc = toss_encoder.transform([toss_decision])[0]
                    venue_enc = venue_encoder.transform([venue])[0]

                    # 2. Provide default values for features the model needs
                    #    but the UI doesn't ask for.
                    
                    # We'll assume the batting team won the toss for this prediction
                    toss_winner_enc = team1_enc 
                    
                    # We'll assume average form (0.5 = 50% win rate) for both teams
                    team1_form = 0.5
                    team2_form = 0.5

                    # 3. Create the input DataFrame with the EXACT column names
                    #    the model was trained on.
                    input_data = pd.DataFrame({
                        'team1_enc': [team1_enc],
                        'team2_enc': [team2_enc],
                        'venue_enc': [venue_enc],
                        'toss_winner_enc': [toss_winner_enc],
                        'toss_decision_enc': [toss_decision_enc],
                        'team1_form': [team1_form],
                        'team2_form': [team2_form]
                    })
                    
                    # 4. Predict
                    pred = model.predict(input_data)[0]
                    
                    # 5. Decode the prediction
                    # The model predicts 1 if team1 (batting_team) wins, 0 if team2 wins
                    winner = batting_team if pred == 1 else bowling_team

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

                        # Season trend
                        season_col = "season" # We fixed this in load_data()
                        win_trend = h2h.groupby(season_col)["winner"].value_counts().unstack().fillna(0)
                        st.line_chart(win_trend)

                        st.caption("Analysis includes head-to-head performance and season-wise trends.")
                
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("This can happen if the encoders are out of sync with the data. Try re-training the model.")

# --------------------------------------------------------------------
# TAB 2: CHATBOT
# --------------------------------------------------------------------
with tab2:
    st.header("IPL Chatbot (AI Assistant)")

    if not model_gemini:
        st.warning("Chatbot is disabled. Please add your GEMINI_API_KEY to the Streamlit secrets.")
    else:
        user_input = st.text_input("Ask me anything about IPL, players, or stats:")
        if st.button("Ask"):
            if user_input.strip() == "":
                st.warning("Please enter a question.")
            else:
                with st.spinner("Thinking..."):
                    try:
                        # --- Create a System Prompt for better answers ---
                        full_prompt = f"""
                        You are an expert IPL (Indian Premier League) cricket analyst. 
                        A user is asking you a question. Answer it concisely and accurately.
                        
                        User Question: {user_input}
                        """
                        response = model_gemini.generate_content(full_prompt)
                        st.write("**Answer:**")
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Error communicating with the AI: {e}")

# --------------------------------------------------------------------
# END
# --------------------------------------------------------------------
