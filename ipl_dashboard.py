import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import requests
from dotenv import load_dotenv
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(page_title="IPL Deep Analytics Dashboard", layout="wide")

# --- Helper function for player avatars ---
def get_player_avatar(player_name):
    initials = "".join([name[0] for name in player_name.split()]).upper()
    return f"https://placehold.co/100x100/222/FFF/png?text={initials}"

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data():
    if not os.path.exists('all_matches.csv') or not os.path.exists('all_deliveries.csv'):
        st.error("Error: Data files missing.")
        return None, None

    all_matches_df = pd.read_csv('all_matches.csv')
    all_deliveries_df = pd.read_csv('all_deliveries.csv')
    all_matches_df['date'] = pd.to_datetime(all_matches_df['date'])
    all_matches_df = all_matches_df.sort_values('date')

    team_matches = pd.concat([
        all_matches_df[['date', 'team1', 'winner']].rename(columns={'team1': 'team'}),
        all_matches_df[['date', 'team2', 'winner']].rename(columns={'team2': 'team'})
    ], ignore_index=True).sort_values(['team', 'date'])
    
    team_matches['is_win'] = (team_matches['team'] == team_matches['winner']).astype(int)
    team_matches['form_win_pct'] = team_matches.groupby('team')['is_win'].rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    team_matches['form_win_pct_prior'] = team_matches.groupby('team')['form_win_pct'].shift(1).fillna(0)

    all_matches_df = pd.merge(
        all_matches_df, team_matches[['date', 'team', 'form_win_pct_prior']],
        left_on=['date', 'team1'], right_on=['date', 'team'], how='left'
    ).rename(columns={'form_win_pct_prior': 'team1_form'})

    all_matches_df = pd.merge(
        all_matches_df, team_matches[['date', 'team', 'form_win_pct_prior']],
        left_on=['date', 'team2'], right_on=['date', 'team'], how='left'
    ).rename(columns={'form_win_pct_prior': 'team2_form'})

    return all_matches_df.drop(columns=['team_x', 'team_y']), all_deliveries_df

@st.cache_resource
def load_model_and_encoders():
    required_files = ['ipl_winner_model.pkl', 'team_encoder.pkl', 'venue_encoder.pkl', 'toss_decision_encoder.pkl']
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Missing model file: {f}")
            st.stop()
    return (
        joblib.load('ipl_winner_model.pkl'),
        joblib.load('team_encoder.pkl'),
        joblib.load('venue_encoder.pkl'),
        joblib.load('toss_decision_encoder.pkl')
    )

# --- Load data & model ---
all_matches_df, all_deliveries_df = load_and_process_data()
model, team_encoder, venue_encoder, toss_decision_encoder = load_model_and_encoders()

st.title("IPL Deep Analytics & Match Predictor")
st.markdown("A professional analytics platform providing insights and predictions based on historical IPL data.")

if all_matches_df is None or model is None:
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Match Prediction Inputs")
active_teams = sorted(all_matches_df['team1'].unique())
team1 = st.sidebar.selectbox("Select Team 1", active_teams, index=active_teams.index('Mumbai Indians'))
team2 = st.sidebar.selectbox("Select Team 2", [t for t in active_teams if t != team1], index=[t for t in active_teams if t != team1].index('Chennai Super Kings'))
active_venues = sorted(all_matches_df['venue'].unique())
venue = st.sidebar.selectbox("Select Venue", active_venues, index=active_venues.index('Wankhede Stadium, Mumbai'))

toss_winner = st.sidebar.radio("Toss Winner", (team1, team2))
toss_decision = st.sidebar.radio("Toss Decision", ("field", "bat"))
team1_form = st.sidebar.slider(f"{team1} Form (Win %)", 0.0, 1.0, 0.6, 0.2)
team2_form = st.sidebar.slider(f"{team2} Form (Win %)", 0.0, 1.0, 0.4, 0.2)

# --- Prediction Button ---
if st.sidebar.button("Predict & Analyze", type="primary"):
    st.header(f"Match Analysis: {team1} vs {team2}")
    
    h2h_matches_df = all_matches_df[((all_matches_df['team1'] == team1) & (all_matches_df['team2'] == team2)) | ((all_matches_df['team1'] == team2) & (all_matches_df['team2'] == team1))].copy()

    if not h2h_matches_df.empty:
        with st.expander("Head-to-Head, Venue, and Toss Insights", expanded=True):
            total_matches = len(h2h_matches_df)
            team1_wins = (h2h_matches_df['winner'] == team1).sum()
            team2_wins = (h2h_matches_df['winner'] == team2).sum()
            toss_winner_wins = (h2h_matches_df['toss_winner'] == h2h_matches_df['winner']).sum()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Matches", total_matches)
            c2.metric(f"{team1} Wins", team1_wins)
            c3.metric(f"{team2} Wins", team2_wins)
            c4.metric("Toss Winner Wins (%)", f"{toss_winner_wins / total_matches:.1%}")
    else:
        st.info("No head-to-head data available for the selected teams.")

    st.markdown("---")
    st.header("Match Winner Prediction")
    match_data = pd.DataFrame([{
        'team1_encoded': team_encoder.transform([team1])[0],
        'team2_encoded': team_encoder.transform([team2])[0],
        'venue_encoded': venue_encoder.transform([venue])[0],
        'toss_winner_encoded': team_encoder.transform([toss_winner])[0],
        'toss_decision_encoded': toss_decision_encoder.transform([toss_decision])[0],
        'team1_form': team1_form,
        'team2_form': team2_form
    }])
    probabilities = model.predict_proba(match_data)[0]
    winner = team1 if probabilities[1] > probabilities[0] else team2

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.subheader("Prediction Summary")
        st.info(f"The model predicts **{winner}** as the winner.")
    with c2:
        st.subheader("Win Probability")
        prob_df = pd.DataFrame({'Team': [team1, team2], 'Probability': [probabilities[1] * 100, probabilities[0] * 100]})
        fig = px.pie(prob_df, values='Probability', names='Team', hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Use the sidebar to configure the match and click 'Predict & Analyze'.")

# --- Gemini IPL Chatbot Integration ---
st.markdown("---")
st.header("💬 Chat with IPL Bot")

# Load API keys
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Function: Google Search Integration ---
def fetch_from_google(query):
    api_key = os.getenv("GOOGLE_SEARCH_KEY")
    cx = os.getenv("GOOGLE_SEARCH_CX")
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={cx}"
    try:
        res = requests.get(url)
        data = res.json()
        if "items" in data:
            return data["items"][0]["snippet"]
        else:
            return None
    except Exception:
        return None

# --- Chat Interface ---
user_query = st.text_input("Ask anything about IPL (e.g., 'Who is the RCB captain in 2025?')")

if user_query:
    with st.spinner("Fetching live IPL insights..."):
        search_data = fetch_from_google(user_query)
        if search_data:
            prompt = f"User asked: {user_query}\n\nLatest info from Google:\n{search_data}\n\nGive a clear, concise IPL answer."
        else:
            prompt = user_query
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(prompt)
            st.success(response.text)
        except Exception as e:
            st.error(f"⚠️ Gemini Error: {e}")


