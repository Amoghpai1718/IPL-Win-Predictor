import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
from dotenv import load_dotenv
import google.generativeai as genai
from googleapiclient.discovery import build
import re
from urllib.parse import urlparse

# =========================
# Page Configuration
# =========================
st.set_page_config(page_title="IPL Deep Analytics Dashboard", layout="wide")

# =========================
# Helper Function
# =========================
def get_player_avatar(player_name):
    """Generate placeholder avatar URL with player initials."""
    initials = "".join([name[0] for name in player_name.split()]).upper()
    return f"https://placehold.co/100x100/222/FFF/png?text={initials}"

# =========================
# Load and Process Data
# =========================
@st.cache_data
def load_and_process_data():
    if not os.path.exists('all_matches.csv') or not os.path.exists('all_deliveries.csv'):
        st.error("Missing data files: 'all_matches.csv' or 'all_deliveries.csv'.")
        return None, None
        
    all_matches_df = pd.read_csv('all_matches.csv')
    all_deliveries_df = pd.read_csv('all_deliveries.csv')
    all_matches_df['date'] = pd.to_datetime(all_matches_df['date'])
    all_matches_df = all_matches_df.sort_values('date')

    team_matches = pd.concat(
        [
            all_matches_df[['date', 'team1', 'winner']].rename(columns={'team1': 'team'}),
            all_matches_df[['date', 'team2', 'winner']].rename(columns={'team2': 'team'})
        ],
        ignore_index=True
    ).sort_values(['team', 'date'])

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

# =========================
# Load Models
# =========================
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

# =========================
# Load Data and Model
# =========================
all_matches_df, all_deliveries_df = load_and_process_data()
model, team_encoder, venue_encoder, toss_decision_encoder = load_model_and_encoders()

st.title("IPL Deep Analytics & Match Predictor")
st.markdown("Comprehensive IPL data analytics and match outcome predictions using ML models.")

if all_matches_df is None or model is None:
    st.stop()

# =========================
# Sidebar Inputs
# =========================
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

# =========================
# Prediction Section
# =========================
if st.sidebar.button("Predict & Analyze", type="primary"):
    st.header(f"Match Analysis: {team1} vs {team2}")

    h2h = all_matches_df[((all_matches_df['team1'] == team1) & (all_matches_df['team2'] == team2)) |
                         ((all_matches_df['team1'] == team2) & (all_matches_df['team2'] == team1))]

    if not h2h.empty:
        with st.expander("Head-to-Head & Toss Insights", expanded=True):
            total = len(h2h)
            t1_wins = (h2h['winner'] == team1).sum()
            t2_wins = (h2h['winner'] == team2).sum()
            toss_impact = (h2h['toss_winner'] == h2h['winner']).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Matches", total)
            c2.metric(f"{team1} Wins", t1_wins)
            c3.metric(f"{team2} Wins", t2_wins)
            c4.metric("Toss-Win Correlation", f"{toss_impact / total:.1%}")
    else:
        st.info("No head-to-head data found for the selected teams.")

    st.markdown("---")
    st.header("Predicted Winner")

    match_df = pd.DataFrame([{
        'team1_encoded': team_encoder.transform([team1])[0],
        'team2_encoded': team_encoder.transform([team2])[0],
        'venue_encoded': venue_encoder.transform([venue])[0],
        'toss_winner_encoded': team_encoder.transform([toss_winner])[0],
        'toss_decision_encoded': toss_decision_encoder.transform([toss_decision])[0],
        'team1_form': team1_form,
        'team2_form': team2_form
    }])

    probs = model.predict_proba(match_df)[0]
    predicted_winner = team1 if probs[1] > probs[0] else team2

    col1, col2 = st.columns([0.6, 0.4])
    with col1:
        st.subheader("Prediction Summary")
        st.info(f"Predicted Winner: **{predicted_winner}**")
    with col2:
        prob_df = pd.DataFrame({'Team': [team1, team2], 'Probability': [probs[1]*100, probs[0]*100]})
        fig = px.pie(prob_df, values='Probability', names='Team', hole=0.4)
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select match details on the sidebar and click 'Predict & Analyze'.")

# =========================
# Gemini + Google Search Chatbot
# =========================
st.markdown("---")
st.header("💬 Chat with IPL Bot (Live Verified Answers)")

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def google_search_items(query, num=5):
    """Fetch top Google search items."""
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_SEARCH_CX"), num=num).execute()
        items = res.get("items", [])
        return [{"title": i.get("title",""), "link": i.get("link",""), "snippet": i.get("snippet","")} for i in items]
    except Exception as e:
        return [{"title": "search_error", "link": "", "snippet": f"Error: {e}"}]

def format_sources(items, max_items=3):
    out = []
    for i, it in enumerate(items[:max_items], 1):
        link, domain = it.get("link",""), urlparse(it.get("link","")).netloc
        out.append(f"{i}. {it.get('title','')}\n{it.get('snippet','')}\n{domain} — {link}")
    return "\n\n".join(out)

def explicit_fact_from_items(items):
    """Extract a clear factual line (like captain info)."""
    for it in items:
        text = " ".join([it.get("title",""), it.get("snippet","")]).lower()
        if "captain" in text or "skipper" in text:
            return it.get("snippet",""), it.get("link","")
    return None, None

user_query = st.text_input("Ask about IPL (e.g., 'Who is the RCB captain now?')")

if user_query:
    with st.spinner("Searching for the latest information..."):
        try:
            q_variants = [
                f"{user_query} site:espncricinfo.com",
                f"{user_query} site:timesofindia.indiatimes.com",
                f"{user_query} site:thehindu.com",
                f"{user_query} site:royalchallengers.com",
                f"{user_query}"
            ]
            items = []
            for q in q_variants:
                items = google_search_items(q, num=5)
                if items and not items[0].get("title") == "search_error":
                    break

            if not items:
                st.error("No search results or invalid API keys.")
            else:
                fact, source = explicit_fact_from_items(items)
                if fact:
                    st.success(f"**Answer (from web):** {fact}\n\nSource: {source}")
                    st.subheader("Top Sources")
                    st.text(format_sources(items, max_items=3))
                else:
                    context = format_sources(items, max_items=5)
                    prompt = (
                        "You are a factual assistant. Use only the below sources to answer briefly. "
                        "Cite the source URLs in one short sentence.\n\n"
                        f"SOURCES:\n{context}\n\nQUESTION: {user_query}"
                    )
                    response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)
                    st.success(response.text)
                    st.subheader("Sources Used")
                    st.text(context)
        except Exception as e:
            st.error(f"Error while fetching data: {e}")

