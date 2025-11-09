import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import requests
from dotenv import load_dotenv
import google.generativeai as genai
from googleapiclient.discovery import build

# --- Page Configuration ---
st.set_page_config(page_title="IPL Deep Analytics Dashboard", layout="wide")

# --- Helper function for player avatars ---
def get_player_avatar(player_name):
    """Generates a placeholder avatar URL with player initials."""
    initials = "".join([name[0] for name in player_name.split()]).upper()
    return f"https://placehold.co/100x100/222/FFF/png?text={initials}"

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data():
    if not os.path.exists('all_matches.csv') or not os.path.exists('all_deliveries.csv'):
        st.error("Error: The clean data files ('all_matches.csv', 'all_deliveries.csv') are missing.")
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

@st.cache_resource
def load_model_and_encoders():
    required_files = ['ipl_winner_model.pkl', 'team_encoder.pkl', 'venue_encoder.pkl', 'toss_decision_encoder.pkl']
    for f in required_files:
        if not os.path.exists(f):
            st.error(f"Error: The model file '{f}' is missing. Please upload it to the repository.")
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
        st.info(f"The model predicts *{winner}* as the likely winner based on data trends.")
    with c2:
        st.subheader("Win Probability")
        prob_df = pd.DataFrame({'Team': [team1, team2], 'Probability': [probabilities[1] * 100, probabilities[0] * 100]})
        fig = px.pie(prob_df, values='Probability', names='Team', hole=0.4)
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Use the sidebar to configure the match details and click 'Predict & Analyze'.")

# --- Gemini + Google Search Chatbot Integration ---
# --- Gemini + Google Search Chatbot Integration (improved) ---
import google.generativeai as genai
from googleapiclient.discovery import build
from dotenv import load_dotenv
import re
from urllib.parse import urlparse

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def google_search_items(query, num=5):
    """Return top search items with title, link, snippet."""
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_SEARCH_CX"), num=num).execute()
        items = res.get("items", [])
        results = []
        for it in items:
            results.append({
                "title": it.get("title", ""),
                "link": it.get("link", ""),
                "snippet": it.get("snippet", "")
            })
        return results
    except Exception as e:
        return [{"title": "search_error", "link": "", "snippet": f"Search error: {e}"}]

def explicit_fact_from_items(items, keywords=("captain", "skipper")):
    """Look for an explicit mention like 'X is the captain' in title/snippet.
       Returns (fact_text, source_link) or (None, None)."""
    pattern = re.compile(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\s+(?:is|named|appointed|announced|confirmed|captain|skipper)", re.IGNORECASE)
    for it in items:
        text = " ".join([it.get("title",""), it.get("snippet","")])
        # direct mentions of "captain" or "skipper" with a name
        if any(k in text.lower() for k in keywords):
            # try to extract "Name is the captain" pattern
            m = re.search(r"([A-Z][a-z]+(?:\s[A-Z][a-z]+))\s+(?:is|named|appointed|confirmed).{0,30}captain|captain\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+))", text)
            if m:
                name = (m.group(1) or m.group(2)).strip() if m.groups() else None
                if name:
                    return (f"{name} (from snippet: {it.get('snippet','')})", it.get("link"))
            # fallback: return full snippet and link if it mentions "captain"
            return (it.get("snippet",""), it.get("link"))
    return (None, None)

def format_sources(items, max_items=3):
    out = []
    for i, it in enumerate(items[:max_items], start=1):
        link = it.get("link","")
        domain = urlparse(link).netloc
        out.append(f"{i}. {it.get('title','')}\n{it.get('snippet','')}\n{domain} — {link}")
    return "\n\n".join(out)

st.markdown("---")
st.header("💬 Chat with IPL Bot (Live + Cited)")

user_query = st.text_input("Ask anything about IPL (e.g., 'Who is the RCB captain now?')")

if user_query:
    with st.spinner("Fetching live info and verifying..."):
        try:
            # 1) Try targeted search queries for authoritative sources first
            q_variants = [
                f"{user_query} RCB captain site:espncricinfo.com",
                f"{user_query} RCB captain site:timesofindia.indiatimes.com",
                f"{user_query} RCB captain site:thehindu.com",
                f"{user_query} RCB captain site:royalchallengers.com",
                f"{user_query} RCB captain"
            ]
            items = []
            for q in q_variants:
                items = google_search_items(q, num=5)
                # if we got results, stop early
                if items and not items[0].get("snippet","").startswith("Search error"):
                    break

            if not items:
                st.error("No search results. Check your GOOGLE_SEARCH_KEY and GOOGLE_SEARCH_CX.")
            else:
                # 2) Look for an explicit fact
                fact, source = explicit_fact_from_items(items)
                if fact:
                    st.success(f"Answer (from web): {fact}\n\nSource: {source}")
                    st.subheader("Source excerpts")
                    st.text(format_sources(items, max_items=3))
                else:
                    # 3) Give the model the top snippets and ask for a concise answer + cite
                    context = format_sources(items, max_items=5)
                    prompt = (
                        "You are a fact-focused assistant. Use ONLY the sources below to answer the question. "
                        "If the sources disagree, say so and list which source says what. "
                        "Cite the source URLs in your final short answer (one sentence answer + sources). "
                        f"SOURCES:\n{context}\n\nQUESTION: {user_query}\n\nAnswer concisely and include citation links."
                    )

                    # Use a currently available model name on your machine (confirm with list_gemini_models)
                    response = genai.GenerativeModel(model_name="models/gemini-2.5-flash").generate_content(prompt)
                    # Show model's answer and the source excerpts
                    st.success(response.text)
                    st.subheader("Source excerpts used for context")
                    st.text(context)
        except Exception as e:
            st.error(f"Error while fetching/answering: {e}")
