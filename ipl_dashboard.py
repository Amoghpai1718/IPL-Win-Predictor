import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import re
from urllib.parse import urlparse
from dotenv import load_dotenv
import google.generativeai as genai
from googleapiclient.discovery import build

# --- Page Configuration ---
st.set_page_config(page_title="IPL Deep Analytics Dashboard", layout="wide")
st.title("🏏 IPL Deep Analytics & Smart Chatbot Dashboard")

# --- Load environment keys ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Helper Functions ---
@st.cache_data
def load_data():
    if not os.path.exists('all_matches.csv') or not os.path.exists('all_deliveries.csv'):
        st.error("Missing data files! Please upload 'all_matches.csv' and 'all_deliveries.csv'.")
        return None, None
    matches = pd.read_csv("all_matches.csv")
    deliveries = pd.read_csv("all_deliveries.csv")
    matches["date"] = pd.to_datetime(matches["date"])
    return matches, deliveries

@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("ipl_winner_model.pkl")
    team_enc = joblib.load("team_encoder.pkl")
    venue_enc = joblib.load("venue_encoder.pkl")
    toss_enc = joblib.load("toss_decision_encoder.pkl")
    return model, team_enc, venue_enc, toss_enc

def google_search_items(query, num=6):
    try:
        service = build("customsearch", "v1", developerKey=os.getenv("GOOGLE_SEARCH_KEY"))
        res = service.cse().list(q=query, cx=os.getenv("GOOGLE_SEARCH_CX"), num=num).execute()
        return res.get("items", [])
    except Exception as e:
        return [{"title": "search_error", "snippet": f"Search error: {e}", "link": ""}]

def concise_sources(items, n=3):
    out = []
    for i, it in enumerate(items[:n], 1):
        domain = urlparse(it.get("link", "")).netloc
        out.append(f"{i}. {it.get('title','')} — {domain}")
    return "\n".join(out)

# --- Load Data & Model ---
matches_df, deliveries_df = load_data()
model, team_enc, venue_enc, toss_enc = load_model_and_encoders()

# --- Tabs Layout ---
tab1, tab2 = st.tabs(["🏆 Predict Match Winner", "💬 Ask the IPL Chatbot"])

# ============================ TAB 1: MATCH PREDICTION ============================
with tab1:
    if matches_df is not None:
        st.subheader("Match Prediction & Detailed Analysis")

        teams = sorted(matches_df["team1"].unique())
        venues = sorted(matches_df["venue"].unique())

        c1, c2, c3 = st.columns(3)
        team1 = c1.selectbox("Select Team 1", teams, index=teams.index("Mumbai Indians"))
        team2 = c2.selectbox("Select Team 2", [t for t in teams if t != team1], index=0)
        venue = c3.selectbox("Select Venue", venues, index=venues.index("Wankhede Stadium, Mumbai"))

        c4, c5, c6 = st.columns(3)
        toss_winner = c4.radio("Toss Winner", (team1, team2))
        toss_decision = c5.radio("Toss Decision", ("bat", "field"))
        st.markdown("")

        # Team form
        c7, c8 = st.columns(2)
        team1_form = c7.slider(f"{team1} Form (Win %)", 0.0, 1.0, 0.6, 0.1)
        team2_form = c8.slider(f"{team2} Form (Win %)", 0.0, 1.0, 0.4, 0.1)

        if st.button("🔮 Predict & Analyze", type="primary", use_container_width=True):
            st.markdown("---")

            # --- Head-to-Head Stats ---
            h2h = matches_df[((matches_df["team1"] == team1) & (matches_df["team2"] == team2)) |
                             ((matches_df["team1"] == team2) & (matches_df["team2"] == team1))]

            if len(h2h) > 0:
                st.subheader("Head-to-Head Analysis")
                total_matches = len(h2h)
                team1_wins = (h2h["winner"] == team1).sum()
                team2_wins = (h2h["winner"] == team2).sum()
                toss_winner_wins = (h2h["toss_winner"] == h2h["winner"]).sum()
                toss_influence = toss_winner_wins / total_matches * 100

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Matches", total_matches)
                c2.metric(f"{team1} Wins", team1_wins)
                c3.metric(f"{team2} Wins", team2_wins)
                c4.metric("Toss Winner Win %", f"{toss_influence:.1f}%")

                st.markdown("")
                win_trend = h2h.groupby("season")["winner"].value_counts().unstack().fillna(0)
                fig = px.line(win_trend, title="Season-wise Winning Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No historical head-to-head data found for these teams.")

            # --- Venue Stats ---
            st.subheader("Venue Insights")
            venue_df = matches_df[matches_df["venue"] == venue]
            venue_fav = venue_df["winner"].value_counts().head(5)
            fig = px.bar(venue_fav, title=f"Top Winning Teams at {venue}", labels={"value": "Wins", "index": "Team"})
            st.plotly_chart(fig, use_container_width=True)

            # --- Predict Winner ---
            input_data = pd.DataFrame([{
                "team1_encoded": team_enc.transform([team1])[0],
                "team2_encoded": team_enc.transform([team2])[0],
                "venue_encoded": venue_enc.transform([venue])[0],
                "toss_winner_encoded": team_enc.transform([toss_winner])[0],
                "toss_decision_encoded": toss_enc.transform([toss_decision])[0],
                "team1_form": team1_form,
                "team2_form": team2_form
            }])

            probs = model.predict_proba(input_data)[0]
            winner = team1 if probs[1] > probs[0] else team2

            st.markdown("---")
            st.subheader("Prediction Result")
            col1, col2 = st.columns([0.5, 0.5])
            with col1:
                st.success(f"**Predicted Winner:** {winner}")
                st.write(f"**Win Probability:** {max(probs)*100:.2f}%")
            with col2:
                fig = px.pie(pd.DataFrame({"Team": [team1, team2],
                                           "Probability": [probs[1]*100, probs[0]*100]}),
                             names="Team", values="Probability", hole=0.4)
                st.plotly_chart(fig, use_container_width=True)

            # --- Summary ---
            st.markdown("### Detailed Match Summary")
            st.write(f"- **Team1:** {team1} (Form: {team1_form*100:.0f}%)")
            st.write(f"- **Team2:** {team2} (Form: {team2_form*100:.0f}%)")
            st.write(f"- **Venue:** {venue}")
            st.write(f"- **Toss Winner:** {toss_winner}, chose to {toss_decision}")
            st.write(f"- **Prediction:** {winner} likely to win based on form, venue, and historical trends.")

# ============================ TAB 2: IPL CHATBOT ============================
with tab2:
    st.subheader("Ask the IPL Chatbot (Live Data + Gemini AI)")
    query = st.text_input("Type your IPL question (e.g., 'Who is the CSK captain in 2025?')")

    if query:
        with st.spinner("Fetching live data and generating answer..."):
            try:
                q_list = [
                    f"{query} site:espncricinfo.com",
                    f"{query} site:timesofindia.indiatimes.com",
                    f"{query} site:cricbuzz.com",
                    f"{query} IPL"
                ]

                results = []
                for q in q_list:
                    items = google_search_items(q)
                    if items:
                        results.extend(items)
                    if len(results) >= 5:
                        break

                if not results:
                    st.error("No results found. Check your API keys in .env.")
                else:
                    context = "\n".join([f"{it['title']} — {it['snippet']}" for it in results[:5]])
                    prompt = (
                        f"Answer concisely using only the factual information below. "
                        f"Do not add anything else. One-sentence answer only.\n\n"
                        f"CONTEXT:\n{context}\n\nQUESTION: {query}"
                    )

                    response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)
                    answer = response.text.strip() if response and response.text else "No clear answer found."
                    st.success(answer)
                    st.caption("Sources:")
                    st.text(concise_sources(results))
            except Exception as e:
                st.error(f"Error: {e}")

