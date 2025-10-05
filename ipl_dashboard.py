import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

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
    """Loads pre-cleaned CSV data and engineers features."""
    if not os.path.exists('all_matches.csv') or not os.path.exists('all_deliveries.csv'):
        st.error("Error: The clean data files ('all_matches.csv', 'all_deliveries.csv') are missing.")
        return None, None
        
    all_matches_df = pd.read_csv('all_matches.csv')
    all_deliveries_df = pd.read_csv('all_deliveries.csv')
    
    all_matches_df['date'] = pd.to_datetime(all_matches_df['date'])
    all_matches_df = all_matches_df.sort_values('date')

    # --- Feature Engineering ---
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
    """Loads the trained model and encoders from disk."""
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

# --- Main App Logic ---
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
            
            t1_venue_stats = all_matches_df[(all_matches_df['venue'] == venue) & ((all_matches_df['team1'] == team1) | (all_matches_df['team2'] == team1))]
            t2_venue_stats = all_matches_df[(all_matches_df['venue'] == venue) & ((all_matches_df['team1'] == team2) | (all_matches_df['team2'] == team2))]
            
            c1, c2 = st.columns(2)
            with c1:
                st.metric(f"{team1} Win % at {venue}", f"{(t1_venue_stats['winner'] == team1).sum() / len(t1_venue_stats):.1%}" if len(t1_venue_stats)>0 else "N/A", f"{len(t1_venue_stats)} matches")
            with c2:
                st.metric(f"{team2} Win % at {venue}", f"{(t2_venue_stats['winner'] == team2).sum() / len(t2_venue_stats):.1%}" if len(t2_venue_stats)>0 else "N/A", f"{len(t2_venue_stats)} matches")

        with st.expander("Recent Performance and Score Trends", expanded=True):
            def get_recent_performance(team_name):
                team_matches = all_matches_df[(all_matches_df['team1'] == team_name) | (all_matches_df['team2'] == team_name)].sort_values('date', ascending=False).head(5)
                if team_matches.empty: 
                    return "N/A", "N/A"
                match_ids = team_matches['match_id'].tolist()
                team_deliveries = all_deliveries_df[(all_deliveries_df['match_id'].isin(match_ids)) & (all_deliveries_df['inning_team'] == team_name)]
                avg_runs = team_deliveries.groupby('match_id')['runs_scored'].sum().mean()
                avg_wickets = team_deliveries.groupby('match_id')['is_wicket'].sum().mean()
                return f"{avg_runs:.0f}", f"{avg_wickets:.0f}"
            
            t1_score, t1_wickets = get_recent_performance(team1)
            t2_score, t2_wickets = get_recent_performance(team2)
            
            c1, c2 = st.columns(2)
            c1.metric(f"Avg. Score - {team1}", t1_score, f"Avg. Wickets Lost: {t1_wickets}")
            c2.metric(f"Avg. Score - {team2}", t2_score, f"Avg. Wickets Lost: {t2_wickets}")

        with st.expander("Key Players and Recent Form", expanded=True):
            h2h_deliveries_df = all_deliveries_df[all_deliveries_df['match_id'].isin(h2h_matches_df['match_id'])]
            
            st.subheader("Impact Players in This Fixture")
            c1, c2 = st.columns(2)
            with c1:
                batsman_stats = h2h_deliveries_df.groupby('batter').agg(total_runs=('runs_scored', 'sum'), balls_faced=('batter', 'count'))
                batsman_stats = batsman_stats[batsman_stats['balls_faced'] > 30].sort_values('total_runs', ascending=False).head(5)
                fig = px.bar(batsman_stats, y=batsman_stats.index, x='total_runs', orientation='h', title="Top Batsmen", labels={'total_runs': 'Runs', 'batter': 'Batsman'})
                fig.update_layout(yaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                bowler_stats = h2h_deliveries_df[h2h_deliveries_df['is_wicket'] == 1].groupby('bowler').agg(total_wickets=('is_wicket', 'count'))
                bowler_stats = bowler_stats.sort_values('total_wickets', ascending=False).head(5)
                fig = px.bar(bowler_stats, y=bowler_stats.index, x='total_wickets', orientation='h', title="Top Bowlers", labels={'total_wickets': 'Wickets', 'bowler': 'Bowler'})
                fig.update_layout(yaxis={'categoryorder':'total descending'})
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Players in Recent Form (Last 5 Games)")
            def display_recent_players(team_name):
                st.markdown(f"#### {team_name}")
                team_matches = all_matches_df[(all_matches_df['team1'] == team_name) | (all_matches_df['team2'] == team_name)].sort_values('date', ascending=False).head(5)
                if team_matches.empty:
                    st.text("No recent matches available.")
                    return
                team_deliveries = all_deliveries_df[all_deliveries_df['match_id'].isin(team_matches['match_id'])]
                top_batsmen = team_deliveries[team_deliveries['inning_team'] == team_name].groupby('batter')['runs_scored'].sum().nlargest(3)
                top_bowlers = team_deliveries[team_deliveries['inning_team'] != team_name].groupby('bowler')['is_wicket'].sum().nlargest(3)

                st.markdown("##### Top Batsmen")
                for player, runs in top_batsmen.items():
                    with st.container(border=True):
                        c1, c2 = st.columns([0.25, 0.75])
                        c1.image(get_player_avatar(player), width=70)
                        c2.metric(label=player, value=f"{runs} Runs")
                st.markdown("##### Top Bowlers")
                for player, wickets in top_bowlers.items():
                    with st.container(border=True):
                        c1, c2 = st.columns([0.25, 0.75])
                        c1.image(get_player_avatar(player), width=70)
                        c2.metric(label=player, value=f"{wickets} Wickets")
            c1, c2 = st.columns(2)
            with c1: display_recent_players(team1)
            with c2: display_recent_players(team2)
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
        factors = []
        if venue in winner:
            factors.append("venue advantage")
        if toss_winner == winner:
            factors.append("toss outcome")
        if (team1_form > team2_form if winner == team1 else team2_form > team1_form):
            factors.append("better recent form")
        justification = f"The model predicts **{winner}** as the winner. Key contributing factors: {', '.join(factors)}." if factors else f"The model predicts **{winner}** as the winner based on historical data."
        st.info(justification)
    with c2:
        st.subheader("Win Probability")
        prob_df = pd.DataFrame({'Team': [team1, team2], 'Probability': [probabilities[1] * 100, probabilities[0] * 100]})
        fig = px.pie(prob_df, values='Probability', names='Team', hole=0.4, color='Team', color_discrete_map={team1: 'royalblue', team2: 'firebrick'})
        fig.update_traces(textinfo='percent+label', pull=[0.05 if winner == team1 else 0, 0.05 if winner == team2 else 0])
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Use the sidebar to configure the match details and click 'Predict & Analyze' to generate the full report.")
