import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import json
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(page_title="IPL Deep Analytics Dashboard", page_icon="🏏", layout="wide")

# --- Helper function for player avatars ---
def get_player_avatar(player_name):
    """Generates a placeholder avatar URL with player initials."""
    initials = "".join([name[0] for name in player_name.split()]).upper()
    return f"https://placehold.co/100x100/222/FFF/png?text={initials}"

# --- Data Folder Setup ---
DATA_FOLDER = 'ipl_json_data/'

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data():
    """Loads, cleans, and processes all data from the JSON files."""
    json_files = glob.glob(os.path.join(DATA_FOLDER, '*.json'))
    if not json_files:
        st.error(f"CRITICAL ERROR: No JSON data files found in '{DATA_FOLDER}'. Please ensure the folder is unzipped correctly.")
        return None, None
        
    all_matches_list, all_deliveries_list = [], []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
            match_id = os.path.basename(file_path).replace('.json', '')
            
            all_matches_list.append({
                'match_id': match_id, 'city': data['info'].get('city'), 'date': data['info']['dates'][0], 
                'venue': data['info']['venue'], 'team1': data['info']['teams'][0], 'team2': data['info']['teams'][1], 
                'toss_winner': data['info']['toss']['winner'], 'toss_decision': data['info']['toss']['decision'], 
                'winner': data['info'].get('outcome', {}).get('winner')
            })
            for inning in data['innings']:
                for over in inning['overs']:
                    for delivery in over['deliveries']:
                        all_deliveries_list.append({
                            'match_id': match_id, 'inning_team': inning['team'], 'batter': delivery['batter'], 
                            'bowler': delivery['bowler'], 'runs_scored': delivery['runs']['total'], 
                            'is_wicket': 1 if 'wickets' in delivery else 0
                        })
        except Exception: continue
    
    if not all_matches_list:
        st.error("Could not process any match files. Check data format.")
        return None, None

    all_matches_df = pd.DataFrame(all_matches_list)
    all_deliveries_df = pd.DataFrame(all_deliveries_list)
    
    # --- Data Cleaning Logic ---
    rename_map = {"Delhi Daredevils": "Delhi Capitals", "Kings XI Punjab": "Punjab Kings", "Royal Challengers Bangalore": "Royal Challengers Bengaluru", "Rising Pune Supergiant": "Rising Pune Supergiants"}
    defunct_teams = ["Deccan Chargers", "Pune Warriors", "Rising Pune Supergiants", "Gujarat Lions", "Kochi Tuskers Kerala"]
    
    for col in ['team1', 'team2', 'winner', 'toss_winner', 'inning_team']:
        if col in all_matches_df.columns: all_matches_df[col] = all_matches_df[col].replace(rename_map)
    if 'inning_team' in all_deliveries_df.columns:
        all_deliveries_df['inning_team'] = all_deliveries_df['inning_team'].replace(rename_map)

    all_matches_df = all_matches_df[~all_matches_df['team1'].isin(defunct_teams) & ~all_matches_df['team2'].isin(defunct_teams)].copy()
    all_deliveries_df = all_deliveries_df[all_deliveries_df['match_id'].isin(all_matches_df['match_id'])].copy()
    
    all_matches_df.dropna(subset=['winner'], inplace=True)
    all_matches_df['date'] = pd.to_datetime(all_matches_df['date'])
    all_matches_df = all_matches_df.sort_values('date')

    # --- Feature Engineering ---
    team_matches = pd.concat([all_matches_df[['date', 'team1', 'winner']].rename(columns={'team1': 'team'}), all_matches_df[['date', 'team2', 'winner']].rename(columns={'team2': 'team'})], ignore_index=True).sort_values(['team', 'date'])
    team_matches['is_win'] = (team_matches['team'] == team_matches['winner']).astype(int)
    team_matches['form_win_pct'] = team_matches.groupby('team')['is_win'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    team_matches['form_win_pct_prior'] = team_matches.groupby('team')['form_win_pct'].shift(1).fillna(0)
    all_matches_df = pd.merge(all_matches_df, team_matches[['date', 'team', 'form_win_pct_prior']], left_on=['date', 'team1'], right_on=['date', 'team'], how='left').rename(columns={'form_win_pct_prior': 'team1_form'})
    all_matches_df = pd.merge(all_matches_df, team_matches[['date', 'team', 'form_win_pct_prior']], left_on=['date', 'team2'], right_on=['date', 'team'], how='left').rename(columns={'form_win_pct_prior': 'team2_form'})
    
    return all_matches_df.drop(columns=['team_x', 'team_y']), all_deliveries_df

@st.cache_resource
def load_model_and_encoders():
    try:
        model = joblib.load('ipl_winner_model.pkl')
        team_encoder = joblib.load('team_encoder.pkl')
        venue_encoder = joblib.load('venue_encoder.pkl')
        toss_decision_encoder = joblib.load('toss_decision_encoder.pkl')
        return model, team_encoder, venue_encoder, toss_decision_encoder
    except FileNotFoundError as e:
        st.error(f"CRITICAL ERROR: A model file is missing: {e}. Please ensure all .pkl files are uploaded.")
        return None, None, None, None

# --- Main App Logic ---
all_matches_df, all_deliveries_df = load_and_process_data()
model, team_encoder, venue_encoder, toss_decision_encoder = load_model_and_encoders()

st.title("🏏 IPL Deep Dive Analytics & Predictor")
st.markdown("An advanced analytics platform using historical data to provide deep insights and predict match outcomes.")

if all_matches_df is None or model is None: 
    st.warning("Application is loading or encountered a critical error. Please check messages above.")
    st.stop()

# --- Sidebar Inputs ---
st.sidebar.header("Match Prediction Inputs")
active_teams = sorted(all_matches_df['team1'].unique())
team1 = st.sidebar.selectbox("Select Team 1", active_teams, index=active_teams.index('Mumbai Indians') if 'Mumbai Indians' in active_teams else 0)
team2_options = [t for t in active_teams if t != team1]
team2 = st.sidebar.selectbox("Select Team 2", team2_options, index=team2_options.index('Chennai Super Kings') if 'Chennai Super Kings' in team2_options else 0)
active_venues = sorted(all_matches_df['venue'].unique())
venue = st.sidebar.selectbox("Select Venue", active_venues, index=active_venues.index('Wankhede Stadium, Mumbai') if 'Wankhede Stadium, Mumbai' in active_venues else 0)

toss_winner = st.sidebar.radio("Toss Winner", (team1, team2))
toss_decision = st.sidebar.radio("Toss Decision", ("field", "bat"))

latest_team1_form = all_matches_df[all_matches_df['team1'] == team1].sort_values('date', ascending=False).iloc[0]['team1_form']
latest_team2_form = all_matches_df[all_matches_df['team2'] == team2].sort_values('date', ascending=False).iloc[0]['team2_form']

team1_form = st.sidebar.slider(f"{team1} Form (Last 5 Games Win %)", 0.0, 1.0, float(latest_team1_form), 0.05)
team2_form = st.sidebar.slider(f"{team2} Form (Last 5 Games Win %)", 0.0, 1.0, float(latest_team2_form), 0.05)

if st.sidebar.button("Predict & Analyze", type="primary"):
    st.header(f"Deep Dive Analysis: {team1} vs {team2}")
    st.markdown("---")
    st.header("🔮 Match Winner Prediction")
    
    try:
        input_data = {'team1_encoded': team_encoder.transform([team1])[0], 'team2_encoded': team_encoder.transform([team2])[0], 'venue_encoded': venue_encoder.transform([venue])[0], 'toss_winner_encoded': team_encoder.transform([toss_winner])[0], 'toss_decision_encoded': toss_decision_encoder.transform([toss_decision])[0], 'team1_form': team1_form, 'team2_form': team2_form}
        match_data = pd.DataFrame([input_data])
        probabilities = model.predict_proba(match_data)[0]
        
        prob_team1 = probabilities[np.where(model.classes_ == team_encoder.transform([team1])[0])[0][0]]
        prob_team2 = probabilities[np.where(model.classes_ == team_encoder.transform([team2])[0])[0][0]]
        winner = team1 if prob_team1 > prob_team2 else team2

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.subheader("Prediction Justification")
            factors = []
            if venue in winner: factors.append("a potential home advantage")
            if toss_winner == winner: factors.append("winning the toss")
            if (team1_form > team2_form and winner == team1) or (team2_form > team1_form and winner == team2): factors.append("being in superior recent form")
            justification = f"The model predicts **{winner}** as the winner. Key influencing factors could include: {', '.join(factors)}." if factors else f"The model predicts **{winner}** based on complex historical patterns."
            st.info(justification)
        with col2:
            st.subheader("Win Probability")
            prob_df = pd.DataFrame({'Team': [team1, team2], 'Probability': [prob_team1 * 100, prob_team2 * 100]})
            fig = px.pie(prob_df, values='Probability', names='Team', hole=0.4, color='Team', color_discrete_map={team1: 'royalblue', team2: 'firebrick'})
            fig.update_traces(textinfo='percent+label', pull=[0.05 if winner == team1 else 0, 0.05 if winner == team2 else 0])
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
else:
    st.info("Select match details from the sidebar and click 'Predict & Analyze' to see the full dashboard.")
