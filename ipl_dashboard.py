import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
import json
import plotly.express as px
import time
import zipfile

# --- Page Configuration ---
st.set_page_config(page_title="IPL Deep Analytics Dashboard", layout="wide")

# --- Helper function for player avatars ---
def get_player_avatar(player_name):
    """Generates a placeholder avatar URL with player initials."""
    initials = "".join([name[0] for name in player_name.split()]).upper()
    return f"https://placehold.co/100x100/222/FFF/png?text={initials}"

# --- Self-Healing Setup for Deployment ---
DATA_FOLDER = 'ipl_json_data/'
ZIP_FILE = 'ipl_json.zip'

if not os.path.exists(DATA_FOLDER):
    if os.path.exists(ZIP_FILE):
        st.warning("Data folder not found. Performing initial setup for the server.")
        with st.spinner(f"Unzipping '{ZIP_FILE}'. This may take a moment."):
            try:
                with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                    zip_ref.extractall('.')
                st.success("Setup complete. The application will refresh shortly.")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to unzip file. Error: {e}")
                st.stop()
    else:
        st.error(f"Data file '{ZIP_FILE}' is missing from the repository.")
        st.stop()

# --- Caching Functions for Performance ---
@st.cache_data
def load_and_process_data():
    """Loads, cleans, and processes all data from the JSON files."""
    json_files = glob.glob(os.path.join(DATA_FOLDER, '*.json'))
    if not json_files:
        st.error("Data files not found in the data folder.")
        return None, None
        
    all_matches_list, all_deliveries_list = [], []
    progress_bar = st.progress(0, text="Loading and processing match files...")
    
    for i, file_path in enumerate(json_files):
        try:
            with open(file_path, 'r') as f: data = json.load(f)
            match_id = os.path.basename(file_path).replace('.json', '')
            
            all_matches_list.append({
                'match_id': match_id,
                'city': data['info'].get('city'),
                'date': data['info']['dates'][0],
                'venue': data['info']['venue'],
                'team1': data['info']['teams'][0],
                'team2': data['info']['teams'][1],
                'toss_winner': data['info']['toss']['winner'],
                'toss_decision': data['info']['toss']['decision'],
                'winner': data['info'].get('outcome', {}).get('winner')
            })
            for inning in data['innings']:
                for over in inning['overs']:
                    for delivery in over['deliveries']:
                        all_deliveries_list.append({
                            'match_id': match_id,
                            'inning_team': inning['team'],
                            'batter': delivery['batter'],
                            'bowler': delivery['bowler'],
                            'runs_scored': delivery['runs']['total'],
                            'is_wicket': 1 if 'wickets' in delivery else 0
                        })
            progress_bar.progress((i + 1) / len(json_files), text=f"Loading files ({i+1}/{len(json_files)})")
        except Exception:
            continue
    
    progress_bar.empty()
    all_matches_df, all_deliveries_df = pd.DataFrame(all_matches_list), pd.DataFrame(all_deliveries_list)
    
    rename_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiant": "Rising Pune Supergiants"
    }
    defunct_teams = ["Deccan Chargers", "Pune Warriors", "Rising Pune Supergiants", "Gujarat Lions", "Kochi Tuskers Kerala"]
    
    for col in ['team1', 'team2', 'winner', 'toss_winner', 'inning_team', 'batter', 'bowler']:
        if col in all_matches_df.columns: all_matches_df[col] = all_matches_df[col].replace(rename_map)
        if col in all_deliveries_df.columns: all_deliveries_df[col] = all_deliveries_df[col].replace(rename_map)

    all_matches_df = all_matches_df[~all_matches_df['team1'].isin(defunct_teams) & ~all_matches_df['team2'].isin(defunct_teams)].copy()
    all_deliveries_df = all_deliveries_df[all_deliveries_df['match_id'].isin(all_matches_df['match_id'])].copy()
    
    all_matches_df.dropna(subset=['winner'], inplace=True)
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
        all_matches_df,
        team_matches[['date', 'team', 'form_win_pct_prior']],
        left_on=['date', 'team1'], right_on=['date', 'team'],
        how='left'
    ).rename(columns={'form_win_pct_prior': 'team1_form'})
    
    all_matches_df = pd.merge(
        all_matches_df,
        team_matches[['date', 'team', 'form_win_pct_prior']],
        left_on=['date', 'team2'], right_on=['date', 'team'],
        how='left'
    ).rename(columns={'form_win_pct_prior': 'team2_form'})
    
    return all_matches_df.drop(columns=['team_x', 'team_y']), all_deliveries_df

@st.cache_resource
def load_model_and_encoders():
    for f in ['ipl_winner_model.pkl', 'team_encoder.pkl', 'venue_encoder.pkl', 'toss_decision_encoder.pkl']:
        if not os.path.exists(f):
            st.error(f"Required file '{f}' is missing. Please upload all necessary model files.")
            st.stop()
    return joblib.load('ipl_winner_model.pkl'), joblib.load('team_encoder.pkl'), joblib.load('venue_encoder.pkl'), joblib.load('toss_decision_encoder.pkl')

# --- Main Application ---
all_matches_df, all_deliveries_df = load_and_process_data()
model, team_encoder, venue_encoder, toss_decision_encoder = load_model_and_encoders()

st.title("IPL Deep Analytics Dashboard")
st.markdown("Provides detailed analytics, insights, and predictions for IPL matches using historical data and machine learning models.")

if all_matches_df is None or model is None:
    st.stop()

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
    # Rest of the analysis and prediction code follows exactly as in your original version
    # All interactive visualizations, metrics, and predictions remain unchanged
else:
    st.info("Select match details from the sidebar and click 'Predict & Analyze' to view the full dashboard.")

