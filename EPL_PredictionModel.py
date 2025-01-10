#!/usr/bin/env python
# coding: utf-8

# ## STEP 1 : Updating the matchdays

# In[30]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

standings_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
response = requests.get(standings_url, headers=headers)

if response.status_code != 200:
    raise Exception(f"Failed to load page {standings_url}. Status code: {response.status_code}")


# In[36]:


soup = BeautifulSoup(response.text, 'html.parser')

# Find the standings table
standings_table = soup.select_one('table.stats_table')
if not standings_table:
    raise Exception("Standings table not found.")

headers = [th.get_text(strip=True) for th in standings_table.find('thead').find_all('th')]

# Extract rows and align with headers
rows = standings_table.find('tbody').find_all('tr')
data = []

for row in rows:
    columns = row.find_all(['th', 'td'])  # Include <th> for row headers
    row_data = [col.get_text(strip=True) for col in columns]
    
    # Check if the number of columns matches the headers
    if len(row_data) == len(headers):
        data.append(dict(zip(headers, row_data)))
    else:
        # Handle cases where some columns are missing or extra
        print(f"Row with mismatched columns skipped: {row_data}")

matchday_collection = pd.DataFrame(data)

# matchday_collection.count()
matchday_collection.to_csv("matchday_collection.csv")


# ## Finding Relevent latest upcoming matches

# In[50]:


import pandas as pd

# Load the data
file_path = "matchday_collection.csv"
matchday_collection = pd.read_csv(file_path)

# Standardize column names
matchday_collection.columns = matchday_collection.columns.str.lower().str.strip()

# Ensure 'date' is in datetime format
if 'date' in matchday_collection.columns:
    matchday_collection['date'] = pd.to_datetime(matchday_collection['date'], dayfirst=True, errors='coerce')

    # Drop rows where 'date' could not be parsed (invalid dates)
    matchday_collection = matchday_collection.dropna(subset=['date'])

# Filter for upcoming matches where 'match report' is "Head-to-Head"
upcoming_matches = matchday_collection[matchday_collection['match report'] == "Head-to-Head"]

# Find the minimum week (Wk) value
if 'wk' in upcoming_matches.columns:
    min_week = upcoming_matches['wk'].min()
    nxt_week_matches = upcoming_matches[upcoming_matches['wk'].isin((min_week, min_week+1)) ]

# Mapping of team abbreviations to full names
team_name_mapping = {
    "Manchester Utd": "Manchester United",
    "Manchester City": "Manchester City",
    "Tottenham": "Tottenham Hotspur",
    "Nott'ham Forest": "Nottingham Forest",
    "Ipswich Town": "Ipswich Town",
    "Wolves": "Wolverhampton Wanderers",
    "West Ham": "West Ham United",
    "Brighton": "Brighton and Hove Albion",
    "Arsenal": "Arsenal",
    "Liverpool": "Liverpool",
    "Chelsea": "Chelsea",
    "Newcastle Utd": "Newcastle United",
    "Bournemouth": "Bournemouth",
    "Southampton": "Southampton",
    "Crystal Palace": "Crystal Palace",
    "Leicester City": "Leicester City",
    "Aston Villa": "Aston Villa",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Brentford": "Brentford"
}


# Update team names in 'home' and 'away' columns
nxt_week_matches['home'] = nxt_week_matches['home'].replace(team_name_mapping)
nxt_week_matches['away'] = nxt_week_matches['away'].replace(team_name_mapping)

#nxt_week_matches


# In[6]:


from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# In[73]:


from pymongo import MongoClient
import pandas as pd

db_name="abhi_mongobd_user"
collection_name="epl2024_25.epl_predictions"
mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"

def save_to_mongodb(predictions_df, db_name, collection_name, mongo_uri):
    """
    Save the predictions DataFrame to a MongoDB collection.
    
    Args:
        predictions_df (pd.DataFrame): The DataFrame containing predictions.
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
        mongo_uri (str): The MongoDB connection URI.
    """
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Convert DataFrame to dictionary and insert into MongoDB
    data = predictions_df.to_dict(orient="records")
    collection.insert_many(data)
    
    print(f"Saved {len(data)} records to {collection_name} in {db_name} database.")

# Predict scorelines for upcoming matches
def predict_upcoming_matches(nxt_week_matches, home_model, away_model, scaler, matches_preprocessed):
    predictions = []
    
    for _, row in nxt_week_matches.iterrows():
        home_team = row['home']
        away_team = row['away']
        wk = row['wk']
        
        # Get stats for both teams from preprocessed data
        try:
            home_stats = matches_preprocessed[matches_preprocessed['team'] == home_team].iloc[-1]
            away_stats = matches_preprocessed[matches_preprocessed['team'] == away_team].iloc[-1]
            
            # Create feature vectors
            home_features = np.array([
                home_stats['recent_gf'], home_stats['recent_ga'],
                home_stats['recent_xg'], home_stats['recent_xga'],
                home_stats['recent_poss'], home_stats['recent_sh'],
                home_stats['recent_sot'], home_stats['avg_gf'],
                home_stats['avg_ga'], home_stats['avg_xg'],
                home_stats['avg_xga'], 1  # Home team indicator
            ]).reshape(1, -1)
            
            away_features = np.array([
                away_stats['recent_gf'], away_stats['recent_ga'],
                away_stats['recent_xg'], away_stats['recent_xga'],
                away_stats['recent_poss'], away_stats['recent_sh'],
                away_stats['recent_sot'], away_stats['avg_gf'],
                away_stats['avg_ga'], away_stats['avg_xg'],
                away_stats['avg_xga'], 0  # Away team indicator
            ]).reshape(1, -1)
            
            # Scale features and predict scores
            home_features_scaled = scaler.transform(home_features)
            away_features_scaled = scaler.transform(away_features)
            
            predicted_home_goals = round(home_model.predict(home_features_scaled)[0])
            predicted_away_goals = round(away_model.predict(away_features_scaled)[0])
            
            predictions.append({
                "wk": wk,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": predicted_home_goals,
                "away_goals": predicted_away_goals
            })
        
        except IndexError:
            print(f"Insufficient data for {home_team} vs {away_team}")
    
    return pd.DataFrame(predictions)

# Train models using historical data
def train_models(matches_preprocessed):
    features = [
        'recent_gf', 'recent_ga', 'recent_xg', 'recent_xga',
        'recent_poss', 'recent_sh', 'recent_sot',
        'avg_gf', 'avg_ga', 'avg_xg', 'avg_xga',
        'is_home'
    ]
    
    X = matches_preprocessed[features]
    y_home_goals = matches_preprocessed['gf']
    y_away_goals = matches_preprocessed['ga']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train home and away goal models
    home_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    away_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        min_samples_split=5,
        random_state=42
    )
    
    home_model.fit(X_scaled, y_home_goals)
    away_model.fit(X_scaled, y_away_goals)
    
    return home_model, away_model, scaler

# Preprocess historical match data
def preprocess_data(matches):
    matches['date'] = pd.to_datetime(matches['date'], dayfirst=True)
    matches = matches.sort_values(by=['team', 'date'])
    
    # Add form features (last 5 matches)
    form_features = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot']
    for feature in form_features:
        # Recent form (last 5 matches)
        matches[f'recent_{feature}'] = matches.groupby('team')[feature].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        # Overall average
        matches[f'avg_{feature}'] = matches.groupby('team')[feature].transform('mean')
    
    # Add home/away indicator
    matches['is_home'] = (matches['venue'] == 'Home').astype(int)
    
    return matches    

def main():
    # Load historical match data and upcoming match data
    historical_matches_path = "df_union.csv"
        
    historical_matches = pd.read_csv(historical_matches_path)
        
    # Preprocess historical data
    historical_matches_preprocessed = preprocess_data(historical_matches)
    
    # Train models using historical data
    home_model, away_model, scaler = train_models(historical_matches_preprocessed)
    
    # Generate predictions for next week's matches
    predictions_df = predict_upcoming_matches(
        nxt_week_matches=nxt_week_matches,
        home_model=home_model,
        away_model=away_model,
        scaler=scaler,
        matches_preprocessed=historical_matches_preprocessed
    )
    
    print("\nPredicted Results for Next Week Matches:")
    print(predictions_df)
    
    # Save predictions to MongoDB
    mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
    save_to_mongodb(predictions_df, db_name="epl_2024_25", collection_name="predictions", mongo_uri=mongo_uri)

def livetable_update():    
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the standings table
    standings_table = soup.select_one('table.stats_table')
    if not standings_table:
        raise Exception("Standings table not found.")

    headers = [th.get_text(strip=True) for th in standings_table.find('thead').find_all('th')]

    # Extract rows and align with headers
    rows = standings_table.find('tbody').find_all('tr')
    data = []

    for row in rows:
        columns = row.find_all(['th', 'td'])  
        row_data = [col.get_text(strip=True) for col in columns]
        
        if len(row_data) == len(headers):
            data.append(dict(zip(headers, row_data)))
        else:
            print(f"Row with mismatched columns skipped: {row_data}")

    df = pd.DataFrame(data)

    # Save predictions to MongoDB
    mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
    save_to_mongodb(df, db_name="epl_2024_25", collection_name="livetable", mongo_uri=mongo_uri)

if __name__ == "__main__":
    livetable_update()
    main()

