#!/usr/bin/env python
# coding: utf-8

# ## STEP 1 : Updating the matchdays

# In[7]:


import requests
import pandas as pd
from bs4 import BeautifulSoup
from pymongo import MongoClient
import pandas as pd

standings_url = "https://fbref.com/en/comps/9/schedule/Premier-League-Scores-and-Fixtures"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
response = requests.get(standings_url, headers=headers)

if response.status_code != 200:
    raise Exception(f"Failed to load page {standings_url}. Status code: {response.status_code}")


# In[37]:


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

# Save to MongoDB
def save_to_mongodb(dataframe, db_name, collection_name, mongo_uri):
    """
    Save a Pandas DataFrame to a MongoDB collection.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
        mongo_uri (str): The MongoDB connection URI.
    """
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    for _, row in dataframe.iterrows():
        record = row.to_dict()

        filter_query = {
            "Wk": record.get("Wk"),
            "Date": record.get("Date"),
            "Home": record.get("Home"),
            "Away": record.get("Away"),
            "Score": record.get("Score")
        }

        collection.update_one(filter_query, {"$set": record}, upsert=True)
    print(f"Processed {len(dataframe)} records in {collection_name} collection.")

mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
db_name = "epl_2024_25"
matchday_collection_name = "matchday_collection" #Collection of all the updated matchs records and scores

# Call the function to save data
save_to_mongodb(matchday_collection, db_name, matchday_collection_name, mongo_uri)


# ## Finding Relevent latest upcoming matches

# In[293]:


import pandas as pd
from pymongo import MongoClient

# MongoDB connection details
mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
db_name = "epl_2024_25"
matchday_collection_name = "matchday_collection"

def fetch_data_from_mongodb(db_name, collection_name, mongo_uri):
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Fetch all documents from the collection
    data = list(collection.find({}, {"_id": 0}))  # Exclude the `_id` field
    return pd.DataFrame(data)

matchday_collection = fetch_data_from_mongodb(db_name, matchday_collection_name, mongo_uri)

# Standardize column names
matchday_collection.columns = matchday_collection.columns.str.lower().str.strip()

# Ensure 'date' is in datetime format
if 'date' in matchday_collection.columns:
    matchday_collection['date'] = pd.to_datetime(matchday_collection['date'], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
matchday_collection = matchday_collection.dropna(subset=['date'])

# Filter for "Head-to-Head" matches and find the minimum week number
head_to_head_matches = matchday_collection[matchday_collection['match report'] == "Head-to-Head"]

if 'wk' in head_to_head_matches.columns:
    head_to_head_matches['wk'] = pd.to_numeric(head_to_head_matches['wk'], errors='coerce').fillna(0).astype(int)
    min_week = head_to_head_matches['wk'].min()

# Filter for "Match Report" matches and find the maximum week number
match_report_matches = matchday_collection[matchday_collection['match report'] == "Match Report"]

if 'wk' in match_report_matches.columns:
    match_report_matches['wk'] = pd.to_numeric(match_report_matches['wk'], errors='coerce').fillna(0).astype(int)
    max_week = match_report_matches['wk'].max()

matchday_collection['wk'] = pd.to_numeric(matchday_collection['wk'], errors='coerce').fillna(0).astype(int)

# Combine matches for both min_week and max_week
close_week_matches = matchday_collection[
    matchday_collection['wk'].isin([min_week])
]

last_week_matches = matchday_collection[
    matchday_collection['wk'].isin([max_week])
]

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
close_week_matches['home'] = close_week_matches['home'].replace(team_name_mapping)
close_week_matches['away'] = close_week_matches['away'].replace(team_name_mapping)


# In[295]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def save_to_mongodb_predictions(dataframe, db_name, collection_name, mongo_uri):
    """
    Save a Pandas DataFrame to a MongoDB collection.
    
    Args:
        dataframe (pd.DataFrame): The DataFrame to save.
        db_name (str): The name of the database.
        collection_name (str): The name of the collection.
        mongo_uri (str): The MongoDB connection URI.
    """
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    for _, row in dataframe.iterrows():
        record = row.to_dict()

        filter_query = {
            "wk": record.get("wk"),
            "date": record.get("date"),
            "time": record.get("time"),
            "home_team": record.get("home_team"),
            "home_goals": record.get("home_goals"),
            "away_team": record.get("away_team"),
            "away_goals": record.get("away_goals"),
            "score": record.get("score")            
        }

        collection.update_one(filter_query, {"$set": record}, upsert=True)
    print(f"Processed {len(dataframe)} records in {collection_name} collection.")

# Predict scorelines for upcoming matches
def predict_upcoming_matches(close_week_matches, home_model, away_model, scaler, matches_preprocessed):
    predictions = []
    
    for _, row in close_week_matches.iterrows():
        home_team = row['home']
        away_team = row['away']
        wk = row['wk']
        date = row['date']
        time = row['time']
        score = row['score']
        
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
                "date" : date,
                "time" : time,
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": predicted_home_goals,
                "away_goals": predicted_away_goals,
                "score": score                
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
    
    # Add form features (last 6 matches)
    form_features = ['gf', 'ga', 'xg', 'xga', 'poss', 'sh', 'sot']
    for feature in form_features:
        # Recent form (last 6 matches)
        matches[f'recent_{feature}'] = matches.groupby('team')[feature].transform(
            lambda x: x.rolling(6, min_periods=1).mean()
        )
        # Overall average
        matches[f'avg_{feature}'] = matches.groupby('team')[feature].transform('mean')
    
    # Add home/away indicator
    matches['is_home'] = (matches['venue'] == 'Home').astype(int)
    return matches    

def main():
    
    historical_matches_collection ="historical_matches"
    mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
    historical_matches = fetch_data_from_mongodb(db_name, historical_matches_collection, mongo_uri)
    
    # Preprocess historical data
    historical_matches_preprocessed = preprocess_data(historical_matches)
    
    # Train models using historical data
    home_model, away_model, scaler = train_models(historical_matches_preprocessed)
    
    # Generate predictions for next week's matches
    predictions_df = predict_upcoming_matches(
        close_week_matches=close_week_matches,
        home_model=home_model,
        away_model=away_model,
        scaler=scaler,
        matches_preprocessed=historical_matches_preprocessed
    )
    
    print("\nPredicted Results for Next Week Matches:")
    
    print(predictions_df)
    
    # Save predictions to MongoDB
    mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
    save_to_mongodb_predictions(predictions_df, db_name="epl_2024_25", collection_name="predictions", mongo_uri=mongo_uri)

if __name__ == "__main__":
    main()


# # Live Table Update

# In[34]:


import requests
import pandas as pd
from bs4 import BeautifulSoup

standings_url = "https://fbref.com/en/comps/9/Premier-League-Stats"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
response = requests.get(standings_url, headers=headers)

if response.status_code != 200:
    raise Exception(f"Failed to load page {standings_url}. Status code: {response.status_code}")
    
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
#save_to_mongodb(df, db_name="epl_2024_25", collection_name="livetable", mongo_uri=mongo_uri)

client = MongoClient(mongo_uri)
db = client["epl_2024_25"]
collection = db["livetable"]

for _, row in df.iterrows():
    record = row.to_dict()
    filter_query = {
        "Rk": record.get("Rk"),
        "Team": record.get("Squad"),
        "Matches Played": record.get("MP"),
        "Won": record.get("W"),
        "Draw": record.get("D"),
        "Loss": record.get("L"),
        "Goal Diffrence": record.get("GD"),
        "Points": record.get("Pts")
    }

    collection.update_one(filter_query, {"$set": record}, upsert=True)
print(f"Processed {len(df)} records in {collection} collection.")