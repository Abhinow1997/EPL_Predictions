historical_matches_path = "df_union.csv"
        
historical_matches = pd.read_csv(historical_matches_path)

mongo_uri = "mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0"
db_name = "epl_2024_25"
historical_collection = "historical_matches" #Collection of all the updated matchs records and scores

def save_to_mongodb_onetime(predictions_df, db_name, collection_name, mongo_uri):
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Convert DataFrame to dictionary and insert into MongoDB
    data = predictions_df.to_dict(orient="records")
    collection.insert_many(data)
    
    print(f"Saved {len(data)} records to {collection_name} in {db_name} database.")

# Call the function to save data
save_to_mongodb_onetime(historical_matches, db_name, historical_collection, mongo_uri)