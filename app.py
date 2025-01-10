from flask import Flask, render_template
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0")
db = client["epl_2024_25"]
collection = db["predictions"]

@app.route("/")
def home():
    predictions = list(collection.find({}, {"_id": 0}))
    for prediction in predictions:
        # Convert 'date' to datetime format and reformat to yyyy-mm-dd
            if 'date' in prediction and prediction['date']:
                if isinstance(prediction['date'], str):  
                    try:
                        prediction['date'] = datetime.strptime(prediction['date'], '%d-%m-%Y').strftime('%Y-%m-%d')
                    except ValueError:
                        prediction['date'] = None  
                elif isinstance(prediction['date'], datetime):
                    prediction['date'] = prediction['date'].strftime('%Y-%m-%d')
        # Ensure 'time' exists, or set a default value
            prediction['time'] = prediction.get('time', '00:00')

    predictions = sorted(predictions, key=lambda x: (x['wk'], x['date'], x['time']))


    # Group predictions by week
    grouped_predictions = {}
    for prediction in predictions:
        wk = prediction['wk']
        if wk not in grouped_predictions:
            grouped_predictions[wk] = []
        grouped_predictions[wk].append(prediction)

    grouped_predictions = dict(sorted(grouped_predictions.items(), reverse=True))

    return render_template("index.html", grouped_predictions=grouped_predictions)
        
if __name__ == "__main__":
    app.run(debug=True)