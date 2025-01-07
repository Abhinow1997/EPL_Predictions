from flask import Flask, render_template
from pymongo import MongoClient
import pandas as pd

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0")
db = client["epl_2024_25"]
collection = db["predictions"]

@app.route("/")
def home():
    # Fetch predictions from MongoDB
    predictions = list(collection.find({}, {"_id": 0}))  # Exclude `_id` field
    
    # Convert to Pandas DataFrame for easier processing
    df = pd.DataFrame(predictions)
    
    # Convert DataFrame to HTML table
    predictions_html = df.to_html(
        classes="table table-striped table-bordered",
        index=False
    )
    
    return render_template("index.html", table=predictions_html)

if __name__ == "__main__":
    app.run(debug=True)