from flask import Flask, render_template
from pymongo import MongoClient
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import os

app = Flask(__name__)

# Connect to MongoDB
client = MongoClient("mongodb+srv://abhi_mongobd_user:abhi_mongobd_user@freecluster0.i05lv.mongodb.net/?retryWrites=true&w=majority&appName=FreeCluster0")
db = client["epl_2024_25"]
collection = db["predictions"]
matchday_data = db["matchday_collection"]
livetable = db["livetable"]

STATIC_DIR = os.path.join(os.getcwd(), 'static', 'images')
os.makedirs(STATIC_DIR, exist_ok=True)

def generate_form_analysis():
    # Fetch data from the MongoDB collection
    livetable_results = list(livetable.find({}, {"_id": 0}))
    form_df = pd.DataFrame(livetable_results)

    # Validate and process data
    if 'Last 5' not in form_df.columns or 'Team' not in form_df.columns:
        raise ValueError("'Last 5' or 'Team' column missing in livetable data.")

    form_analysis = []
    for _, row in form_df.iterrows():
        team = row['Team']
        last_5 = row['Last 5']
        wins = last_5.count('W')
        draws = last_5.count('D')
        losses = last_5.count('L')
        form_analysis.append({"Team": team, "Wins": wins, "Draws": draws, "Losses": losses})

    # Convert analysis to DataFrame
    analysis_df = pd.DataFrame(form_analysis)

    # First Plot: Stacked Bar Chart
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    plt.figure(figsize=(12, 8), dpi=150)
    analysis_df.plot(
        kind='bar',
        x='Team',
        y=['Wins', 'Draws', 'Losses'],
        stacked=True,
        color=['#00ff85', '#aaaaaa', '#e90052'],
        ax=plt.gca()
    )
    plt.title('Last 5 Games Form Analysis', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Teams', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='Result', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the first plot
    plot_path1 = os.path.join(STATIC_DIR, 'form_analysis.png')
    plt.savefig(plot_path1)
    plt.close()  # Clear the plot for the next one

    # Second Plot: Horizontal Bar Chart for Wins
    plt.figure(figsize=(12, 8), dpi=150)
    form_table_sorted = analysis_df.sort_values('Wins', ascending=True)
    ax = form_table_sorted.plot(
        kind='barh', 
        y='Wins', 
        x='Team',
        color=['#00ff85'],
        width=0.7
    )
    plt.title('Premier League Teams - Wins in Last 5 Games', pad=20, fontsize=16, fontweight='bold')
    plt.xlabel('Number of Wins', fontsize=12, fontweight='bold')
    plt.ylabel('Teams', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    for container in ax.containers:
        ax.bar_label(container, padding=5, fontsize=10)
    ax.set_facecolor('#f8f9fa')
    plt.gcf().set_facecolor('#ffffff')
    plt.tight_layout()

    # Save the second plot
    plot_path2 = os.path.join(STATIC_DIR, 'form_analysis2.png')
    plt.savefig(plot_path2)
    plt.close()  # Clear the plot

    return plot_path1, plot_path2

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

    last_matchday = list(matchday_data.find({}, {"_id": 0}))
    
    # Group predictions by week
    grouped_predictions = {}
    for prediction in predictions:
        wk = prediction['wk']
        if wk not in grouped_predictions:
            grouped_predictions[wk] = []
        grouped_predictions[wk].append(prediction)

    grouped_predictions = dict(sorted(grouped_predictions.items(), reverse=True))

    return render_template("index.html", grouped_predictions=grouped_predictions,last_matchday=last_matchday)

@app.route('/form-analysis')
def form_analysis_route():
    try:
        plot_path1, plot_path2 = generate_form_analysis()
        return render_template(
            "form_analysis.html",
            plot_path1=plot_path1,
            plot_path2=plot_path2
        )
    except Exception as e:
        return str(e)
        
if __name__ == "__main__":
    app.run(debug=True)