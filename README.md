# Cricket Auction Prediction App  

A Machine Learning + Streamlit web app that predicts the **market value of cricket players** based on their past performance stats.  
This project was built as part of a Data Science assignment.  

---

# Features
- Predicts player's **market value** using:
  - Matches Played  
  - Runs Scored  
  - Wickets Taken  
  - Batting Strike Rate  
  - Bowling Economy  

- Shows **Top 10 Players** ranked by their predicted market value.  
- Interactive **Streamlit web app** with clean UI.  
- Dataset of **50+ cricket players** including batters, bowlers, allrounders, and wicket-keepers.  

---

# Dataset
- Source: Custom-built dataset with 50 players.  
- Columns used for model training:  
  - `matches`  
  - `runs`  
  - `wickets`  
  - `batting_strike_rate`  
  - `bowling_economy`  
- Target variable: **Market Value (in Cr.)**

---

# Tech Stack
- **Python 3**  
- **Pandas, NumPy** → Data processing  
- **Scikit-learn** → ML Model (Linear Regression / RandomForest)  
- **Joblib** → Model saving  
- **Streamlit** → Web App Deployment  

---

# How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/kartik2853/cricket_auction_prediction_app.git
   cd cricket_auction_prediction_app
