# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ---------------- Page config ----------------
st.set_page_config(page_title="IPL Auction Predictor", page_icon="🏏", layout="wide")

# ---------------- CSS (keep login look same + glass inputs) ----------------
PAGE_CSS = """
<style>
body {
  background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
  color: #111;
}
.main {
  background-color: rgba(0,0,0,0) !important;
}
.login-card {
  background: rgba(255,255,255,0.92);
  padding: 28px;
  border-radius: 14px;
  box-shadow: 0 12px 30px rgba(15,23,42,0.08);
  max-width:800px;
  margin: 0 auto;
}
h1, h3, h4 { margin: 0; }
.input-style {
  background-color: rgba(0,0,0,0);
}
.stButton>button {
  background: linear-gradient(90deg,#ff6a00,#ee0979);
  color: white;
  font-weight:700;
  border-radius:10px;
  padding:8px 14px;
}
.result-card {
  background: rgba(255,255,255,0.95);
  border-radius:12px;
  padding:12px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
}
.small-card {
  background: #fff;
  padding:8px;
  border-radius:8px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.04);
  margin-bottom:8px;
}
</style>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# ---------------- Helper functions ----------------
def find_dataset_path():
    names = ["cricket_dataset.csv", "cricket dataset.csv", "cricket_dataset.csv".lower()]
    for n in names:
        if os.path.exists(n):
            return n
    return None

def load_model(path="model.pkl"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def train_and_save_model(dataset_path, save_path="model.pkl"):
    # Load dataset
    df = pd.read_csv(dataset_path)
    features = ["Matches","Runs","Batting Avg","Strike Rate","Wickets","Economy","Best Bowling","Stumpings","Catches"]
    X = df[features]
    y = df["Market Value (CR)"]
    # simple train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    # save
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    return model, rmse, r2

def find_similar_players(df, input_vec, topk=3):
    from sklearn.preprocessing import StandardScaler
    features = ["Matches","Runs","Batting Avg","Strike Rate","Wickets","Economy","Best Bowling","Stumpings","Catches"]
    X = df[features].values.astype(float)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iv = scaler.transform(np.array(input_vec).reshape(1,-1))
    dists = np.linalg.norm(Xs - iv, axis=1)
    idxs = np.argsort(dists)[:topk]
    return df.iloc[idxs]

# ---------------- Session state ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# Allow both common demo creds so user won't get stuck
CREDENTIALS = {"admin":"password123", "user":"user123", "admin2":"1234", "admin":"1234"}  # admin duplicated intentionally - last wins

# ---------------- LOGIN PAGE (image left + form right) ----------------
if not st.session_state.logged_in:
    left_col, right_col = st.columns([1, 1])

    # Left side image (replace with your own)
    with left_col:
        img_path = os.path.join(os.path.dirname(__file__), "cricket_image.jpg")
st.image(img_path, use_container_width=True, caption="Welcome to IPL Auction Predictor")


    # Right side login form
with right_col:
        st.markdown("<div class='login-card'>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align:center; color:#f6b01e;'>🏏 IPL Auction Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#333; margin-bottom:16px;'>Login to Continue</h3>", unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        if st.button("Login", use_container_width=True):
            ok = False
            if (username == "admin" and password in ["password123","1234"]) or (username == "user" and password == "user123"):
                ok = True
            if ok:
                st.session_state.logged_in = True
                st.success("Login Successful! Redirecting...")
                st.experimental_rerun() if hasattr(st, "experimental_rerun") else st.rerun()
            else:
                st.error("Invalid username or password 🚫")

        st.markdown("*Demo accounts:* admin/password123  or  admin/1234  or user/user123")
        st.markdown("</div>", unsafe_allow_html=True)

st.stop()
# ---------------- Load dataset path & model ----------------
dataset_path = find_dataset_path()
model = load_model()

# ---------------- PREDICTION PAGE ----------------
# Top header
st.markdown("<h1 style='text-align:center; color:#f6b01e;'>🏆 IPL Auction Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#444; margin-bottom:18px;'>Fill all fields (left) and get predicted Market Value (in Crores)</h4>", unsafe_allow_html=True)



# layout: left form, right graph
col1, col2 = st.columns([1,1])

with col1:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("### 📝 Player Details (all fields required)")
    player_name = st.text_input("Player Name (optional)")
    matches = st.number_input("Matches", min_value=0, step=1, value=0)
    runs = st.number_input("Runs", min_value=0, step=1, value=0)
    bat_avg = st.number_input("Batting Avg", min_value=0.0, step=0.1, value=0.0)
    strike_rate = st.number_input("Strike Rate", min_value=0.0, step=0.1, value=0.0)
    wickets = st.number_input("Wickets", min_value=0, step=1, value=0)
    economy = st.number_input("Economy", min_value=0.0, step=0.1, value=0.0)
    best_bowling = st.number_input("Best Bowling (wickets in a match)", min_value=0, step=1, value=0)
    stumpings = st.number_input("Stumpings", min_value=0, step=1, value=0)
    catches = st.number_input("Catches", min_value=0, step=1, value=0)

    st.markdown("---")
    if st.button("💰 Predict Price (Use model)"):
        # check model
        if model is None:
            st.error("Model not found. Please train model first (sidebar) or run model_training.py in project folder.")
        else:
            features = np.array([matches, runs, bat_avg, strike_rate, wickets, economy, best_bowling, stumpings, catches]).reshape(1,-1)
            try:
                pred = model.predict(features)[0]
                st.success(f"🏆 Predicted Market Value: ₹ {pred:.2f} Crores")
                conf = 60 + min(35, int(np.clip(pred, 0, 35)))  # rough confidence heuristic
                st.info(f"Confidence (estimate): {conf}%")
                # allow download as CSV
                out_df = pd.DataFrame([{
                    "Player Name": player_name,
                    "Matches": matches, "Runs": runs, "Batting Avg": bat_avg, "Strike Rate": strike_rate,
                    "Wickets": wickets, "Economy": economy, "Best Bowling": best_bowling,
                    "Stumpings": stumpings, "Catches": catches, "Predicted Market Value (CR)": pred
                }])
                csv_bytes = out_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Prediction (CSV)", data=csv_bytes, file_name="prediction.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prediction error: {e}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("### 📈 Player Performance Visualization")

    # Create a bar chart for the provided metrics
    metrics = ["Matches","Runs","Bat Avg","Strike Rate","Wickets","Economy","Best Bowling","Stumpings","Catches"]
    values = [matches, runs, bat_avg, strike_rate, wickets, economy, best_bowling, stumpings, catches]
    fig, ax = plt.subplots(figsize=(7,4))
    bar_colors = ['#ff8a00','#ff4b4b','#6a11cb','#2575fc','#f6b01e','#00b894','#e17055','#00cec9','#fd79a8']
    ax.bar(metrics, values, color=bar_colors)
    ax.set_xticklabels(metrics, rotation=45, ha='right', fontsize=9)
    ax.set_title("Player Stats Overview")
    ax.grid(axis='y', alpha=0.2)
    st.pyplot(fig, use_container_width=True)

    # If dataset exists show similar players
    if dataset_path and os.path.exists(dataset_path):
        try:
            df_all = pd.read_csv(dataset_path)
            sim = find_similar_players(df_all, [matches, runs, bat_avg, strike_rate, wickets, economy, best_bowling, stumpings, catches], topk=3)
            st.markdown("#### 🔎 Similar Players (from dataset)")
            for i, row in sim.iterrows():
                st.markdown(f"<div class='small-card'><b>{row['Player Name']}</b> — Market Value: ₹ {row['Market Value (CR)']:.2f} CR</div>", unsafe_allow_html=True)
        except Exception as e:
            st.info("Could not compute similar players (check dataset format).")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer / hints
st.markdown("---")

st.markdown("*Hints:* If model missing, either run python model_training.py in project folder or click sidebar 'Train model now'. Make sure cricket_dataset.csv is present and columns match exactly: Matches, Runs, Batting Avg, Strike Rate, Wickets, Economy, Best Bowling, Stumpings, Catches, Market Value (CR).")


