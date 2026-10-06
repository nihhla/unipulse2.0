"""
UniPulse 2.0: A Big Data Powered Smart Campus Platform
Unified Flask Application integrating:
1. Smart Navigation (Wayfinding & Live Building Occupancy)
2. AI Marketplace with Fraud Detection (Random Forest Anomaly Scorer)
3. Campus Operations & Analytics Dashboard (Live KPIs & Footfall Stream)
4. Student Performance & Early Warning Predictor (ML Classification & Regression)
5. Real-Time Campus Alerts (Emergency, Security, Schedule broadcast)
6. Personalized AI Academic Recommendations (Electives, Events, Labs)
7. Big Data Ingestion & Analytics Pipeline (Batch telemetry processing)
"""
import os
import sys
import json
import sqlite3
import random
import pickle
from datetime import datetime
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from werkzeug.security import generate_password_hash, check_password_hash

# Set base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Add models to sys.path for recommendation engine
sys.path.append(MODELS_DIR)
import recommendation_engine

app = Flask(__name__)
app.secret_key = "unipulse-secret-key-smart-campus-2026"

# Load Trained Machine Learning Models
STUDENT_MODEL_PATH = os.path.join(MODELS_DIR, "student_risk_model.pkl")
FRAUD_MODEL_PATH = os.path.join(MODELS_DIR, "fraud_detection_model.pkl")

student_model_data = None
fraud_model_data = None

if os.path.exists(STUDENT_MODEL_PATH):
    with open(STUDENT_MODEL_PATH, "rb") as f:
        student_model_data = pickle.load(f)

if os.path.exists(FRAUD_MODEL_PATH):
    with open(FRAUD_MODEL_PATH, "rb") as f:
        fraud_model_data = pickle.load(f)

# ---------------------------------------------------------
# DATABASE & BIG DATA LOGIN LOGGING
# ---------------------------------------------------------
DB_PATH = os.path.join(BASE_DIR, "unipulse_users.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'Student',
            full_name TEXT,
            department TEXT
        )
    """)
    # Seed default demo accounts
    demo_users = [
        ("student", generate_password_hash("student123"), "Student", "Alex Rivera", "Computer Science"),
        ("faculty", generate_password_hash("faculty123"), "Faculty", "Dr. Sarah Connor", "Computer Science"),
        ("admin", generate_password_hash("admin123"), "Administrator", "Campus Operations", "Administration")
    ]
    for u in demo_users:
        cur.execute("INSERT OR IGNORE INTO users (username, password, role, full_name, department) VALUES (?, ?, ?, ?, ?)", u)
    con.commit()
    con.close()

init_db()

def log_login_event(username, success, role):
    log_file = os.path.join(LOGS_DIR, "logins_stream.csv")
    exists = os.path.exists(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        if not exists:
            f.write("timestamp,username,status,role\n")
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{username},{'SUCCESS' if success else 'FAILED'},{role}\n")

# ---------------------------------------------------------
# CAMPUS TELEMETRY & SIMULATED SENSOR STATE
# ---------------------------------------------------------
campus_locations = [
    {"id": 1, "name": "Main Library", "code": "LIB", "lat": 12.9345, "lng": 77.6050, "capacity": 400, "occupancy": 240},
    {"id": 2, "name": "Computer Science Block", "code": "CSB", "lat": 12.9355, "lng": 77.6065, "capacity": 250, "occupancy": 160},
    {"id": 3, "name": "Main Cafeteria", "code": "CAF", "lat": 12.9335, "lng": 77.6070, "capacity": 300, "occupancy": 195},
    {"id": 4, "name": "Sports Complex & Gym", "code": "SPT", "lat": 12.9365, "lng": 77.6045, "capacity": 150, "occupancy": 45},
    {"id": 5, "name": "Central Auditorium", "code": "AUD", "lat": 12.9325, "lng": 77.6055, "capacity": 600, "occupancy": 120},
    {"id": 6, "name": "Science & Hardware Labs", "code": "LAB", "lat": 12.9370, "lng": 77.6075, "capacity": 200, "occupancy": 110}
]

state = {
    "students_on_campus": 1420,
    "occupancy_percent": 58.2,
    "energy_kwh": 342,
    "active_alerts_count": 3,
    "footfall_history": [random.randint(120, 380) for _ in range(24)],
    "alerts": [
        {"id": 1, "type": "🚨 Emergency", "severity": "critical", "message": "Fire drill scheduled at Block B, Floor 2", "time": "12:45 PM"},
        {"id": 2, "type": "📢 Campus Update", "severity": "warning", "message": "Library study floor 3 reaches 92% capacity", "time": "01:05 PM"},
        {"id": 3, "type": "🎉 Event", "severity": "info", "message": "UniPulse 2.0 Big Data Showcase at Auditorium", "time": "01:15 PM"}
    ],
    "marketplace_items": [
        {"id": "MK101", "title": "Data Structures & Algorithms in Python", "category": "Textbooks", "price": 25.0, "seller": "Ameya V.", "seller_rating": 4.9, "risk": "Safe", "fraud_prob": 0.02},
        {"id": "MK102", "title": "Raspberry Pi 4 Model B (4GB RAM)", "category": "Electronics", "price": 45.0, "seller": "Manvin R.", "seller_rating": 4.8, "risk": "Safe", "fraud_prob": 0.05},
        {"id": "MK103", "title": "Texas Instruments Graphing Calculator", "category": "Calculators", "price": 30.0, "seller": "Lidha I.", "seller_rating": 4.7, "risk": "Safe", "fraud_prob": 0.04},
        {"id": "MK104", "title": "Brand New iPhone 15 Pro Max (Quick Cash Wire)", "category": "Electronics", "price": 99.0, "seller": "Unknown_99", "seller_rating": 1.2, "risk": "Fraud Alert", "fraud_prob": 0.94}
    ]
}

def update_campus_sensors():
    # Simulate fluctuations
    state["students_on_campus"] = max(800, min(2800, state["students_on_campus"] + random.randint(-20, 25)))
    state["occupancy_percent"] = round((state["students_on_campus"] / 2600) * 100, 1)
    state["energy_kwh"] = max(240, min(500, state["energy_kwh"] + random.randint(-5, 6)))
    
    # Update footfall stream
    new_val = max(90, min(480, state["footfall_history"][-1] + random.randint(-25, 30)))
    state["footfall_history"].append(new_val)
    state["footfall_history"] = state["footfall_history"][-24:]

    # Update building occupancy
    for loc in campus_locations:
        loc["occupancy"] = max(10, min(loc["capacity"], loc["occupancy"] + random.randint(-8, 9)))
        loc["pct"] = round((loc["occupancy"] / loc["capacity"]) * 100, 1)
        loc["status"] = "red" if loc["pct"] > 85 else "orange" if loc["pct"] > 55 else "green"

# ---------------------------------------------------------
# ROUTES: AUTHENTICATION
# ---------------------------------------------------------
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        con = sqlite3.connect(DB_PATH)
        con.row_factory = sqlite3.Row
        user = con.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        con.close()

        if user and check_password_hash(user["password"], password):
            session["user"] = user["username"]
            session["role"] = user["role"]
            session["full_name"] = user["full_name"]
            session["department"] = user["department"]
            log_login_event(username, True, user["role"])
            return redirect(url_for("dashboard"))
        else:
            log_login_event(username, False, "Unknown")
            error = "Invalid username or password. Try demo accounts below."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------------------------------------------------
# ROUTES: MAIN UNIFIED SMART CAMPUS DASHBOARD
# ---------------------------------------------------------
@app.route("/dashboard")
def dashboard():
    if "user" not in session:
        return redirect(url_for("login"))
    update_campus_sensors()
    return render_template("dashboard.html", 
                           user=session, 
                           locations=campus_locations, 
                           state=state)

# ---------------------------------------------------------
# API: LIVE SENSOR & TELEMETRY STREAM
# ---------------------------------------------------------
@app.route("/api/live_telemetry")
def api_live_telemetry():
    update_campus_sensors()
    return jsonify({
        "students_on_campus": state["students_on_campus"],
        "occupancy_percent": state["occupancy_percent"],
        "energy_kwh": state["energy_kwh"],
        "footfall_history": state["footfall_history"],
        "locations": campus_locations,
        "alerts": state["alerts"],
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# ---------------------------------------------------------
# API: FEATURE 4 - STUDENT PERFORMANCE PREDICTOR
# ---------------------------------------------------------
@app.route("/api/predict_student_performance", methods=["POST"])
def predict_student_performance():
    if not student_model_data:
        return jsonify({"error": "Student performance model not loaded"}), 500
    
    data = request.json or {}
    try:
        attendance = float(data.get("attendance", 85))
        internal_quiz = float(data.get("internal_quiz", 78))
        lms_logins = float(data.get("lms_logins", 14))
        library_hours = float(data.get("library_hours", 8))
        lab_sessions = float(data.get("lab_sessions", 90))
        assignment_delay = float(data.get("assignment_delay", 0.5))
        study_hours = float(data.get("study_hours", 4.0))
        engagement_score = float(data.get("engagement_score", 7.5))

        cols = student_model_data["feature_cols"]
        feature_df = pd.DataFrame([[
            attendance, internal_quiz, lms_logins, library_hours,
            lab_sessions, assignment_delay, study_hours, engagement_score
        ]], columns=cols)

        scaler = student_model_data["scaler"]
        clf = student_model_data["classifier"]
        reg = student_model_data["regressor"]

        feature_scaled = scaler.transform(feature_df)
        risk_prediction = clf.predict(feature_scaled)[0]
        risk_probs = {cls_name: round(float(p) * 100, 1) for cls_name, p in zip(clf.classes_, clf.predict_proba(feature_scaled)[0])}
        gpa_pred = round(float(reg.predict(feature_scaled)[0]), 2)

        # Early Warning recommendations
        if risk_prediction == "High Risk":
            status_color = "#e53935"
            intervention = "URGENT INTERVENTION: Trigger faculty alert, assign academic peer mentor, and mandate weekly tutoring clinic."
        elif risk_prediction == "Moderate Risk":
            status_color = "#fb8c00"
            intervention = "MONITORING NEEDED: Recommend library study hours increase and send study guide for weak quiz topics."
        elif risk_prediction == "On Track":
            status_color = "#43a047"
            intervention = "HEALTHY STANDING: Student is progressing well. Recommend career skill development and project electives."
        else:
            status_color = "#1e88e5"
            intervention = "HONORS RECOGNITION: High Achiever. Recommend for undergraduate research assistantship and honors awards."

        return jsonify({
            "status": "success",
            "risk_level": risk_prediction,
            "predicted_gpa": gpa_pred,
            "confidence_distribution": risk_probs,
            "intervention": intervention,
            "status_color": status_color
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ---------------------------------------------------------
# API: FEATURE 2 - MARKETPLACE FRAUD DETECTION
# ---------------------------------------------------------
@app.route("/api/detect_marketplace_fraud", methods=["POST"])
def detect_marketplace_fraud():
    if not fraud_model_data:
        return jsonify({"error": "Fraud detection model not loaded"}), 500

    data = request.json or {}
    try:
        price_ratio = float(data.get("price_ratio", 1.0))
        seller_age_days = float(data.get("seller_age_days", 120))
        seller_rating = float(data.get("seller_rating", 4.5))
        past_sales = float(data.get("past_sales", 6))
        daily_listings = float(data.get("daily_listings", 2))
        wire_request = int(data.get("wire_request", 0))

        cols = fraud_model_data["feature_cols"]
        feat_df = pd.DataFrame([[price_ratio, seller_age_days, seller_rating, past_sales, daily_listings, wire_request]], columns=cols)
        scaler = fraud_model_data["scaler"]
        clf = fraud_model_data["model"]

        feat_scaled = scaler.transform(feat_df)
        pred = clf.predict(feat_scaled)[0]
        prob = clf.predict_proba(feat_scaled)[0][1]

        is_fraud = bool(pred == 1 or prob > 0.5)
        fraud_pct = round(prob * 100, 1)

        verdict = "FLAGGED AS SUSPICIOUS" if is_fraud else "VERIFIED SAFE LISTING"
        verdict_color = "#e53935" if is_fraud else "#43a047"

        return jsonify({
            "status": "success",
            "is_fraud": is_fraud,
            "fraud_probability": fraud_pct,
            "verdict": verdict,
            "verdict_color": verdict_color,
            "explanation": "High price deviation or unverified off-platform wire detected" if is_fraud else "Seller profile and price range conform to standard campus marketplace behavior."
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# ---------------------------------------------------------
# API: FEATURE 6 - PERSONALIZED RECOMMENDATIONS
# ---------------------------------------------------------
@app.route("/api/get_recommendations", methods=["POST"])
def api_get_recommendations():
    data = request.json or {}
    profile = {
        "major": data.get("major", session.get("department", "Computer Science")),
        "predicted_gpa": float(data.get("gpa", 7.8)),
        "risk_level": data.get("risk_level", "On Track"),
        "interests": data.get("interests", ["ai", "big data"])
    }
    recs = recommendation_engine.recommend(profile)
    return jsonify(recs)

# ---------------------------------------------------------
# API: BROADCAST REAL-TIME ALERT
# ---------------------------------------------------------
@app.route("/api/broadcast_alert", methods=["POST"])
def broadcast_alert():
    data = request.json or {}
    msg = data.get("message", "General Announcement")
    severity = data.get("severity", "info")
    
    icon_map = {
        "critical": "🚨 Emergency",
        "warning": "📢 Campus Update",
        "info": "🎉 Event"
    }
    new_alert = {
        "id": len(state["alerts"]) + 1,
        "type": icon_map.get(severity, "ℹ️ Info"),
        "severity": severity,
        "message": msg,
        "time": datetime.now().strftime("%I:%M %p")
    }
    state["alerts"].insert(0, new_alert)
    state["alerts"] = state["alerts"][:8]

    # Save to alerts.log
    with open(os.path.join(LOGS_DIR, "alerts.log"), "a", encoding="utf-8") as f:
        f.write(json.dumps(new_alert) + "\n")

    return jsonify({"status": "success", "alert": new_alert})

# ---------------------------------------------------------
# API: BIG DATA FOOTFALL REPORT
# ---------------------------------------------------------
@app.route("/api/big_data_summary")
def big_data_summary():
    analytics_file = os.path.join(RESULTS_DIR, "campus_footfall_analytics.json")
    if os.path.exists(analytics_file):
        with open(analytics_file, "r") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "pending", "message": "Run train_models.py to generate summary"})


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" UNIPULSE 2.0: SMART CAMPUS PLATFORM")
    print(" Big Data, AI, & Real-Time Analytics Engine")
    print("="*60)
    print(" Server launching on http://127.0.0.1:5050")
    print(" Demo accounts:")
    print("   Student:  student / student123")
    print("   Faculty:  faculty / faculty123")
    print("   Admin:    admin   / admin123")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5050, debug=False)
