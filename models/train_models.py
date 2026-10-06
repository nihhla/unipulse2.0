"""
UniPulse 2.0 - Machine Learning & Analytical Models Training Pipeline
Trains:
1. Student Performance Early Warning Model (Classifier + Regressor)
2. Marketplace Fraud & Anomaly Detection Model
3. Congestion & Footfall Analytics
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import classification_report, accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

np.random.seed(42)

# ==============================================================================
# 1. MODEL 1: STUDENT PERFORMANCE & AT-RISK PREDICTION
# ==============================================================================
def generate_and_train_student_model():
    print("\n[1/3] Training Student Performance & Risk Prediction Model...")
    n_samples = 2500

    # Simulated student features
    attendance = np.random.beta(7, 2, n_samples) * 100               # 0-100%
    internal_quiz = np.random.beta(6, 2.5, n_samples) * 100          # 0-100
    lms_logins_per_week = np.random.poisson(12, n_samples)           # count
    library_hours_per_week = np.random.exponential(6, n_samples)     # hours
    lab_sessions_pct = np.random.beta(7.5, 2, n_samples) * 100       # 0-100%
    assignment_delay_days = np.random.exponential(1.5, n_samples)   # days
    study_hours_day = np.random.gamma(3, 1.2, n_samples)            # hours/day
    engagement_score = np.random.uniform(2, 10, n_samples)          # 1-10

    # Underlying academic index
    academic_index = (
        0.30 * attendance +
        0.25 * internal_quiz +
        0.15 * lab_sessions_pct +
        0.10 * np.clip(library_hours_per_week * 5, 0, 100) +
        0.10 * np.clip(study_hours_day * 15, 0, 100) +
        0.10 * np.clip(lms_logins_per_week * 4, 0, 100) -
        0.10 * np.clip(assignment_delay_days * 12, 0, 100) +
        np.random.normal(0, 4, n_samples)
    )

    # GPA on 10.0 scale
    gpa = np.clip(academic_index / 10.0 + np.random.normal(0, 0.35, n_samples), 2.0, 10.0)

    # Risk Levels:
    # 0: High Risk (GPA < 5.5 or Attendance < 60%)
    # 1: Moderate Risk (5.5 <= GPA < 7.0)
    # 2: On Track (7.0 <= GPA < 8.5)
    # 3: High Achiever (GPA >= 8.5)
    risk_labels = []
    for g, att in zip(gpa, attendance):
        if g < 5.5 or att < 60:
            risk_labels.append("High Risk")
        elif g < 7.0:
            risk_labels.append("Moderate Risk")
        elif g < 8.5:
            risk_labels.append("On Track")
        else:
            risk_labels.append("High Achiever")

    df_students = pd.DataFrame({
        "student_id": [f"STU{1000 + i}" for i in range(n_samples)],
        "attendance": np.round(attendance, 1),
        "internal_quiz": np.round(internal_quiz, 1),
        "lms_logins": lms_logins_per_week,
        "library_hours": np.round(library_hours_per_week, 1),
        "lab_sessions_pct": np.round(lab_sessions_pct, 1),
        "assignment_delay": np.round(assignment_delay_days, 1),
        "study_hours": np.round(study_hours_day, 1),
        "engagement_score": np.round(engagement_score, 1),
        "predicted_gpa": np.round(gpa, 2),
        "risk_level": risk_labels
    })

    df_students.to_csv(os.path.join(DATA_DIR, "student_academic_records.csv"), index=False)

    feature_cols = ["attendance", "internal_quiz", "lms_logins", "library_hours", 
                    "lab_sessions_pct", "assignment_delay", "study_hours", "engagement_score"]
    X = df_students[feature_cols]
    y_class = df_students["risk_level"]
    y_reg = df_students["predicted_gpa"]

    X_train, X_test, y_train, y_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42, stratify=y_class
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Classifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Train Regressor for GPA
    reg = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    reg.fit(X_train_scaled, yr_train)
    yr_pred = reg.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)

    print(f"  Classification Accuracy: {acc * 100:.2f}% | Weighted F1: {f1:.4f}")
    print(f"  GPA Regression RMSE: {rmse:.3f} | R2 Score: {r2:.4f}")

    # Feature Importance
    importances = dict(zip(feature_cols, [round(float(x), 4) for x in clf.feature_importances_]))

    model_artifacts = {
        "classifier": clf,
        "regressor": reg,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "classes": clf.classes_.tolist()
    }

    with open(os.path.join(BASE_DIR, "student_risk_model.pkl"), "wb") as f:
        pickle.dump(model_artifacts, f)

    metrics = {
        "model_name": "Student Performance & At-Risk Predictor",
        "algorithm": "Random Forest Ensemble (Classification + Regression)",
        "samples_trained": len(X_train),
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "gpa_rmse": round(rmse, 4),
        "gpa_r2": round(r2, 4),
        "feature_importances": importances,
        "classes": clf.classes_.tolist()
    }
    with open(os.path.join(RESULTS_DIR, "student_model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ==============================================================================
# 2. MODEL 2: AI MARKETPLACE FRAUD DETECTION
# ==============================================================================
def generate_and_train_fraud_model():
    print("\n[2/3] Training AI Marketplace Fraud Detection Model...")
    n_samples = 3000

    price_diff_ratio = np.random.normal(1.0, 0.25, n_samples)       # close to 1 is normal
    seller_age_days = np.random.exponential(180, n_samples)          # days active
    seller_rating = np.random.uniform(3.5, 5.0, n_samples)           # 1-5
    past_sales = np.random.poisson(8, n_samples)                     # sales count
    daily_listings = np.random.poisson(2, n_samples)                 # listings today
    wire_request = np.random.choice([0, 1], n_samples, p=[0.94, 0.06])
    price_usd = np.random.uniform(10, 300, n_samples)

    # Incur synthetic fraud patterns
    fraud_flags = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = 0
        # Highly discounted items from brand new accounts
        if price_diff_ratio[i] < 0.35 and seller_age_days[i] < 5:
            score += 3
        # Asking for off-platform wiring
        if wire_request[i] == 1:
            score += 3
        # Account spamming 10+ listings in 1 day
        if daily_listings[i] > 8:
            score += 2
        # Abnormally low rating + low sales
        if seller_rating[i] < 2.5 and past_sales[i] < 2:
            score += 2
        if score >= 3 or (np.random.random() < 0.02):
            fraud_flags[i] = 1

    df_marketplace = pd.DataFrame({
        "item_id": [f"ITEM_{2000 + i}" for i in range(n_samples)],
        "price_ratio": np.round(price_diff_ratio, 2),
        "seller_age_days": np.round(seller_age_days, 0),
        "seller_rating": np.round(seller_rating, 2),
        "past_sales": past_sales,
        "daily_listings": daily_listings,
        "wire_request": wire_request,
        "price_usd": np.round(price_usd, 2),
        "is_fraud": fraud_flags
    })

    df_marketplace.to_csv(os.path.join(DATA_DIR, "marketplace_transactions.csv"), index=False)

    feature_cols = ["price_ratio", "seller_age_days", "seller_rating", "past_sales", "daily_listings", "wire_request"]
    X = df_marketplace[feature_cols]
    y = df_marketplace["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced", random_state=42)
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"  Marketplace Fraud Model Accuracy: {acc * 100:.2f}% | Fraud F1: {f1:.4f}")

    fraud_artifacts = {
        "model": clf,
        "scaler": scaler,
        "feature_cols": feature_cols
    }

    with open(os.path.join(BASE_DIR, "fraud_detection_model.pkl"), "wb") as f:
        pickle.dump(fraud_artifacts, f)

    fraud_metrics = {
        "model_name": "AI Marketplace Fraud Detector",
        "algorithm": "Balanced Random Forest Classifier",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4),
        "feature_cols": feature_cols,
        "total_transactions_analyzed": n_samples,
        "detected_fraud_ratio": round(float(y.mean()), 4)
    }
    with open(os.path.join(RESULTS_DIR, "fraud_model_metrics.json"), "w") as f:
        json.dump(fraud_metrics, f, indent=2)

    return fraud_metrics


# ==============================================================================
# 3. BIG DATA CAMPUS CONGESTION & FOOTFALL ANALYTICS
# ==============================================================================
def analyze_campus_footfall():
    print("\n[3/3] Processing Campus Footfall Telemetry & Hotspot Analytics...")
    events_path = os.path.join(BASE_DIR, "..", "..", "campus-dashboard", "data", "campus_events.csv")
    
    if os.path.exists(events_path):
        df = pd.read_csv(events_path)
    else:
        # Fallback to local
        df = pd.DataFrame({
            "timestamp": ["2026-10-05 10:00:00"] * 10,
            "hour": [10] * 10,
            "zone": ["Library"] * 10,
            "event": ["entry"] * 10,
            "student_id": ["STU1001"] * 10
        })

    entries = df[df["event"] == "entry"]
    zone_counts = entries["zone"].value_counts().to_dict()
    hour_counts = entries.groupby("hour").size().to_dict()

    # Peak hours sorted
    peak_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    hotspot_zones = sorted(zone_counts.items(), key=lambda x: x[1], reverse=True)

    analytics_summary = {
        "total_telemetry_events": len(df),
        "total_entries": len(entries),
        "unique_students_monitored": int(df["student_id"].nunique()),
        "zone_distribution": zone_counts,
        "hourly_distribution": {int(k): int(v) for k, v in hour_counts.items()},
        "top_hotspots": [{"zone": z, "entries": count} for z, count in hotspot_zones],
        "peak_congestion_hours": [{"hour": f"{h:02d}:00", "entries": count} for h, count in peak_hours]
    }

    with open(os.path.join(RESULTS_DIR, "campus_footfall_analytics.json"), "w") as f:
        json.dump(analytics_summary, f, indent=2)

    print("  Aggregated", len(df), "IoT/RFID events across", len(zone_counts), "campus zones.")
    print("  Top Hotspot:", hotspot_zones[0] if hotspot_zones else "N/A")
    return analytics_summary


if __name__ == "__main__":
    generate_and_train_student_model()
    generate_and_train_fraud_model()
    analyze_campus_footfall()
    print("\n[SUCCESS] All Smart Campus Machine Learning & Big Data Models Trained and Saved Successfully!")
