# UniPulse 2.0 — Smart Campus Platform
### Big Data, AI & Real-Time Analytics Engine

This folder contains the **Unified Smart Campus Platform** and the **Machine Learning & Big Data Models** built directly from the presentation requirements of **UniPulse 2.0**.

---

## 🌟 The 6 Core Features Implemented

1. **Smart Navigation & Wayfinding (`Feature 1`)**:
   - Interactive Leaflet GIS map with geolocated campus buildings (`Main Library`, `CS Block`, `Cafeteria`, `Sports Complex`, `Auditorium`, `Science Labs`).
   - Real-time color-coded capacity indicators: Green (<55%), Orange (55-85%), Red (>85% Congested).
   - Automatic routing and pathfinding navigation.

2. **AI Campus Marketplace & Fraud Detection (`Feature 2`)**:
   - Peer-to-peer campus student commerce listings.
   - Machine learning fraud detection model (`models/fraud_detection_model.pkl`) evaluating price disparity, seller age, rating history, and unverified off-platform payment attempts.
   - **Performance:** 98.67% accuracy, 0.9048 F1-Score.

3. **Campus Operations & Resource Dashboard (`Feature 3`)**:
   - Live simulated edge telemetry updating every 3 seconds: active student footfall, overall occupancy %, power consumption (kWh), and incident counts.

4. **Student Performance & Early Warning Prediction Model (`Feature 4`)**:
   - Machine learning model (`models/student_risk_model.pkl`) predicting at-risk students and estimated GPA based on attendance %, internal quiz scores, LMS logins, library hours, and lab completion.
   - Outputs: Risk Tier (`High Risk`, `Moderate Risk`, `On Track`, `High Achiever`), predicted GPA on 10.0 scale, class probabilities, and automated intervention actions.
   - **Performance:** 72.6% Classification Accuracy, Weighted F1 0.7073, GPA RMSE 0.574.

5. **Real-Time Campus Alerts System (`Feature 5`)**:
   - Low-latency pub/sub style event stream categorizing alerts into Critical Emergencies (🚨), Operational Updates (📢), and University Events (🎉).
   - Filter tabs and staff broadcast simulation.

6. **Personalized AI Recommendations (`Feature 6`)**:
   - Recommender engine (`models/recommendation_engine.py`) curating elective courses, hackathons/workshops, and specialized research facilities customized to student major, GPA, and risk level.

---

## 🚀 Quick Start Guide

### Step 1: Install Requirements
```bash
pip install flask scikit-learn pandas numpy
```

### Step 2: (Optional) Re-train the Machine Learning Models
```bash
python models/train_models.py
```

### Step 3: Run the Unified Platform
```bash
python app.py
```
Open your browser at: **`http://127.0.0.1:5050`**

### Demo Logins:
- **Student:** `student` / `student123`
- **Faculty:** `faculty` / `faculty123`
- **Admin:** `admin` / `admin123`
