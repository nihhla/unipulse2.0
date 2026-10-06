"""
UniPulse 2.0 - Personalized Recommendation Engine
Implements AI-curated suggestions for:
1. Elective Courses & Specializations
2. Campus Academic & Tech Events
3. Specialized Study Facilities & Labs
Tailored to each student's major, interests, GPA tier, and activity patterns.
"""
import json
import os

CATALOG = {
    "courses": [
        {
            "id": "CS401",
            "title": "Big Data Engineering & Apache Spark",
            "department": "Computer Science",
            "level": "Advanced",
            "tags": ["big data", "spark", "hadoop", "cloud", "kafka"],
            "description": "Distributed data pipelines, streaming architectures, and analytics at scale.",
            "target_gpa_min": 6.5
        },
        {
            "id": "CS405",
            "title": "Applied Machine Learning & Deep Learning",
            "department": "Computer Science",
            "level": "Advanced",
            "tags": ["ai", "machine learning", "neural networks", "python", "scikit-learn"],
            "description": "Supervised & unsupervised models, predictive analytics, and computer vision.",
            "target_gpa_min": 7.0
        },
        {
            "id": "EC302",
            "title": "IoT Edge Computing & Sensor Systems",
            "department": "Electronics / IoT",
            "level": "Intermediate",
            "tags": ["iot", "sensors", "embedded", "smart campus", "hardware"],
            "description": "Smart sensor integration, microcontrollers, and low-latency mesh communication.",
            "target_gpa_min": 6.0
        },
        {
            "id": "BA205",
            "title": "Business Intelligence & Data Visualization",
            "department": "Management / IT",
            "level": "Beginner to Intermediate",
            "tags": ["power bi", "dashboard", "analytics", "sql", "reporting"],
            "description": "Executive KPI dashboards, storytelling with data, and operational forecasting.",
            "target_gpa_min": 5.0
        },
        {
            "id": "CS101R",
            "title": "Foundational Python & Algorithmic Thinking (Remedial)",
            "department": "Computer Science",
            "level": "Foundational",
            "tags": ["python", "basics", "tutoring", "remedial", "programming"],
            "description": "Hands-on guided practice and peer mentorship for core programming.",
            "target_gpa_min": 0.0
        }
    ],
    "events": [
        {
            "id": "EVT101",
            "title": "Smart Campus AI Hackathon 2026",
            "zone": "Auditorium",
            "date": "Oct 12, 2026 - 10:00 AM",
            "tags": ["ai", "hackathon", "innovation", "big data", "programming"],
            "recommended_for": ["Computer Science", "Electronics / IoT", "Management / IT"]
        },
        {
            "id": "EVT102",
            "title": "Industry Keynote: Next-Gen Cloud & Spark Architectures",
            "zone": "LectureHalls",
            "date": "Oct 15, 2026 - 03:00 PM",
            "tags": ["cloud", "spark", "big data", "careers"],
            "recommended_for": ["Computer Science", "Management / IT"]
        },
        {
            "id": "EVT103",
            "title": "Peer Tutoring & Academic Recovery Workshop",
            "zone": "Library",
            "date": "Every Wednesday - 04:00 PM",
            "tags": ["tutoring", "academic recovery", "study skills", "remedial"],
            "recommended_for": ["High Risk", "Moderate Risk"]
        },
        {
            "id": "EVT104",
            "title": "Robotics & IoT Sensor Hands-on Demo",
            "zone": "Labs",
            "date": "Oct 18, 2026 - 02:00 PM",
            "tags": ["iot", "sensors", "hardware", "robotics"],
            "recommended_for": ["Electronics / IoT", "Computer Science"]
        }
    ],
    "resources": [
        {
            "id": "RES01",
            "name": "NVIDIA GPU AI Compute Cluster",
            "zone": "Computer Science Block",
            "access": "Research & High Achievers",
            "description": "High-performance cluster for PySpark and neural network model training."
        },
        {
            "id": "RES02",
            "name": "Silent Study Pods & IEEE Digital Explorer",
            "zone": "Main Library",
            "access": "Open to all students",
            "description": "Quiet focus stations with direct full-text access to engineering journals."
        },
        {
            "id": "RES03",
            "name": "IoT Hardware Sandbox & FabLab",
            "zone": "Labs",
            "access": "Open with Lab Instructor",
            "description": "Raspberry Pi, Arduino, ESP32, and environmental sensor testing bench."
        }
    ]
}

def recommend(student_profile):
    """
    student_profile format:
    {
        "major": "Computer Science",
        "predicted_gpa": 7.8,
        "risk_level": "On Track",  # High Risk / Moderate Risk / On Track / High Achiever
        "interests": ["big data", "ai"]
    }
    """
    major = student_profile.get("major", "Computer Science")
    gpa = float(student_profile.get("predicted_gpa", 7.0))
    risk = student_profile.get("risk_level", "On Track")
    interests = [i.lower() for i in student_profile.get("interests", ["ai", "big data"])]

    # 1. Recommend Courses
    course_scores = []
    for c in CATALOG["courses"]:
        score = 0
        if c["department"] == major or c["department"] == "Management / IT":
            score += 3
        # Match interests
        for tag in c["tags"]:
            if any(interest in tag for interest in interests):
                score += 2
        # Risk specific
        if risk in ["High Risk", "Moderate Risk"] and "remedial" in c["tags"]:
            score += 5
        elif risk == "High Achiever" and c["level"] == "Advanced":
            score += 4
        
        # Check prerequisites
        if gpa >= c["target_gpa_min"]:
            course_scores.append((score, c))

    course_scores.sort(key=lambda x: x[0], reverse=True)
    rec_courses = [item[1] for item in course_scores[:3]]

    # 2. Recommend Events
    event_scores = []
    for e in CATALOG["events"]:
        score = 0
        if major in e["recommended_for"]:
            score += 3
        if risk in e["recommended_for"]:
            score += 6
        for tag in e["tags"]:
            if any(interest in tag for interest in interests):
                score += 2
        event_scores.append((score, e))

    event_scores.sort(key=lambda x: x[0], reverse=True)
    rec_events = [item[1] for item in event_scores[:3]]

    # 3. Recommend Resources
    rec_resources = CATALOG["resources"]

    return {
        "student_profile": student_profile,
        "recommended_courses": rec_courses,
        "recommended_events": rec_events,
        "recommended_resources": rec_resources
    }

if __name__ == "__main__":
    test_student = {
        "major": "Computer Science",
        "predicted_gpa": 8.4,
        "risk_level": "High Achiever",
        "interests": ["ai", "big data", "spark"]
    }
    output = recommend(test_student)
    print("Personalized Recommendations Sample:")
    print(json.dumps(output, indent=2))
