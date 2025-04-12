import pandas as pd
from flask import Flask, request, jsonify
from sklearn.cluster import KMeans

# 1️⃣ CSV Dosyasını Oku ve Veriyi İşle
df = pd.read_csv("study_sessions_detailed.csv")
df['date'] = pd.to_datetime(df['date'])

# Kullanıcı bazında ortalama çalışma istatistiklerini hesapla
user_stats = df.groupby("user_id").agg(
    avg_session_duration=("session_duration", "mean"),
    avg_break_duration=("break_duration", "mean"),
    avg_distractions=("distractions", "mean"),
    avg_productivity_score=("productivity_score", "mean"),
    avg_focus_level=("focus_level", "mean"),
    total_tasks_completed=("task_completed", "sum")
).reset_index()

# 2️⃣ Kullanıcıları K-Means ile Gruplandır (Opsiyonel: Kişiselleştirme için)
def perform_clustering():
    features = user_stats[[
        "avg_session_duration", "avg_break_duration", 
        "avg_distractions", "avg_productivity_score", 
        "avg_focus_level", "total_tasks_completed"
    ]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_stats["cluster"] = kmeans.fit_predict(features)

perform_clustering()

# 3️⃣ AI Destekli Kişiselleştirilmiş Çalışma Raporu Üretme
def generate_recommendation(user_id):
    user_data = user_stats[user_stats["user_id"] == user_id]
    if user_data.empty:
        return "No data available for this user."
    user_data = user_data.iloc[0]
    cluster = user_data["cluster"]
    recommendations = []

    # Cluster tabanlı öneriler
    if cluster == 0:
        recommendations.append("Your overall performance is low; consider reducing session duration and increasing breaks.")
    elif cluster == 1:
        recommendations.append("Your performance is average; try optimizing your environment to reduce distractions.")
    elif cluster == 2:
        recommendations.append("Excellent performance! Maintain your current routine and consistency.")

    # Spesifik öneriler
    if user_data["avg_session_duration"] > 60:
        recommendations.append("Your work sessions are long, try shorter Pomodoro cycles.")
    else:
        recommendations.append("Your work session duration is optimal.")

    if user_data["avg_break_duration"] < 5:
        recommendations.append("Consider extending your break duration to improve focus.")
    else:
        recommendations.append("Your break durations are well balanced.")

    if user_data["avg_distractions"] > 3:
        recommendations.append("Minimize distractions by working in a quieter environment.")
    else:
        recommendations.append("Your distraction levels are low.")

    if user_data["avg_productivity_score"] < 3:
        recommendations.append("Your productivity seems low; experiment with new techniques.")
    else:
        recommendations.append("Your productivity scores are good.")

    if user_data["avg_focus_level"] < 6:
        recommendations.append("Your focus level is below average; consider concentration improvement techniques.")
    else:
        recommendations.append("Your focus level is impressive!")

    return {
        "user_id": user_id,
        "study_report": recommendations,
        "cluster": int(cluster)
    }

# 4️⃣ Haftalık İlerleme Verisini Hazırlama
def get_weekly_progress(user_id):
    # Temel olarak tüm günler için varsayılan değerleri belirle
    base_progress = {"M": 0.0, "T": 0.0, "W": 0.0, "T2": 0.0, "F": 0.0, "S": 0.0, "S2": 0.0}
    
    user_data = df[df["user_id"] == user_id]
    if user_data.empty:
        return base_progress

    # Gün bazında ortalama çalışma süresi
    weekly_progress = user_data.groupby(user_data["date"].dt.strftime('%A'))["session_duration"].mean().to_dict()
    
    # Gün isimlerini kısalt: "Monday"->"M", "Tuesday"->"T", "Wednesday"->"W", "Thursday"->"T2", "Friday"->"F", "Saturday"->"S", "Sunday"->"S2"
    day_map = {
        "Monday": "M",
        "Tuesday": "T",
        "Wednesday": "W",
        "Thursday": "T2",
        "Friday": "F",
        "Saturday": "S",
        "Sunday": "S2"
    }
    
    for day, duration in weekly_progress.items():
        key = day_map.get(day, day)
        # Normalizasyon: Maksimum 90 dakikaya göre normalize et, böylece değer 0 ile 1 arasında olur.
        base_progress[key] = round(duration / 90, 2)
    
    return base_progress

# 5️⃣ Flask API Tanımla
app = Flask(__name__)

# /stats Endpoint: Günlük veriler, haftalık ilerleme ve çalışma raporu
@app.route('/stats', methods=['GET'])
def stats():
    try:
        user_id = int(request.args.get('user_id'))
    except:
        return jsonify({"error": "Please provide a valid user_id"}), 400

    report = generate_recommendation(user_id)
    weekly_progress = get_weekly_progress(user_id)
    
    total_sessions = df[df["user_id"] == user_id].shape[0]
    user_data = user_stats[user_stats["user_id"] == user_id].iloc[0]
    avg_focus = user_data["avg_focus_level"]
    focus_rate_calculated = f"{round(avg_focus * 10)}%"
    latest_data = df[df["user_id"] == user_id].sort_values("date", ascending=False).iloc[0]

    return jsonify({
        "user_id": user_id,
        "today_focus_time": f"{latest_data['session_duration']} min",
        "completed_pomodoros": str(total_sessions),
        "focus_rate": focus_rate_calculated,
        "weekly_progress": weekly_progress,
        "study_report": report["study_report"],
        "avg_session_duration": round(user_data["avg_session_duration"], 2),
        "avg_break_duration": round(user_data["avg_break_duration"], 2),
        "avg_productivity_score": round(user_data["avg_productivity_score"], 2),
        "avg_focus_level": round(user_data["avg_focus_level"], 2),
        "total_tasks_completed": int(user_data["total_tasks_completed"]),
    })

# /studyReport Endpoint: Sadece çalışma raporu
@app.route('/studyReport', methods=['GET'])
def study_report_endpoint():
    try:
        user_id = int(request.args.get('user_id'))
    except:
        return jsonify({"error": "Please provide a valid user_id"}), 400

    report = generate_recommendation(user_id)
    return jsonify(report)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
