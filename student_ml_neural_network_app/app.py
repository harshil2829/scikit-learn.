from flask import Flask, render_template, request
import sqlite3
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        total INTEGER,
        percentage REAL
    )
    """)
    conn.commit()
    conn.close()

init_db()

def train_model():
    conn = sqlite3.connect("students.db")
    cursor = conn.cursor()
    cursor.execute("SELECT total, percentage FROM students")
    data = cursor.fetchall()
    conn.close()

    if len(data) < 3:
        return None, None

    X = np.array([row[0] for row in data]).reshape(-1, 1)
    y = np.array([row[1] for row in data])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPRegressor(
        hidden_layer_sizes=(10, 5),
        activation='relu',
        max_iter=5000,
        random_state=42
    )

    model.fit(X_scaled, y)

    return model, scaler

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = None

    if request.method == "POST":
        name = request.form["name"]
        total = int(request.form["total"])
        percentage = total / 5

        conn = sqlite3.connect("students.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO students (name, total, percentage) VALUES (?, ?, ?)",
            (name, total, percentage)
        )
        conn.commit()
        conn.close()

        model, scaler = train_model()

        if model:
            total_scaled = scaler.transform([[total]])
            predicted = model.predict(total_scaled)
            prediction = round(predicted[0], 2)
        else:
            message = "Not enough data to train Neural Network (Need at least 3 students)"

    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
