# =====================================
# RANDOM FOREST — ACADEMIC STRESS UI
# =====================================

import pandas as pd
import numpy as np
import gradio as gr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("academic Stress level - maintainance 1 2.csv")

# Drop timestamp if exists
if 'Timestamp' in df.columns:
    df = df.drop(columns=['Timestamp'])


# -------------------------------
# Encode Categorical Columns
# -------------------------------
encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le


# -------------------------------
# Target Column
# -------------------------------
target = "Rate your academic stress index "

X = df.drop(columns=[target])
y = df[target]


# -------------------------------
# Train Model
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model Accuracy:", acc)


# -------------------------------
# Prediction Function
# -------------------------------
def predict_stress(stage, peer, home, env, coping, habit, competition):

    # Encode categorical inputs
    stage_val = encoders['Your Academic Stage'].transform([stage])[0]
    env_val = encoders['Study Environment'].transform([env])[0]
    coping_val = encoders['What coping strategy you use as a student?'].transform([coping])[0]
    habit_val = encoders['Do you have any bad habits like smoking, drinking on a daily basis?'].transform([habit])[0]

    data = np.array([[
        stage_val,
        peer,
        home,
        env_val,
        coping_val,
        habit_val,
        competition
    ]])

    pred = model.predict(data)[0]

    return f"Predicted Stress Index: {pred}\nModel Accuracy: {acc:.2f}"


# -------------------------------
# Gradio Interface
# -------------------------------
ui = gr.Interface(
    fn=predict_stress,
    inputs=[
        gr.Dropdown(["undergraduate", "postgraduate"], label="Academic Stage"),
        gr.Number(label="Peer Pressure (1-5)"),
        gr.Number(label="Academic Pressure From Home (1-5)"),
        gr.Dropdown(["Peaceful", "Noisy"], label="Study Environment"),
        gr.Dropdown(
            [
                "Analyze the situation and handle it with intelligence",
                "Social support (friends, family)",
                "Others"
            ],
            label="Coping Strategy"
        ),
        gr.Dropdown(["No", "Yes"], label="Bad Habits"),
        gr.Number(label="Academic Competition (1-5)")
    ],
    outputs="text",
    title="Academic Stress Prediction (Random Forest)",
)

ui.launch()

Kaggle link-https://www.kaggle.com/datasets/ayeshaimran1619/student-academic-stress-level
