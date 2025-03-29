import streamlit as st
import joblib
import pandas as pd

# Load trained model
rf = joblib.load(r"C:\Users\laava\Desktop\sem 6\AOML\project\random_forest_model.pkl")

# MBTI Personality Type Mapping
personality_map = {
    0: "ENFJ",
    1: "ENFP",
    2: "ENTJ",
    3: "ENTP",
    4: "ESFJ",
    5: "ESFP",
    6: "ESTJ",
    7: "ESTP",
    8: "INFJ",
    9: "INFP",
    10: "INTJ",
    11: "INTP",
    12: "ISFJ",
    13: "ISFP",
    14: "ISTJ",
    15: "ISTP"
}

# List of selected feature names (Now 25 Questions)
selected_features = [
    "You are prone to worrying that things will take a turn for the worse.",
    "You often end up doing things at the last possible moment.",
    "At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know",
    "You enjoy going to art museums.",
    "Your mood can change very quickly.",
    "You find it easy to empathize with a person whose experiences are very different from yours.",
    "You are not too interested in discussing various interpretations and analyses of creative works.",
    "Your happiness comes more from helping others accomplish things than your own accomplishments.",
    "You often feel overwhelmed.",
    "You like to have a to-do list for each day.",
    "You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.",
    "You are more inclined to follow your head than your heart.",
    "You have always been fascinated by the question of what, if anything, happens after death.",
    "You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine.",
    "You enjoy participating in group activities.",
    "You rarely worry about whether you make a good impression on people you meet.",
    "Seeing other people cry can easily make you feel like you want to cry too",
    "You like books and movies that make you come up with your own interpretation of the ending.",
    "You struggle with deadlines.",
    "You often make a backup plan for a backup plan.",
    "After a long and exhausting week, a lively social event is just what you need.",
    "You avoid making phone calls.",
    "You often have a hard time understanding other peoples feelings.",
    "You would pass along a good opportunity if you thought someone else needed it more.",
    "You usually stay calm, even under a lot of pressure"
]

# Streamlit UI
st.title("MBTI Personality Prediction")

# Get user inputs
user_input = {}
for feature in selected_features:
    user_input[feature] = st.slider(feature, -3, 3, 0)  # Default value = 0

# Predict button
if st.button("Predict"):
    # Convert input to DataFrame
    df = pd.DataFrame([user_input])

    # Make prediction
    predicted_label = rf.predict(df)[0]
    
    # Convert to MBTI personality type
    predicted_personality = personality_map.get(predicted_label, "Unknown Personality")

    # Display result
    st.subheader(f"Predicted MBTI Personality Type: {predicted_personality}")