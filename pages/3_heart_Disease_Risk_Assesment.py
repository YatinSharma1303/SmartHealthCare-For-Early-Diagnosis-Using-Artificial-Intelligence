import base64
import datetime
import io
import json
import pickle as pkl
import textwrap
import warnings
from html import escape
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from theme_config import init_theme, get_theme_styles, render_theme_toggle

warnings.simplefilter(action="ignore", category=UserWarning)

# =============================================================================
# CONSTANTS (unchanged from original)
# =============================================================================
MODEL_PATH = "models/third_feature_models/best_model.pkl"
ENCODER_PATH = "models/third_feature_models/cbe_encoder.pkl"
ICON_PATH = "utils/heart_disease.jpg"
SIDEBAR_IMAGE_PATH = "utils/ph5.png"

PROJECT_TITLE = "SmartHealthCare For Early Diagnosis Using Artificial Intelligence"
CREATOR_NAME = "Yatin Sharma"
APP_VERSION = "2.0 Material Expressive"

FEATURE_NAME_MAPPING = {
    "ever_diagnosed_with_heart_attack": "Heart Attack",
    "general_health": "General Health",
    "ever_diagnosed_with_a_stroke": "Stroke",
    "ever_told_you_have_kidney_disease": "Kidney Disease",
    "ever_told_you_had_diabetes": "Diabetes",
    "physical_health_status": "Physical Health",
    "ever_told_you_had_a_depressive_disorder": "Depression",
    "sleep_category": "Sleep",
    "age_category": "Age",
    "length_of_time_since_last_routine_checkup": "Checkup Time",
    "BMI": "BMI",
    "smoking_status": "Smoking",
    "exercise_status_in_past_30_Days": "Exercise",
    "binge_drinking_status": "Binge Drinking",
    "drinks_category": "Alcohol",
    "could_not_afford_to_see_doctor": "Doctor Access",
    "health_care_provider": "Healthcare Provider",
    "asthma_Status": "Asthma",
    "difficulty_walking_or_climbing_stairs": "Mobility",
    "mental_health_status": "Mental Health",
}

AGE_RISK_CATEGORIES = {
    "Age_55_to_59",
    "Age_60_to_64",
    "Age_65_to_69",
    "Age_70_to_74",
    "Age_75_to_79",
    "Age_80_or_older",
}

RECOMMENDATION_RULES = [
    {"feature": "ever_diagnosed_with_heart_attack",
     "condition": lambda x: x.get("ever_diagnosed_with_heart_attack") == "yes",
     "message": "- History of heart attack contributed {importance:.2f}% to your risk. Regularly visit your cardiologist, take prescribed medicines, and seek medical help for new or worsening symptoms."},
    {"feature": "ever_diagnosed_with_a_stroke",
     "condition": lambda x: x.get("ever_diagnosed_with_a_stroke") == "yes",
     "message": "- History of stroke contributed {importance:.2f}% to your risk. Follow your doctor's advice, take medicines consistently, and continue approved physical therapy or activity."},
    {"feature": "age_category",
     "condition": lambda x: x.get("age_category") in AGE_RISK_CATEGORIES,
     "message": "- Age contributed {importance:.2f}% to your risk. You cannot change your age, but regular checkups, activity, healthy eating, and avoiding smoking can lower risk."},
    {"feature": "general_health",
     "condition": lambda x: x.get("general_health") in ["fair", "poor"],
     "message": "- General health contributed {importance:.2f}% to your risk. Focus on regular checkups, balanced food, physical activity, and managing existing conditions."},
    {"feature": "ever_told_you_have_kidney_disease",
     "condition": lambda x: x.get("ever_told_you_have_kidney_disease") == "yes",
     "message": "- Kidney disease contributed {importance:.2f}% to your risk. Monitor kidney function and follow your doctor's treatment and diet advice."},
    {"feature": "ever_told_you_had_diabetes",
     "condition": lambda x: x.get("ever_told_you_had_diabetes") == "yes",
     "message": "- Diabetes contributed {importance:.2f}% to your risk. Manage blood sugar with diet, exercise, monitoring, and medication as prescribed."},
    {"feature": "smoking_status",
     "condition": lambda x: x.get("smoking_status") != "never_smoked",
     "message": "- Smoking contributed {importance:.2f}% to your risk. Quitting smoking can strongly reduce heart disease risk."},
    {"feature": "exercise_status_in_past_30_Days",
     "condition": lambda x: x.get("exercise_status_in_past_30_Days") == "no",
     "message": "- Lack of exercise contributed {importance:.2f}% to your risk. Try to build regular physical activity into your week, starting gently if needed."},
    {"feature": "binge_drinking_status",
     "condition": lambda x: x.get("binge_drinking_status") == "yes",
     "message": "- Binge drinking contributed {importance:.2f}% to your risk. Reducing alcohol intake can help lower heart disease risk."},
    {"feature": "drinks_category",
     "condition": lambda x: x.get("drinks_category") in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"],
     "message": "- Alcohol consumption contributed {importance:.2f}% to your risk. Limiting alcohol can help protect your heart."},
    {"feature": "sleep_category",
     "condition": lambda x: x.get("sleep_category") in ["short_sleep_4_to_5_hours", "very_short_sleep_0_to_3_hours"],
     "message": "- Sleep contributed {importance:.2f}% to your risk. Aim for 7 to 9 hours of quality sleep when possible."},
    {"feature": "physical_health_status",
     "condition": lambda x: x.get("physical_health_status") in ["1_to_13_days_not_good", "14_plus_days_not_good"],
     "message": "- Physical health contributed {importance:.2f}% to your risk. Speak with a healthcare provider if poor physical health is frequent."},
    {"feature": "mental_health_status",
     "condition": lambda x: x.get("mental_health_status") in ["1_to_13_days_not_good", "14_plus_days_not_good"],
     "message": "- Mental health contributed {importance:.2f}% to your risk. Consider support from a mental health professional and stress-reducing habits."},
    {"feature": "asthma_Status",
     "condition": lambda x: x.get("asthma_Status") in ["current_asthma", "former_asthma"],
     "message": "- Asthma contributed {importance:.2f}% to your risk. Follow your asthma plan and use medicines as prescribed."},
    {"feature": "ever_told_you_had_a_depressive_disorder",
     "condition": lambda x: x.get("ever_told_you_had_a_depressive_disorder") == "yes",
     "message": "- Depression contributed {importance:.2f}% to your risk. Mental health support, sleep, movement, and routine care can help."},
    {"feature": "difficulty_walking_or_climbing_stairs",
     "condition": lambda x: x.get("difficulty_walking_or_climbing_stairs") == "yes",
     "message": "- Difficulty walking or climbing stairs contributed {importance:.2f}% to your risk. Ask a healthcare provider about safe ways to improve mobility."},
    {"feature": "length_of_time_since_last_routine_checkup",
     "condition": lambda x: x.get("length_of_time_since_last_routine_checkup") != "past_year",
     "message": "- Time since last routine checkup contributed {importance:.2f}% to your risk. Regular checkups help find and manage health issues early."},
    {"feature": "could_not_afford_to_see_doctor",
     "condition": lambda x: x.get("could_not_afford_to_see_doctor") == "yes",
     "message": "- Difficulty affording care contributed {importance:.2f}% to your risk. Look into community clinics, sliding-scale clinics, or insurance support."},
    {"feature": "health_care_provider",
     "condition": lambda x: x.get("health_care_provider") == "no",
     "message": "- Not having a primary healthcare provider contributed {importance:.2f}% to your risk. A regular provider can help prevent and manage health problems."},
    {"feature": "BMI",
     "condition": lambda x: x.get("BMI") in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"],
     "message": "- BMI contributed {importance:.2f}% to your risk. A balanced diet, regular activity, and professional guidance can help reduce risk."},
]

HEART_TIPS = [
    ("💧", "Hydration",         "Drink water before reaching for sugary drinks — hydration supports heart function and reduces unnecessary calorie intake."),
    ("🚶", "Post-Meal Walk",    "A brisk 10-minute walk after meals can lower blood pressure and blood sugar more effectively than a single long session."),
    ("🥗", "Leafy Greens",      "Add one extra serving of leafy greens to your largest meal — spinach, kale, and arugula are loaded with heart-protective nitrates."),
    ("🧘", "Deep Breathing",    "Take 5 minutes to breathe deeply — just 5 slow breaths per minute activates the parasympathetic system and reduces cardiac stress."),
    ("🌰", "Healthy Fats",      "A small handful of unsalted walnuts or almonds daily is linked with measurably lower LDL cholesterol and heart disease risk."),
    ("😴", "Better Sleep",      "Going to bed 30 minutes earlier can improve cardiovascular recovery — poor sleep raises blood pressure and inflammatory markers."),
    ("🐟", "Omega-3s",          "Try fatty fish (salmon, mackerel, sardines) twice a week — omega-3 fatty acids reduce triglycerides and prevent arrhythmias."),
    ("🧂", "Watch Sodium",      "Read sodium labels carefully — most people consume 2× the recommended 2,300 mg/day, silently raising blood pressure."),
    ("🍓", "Berry Power",       "Berries (blueberries, strawberries, raspberries) are rich in anthocyanins that support arterial flexibility and lower oxidative stress."),
    ("🚭", "No Smoking",        "Even 24 hours smoke-free begins to reduce cardiac risk — within 1 year your heart attack risk drops by half."),
    ("📵", "Screen Limits",     "Limit screen time before bed — blue light delays melatonin, disrupts sleep quality, and indirectly worsens cardiovascular recovery."),
    ("🌾", "Whole Grains",      "Swap refined grains for whole grains — oats, quinoa, and brown rice help manage cholesterol and keep blood sugar stable."),
    ("💪", "Strength Train",    "Two strength-training sessions per week strengthen the heart muscle, improve insulin sensitivity, and reduce resting heart rate."),
    ("🌞", "Outdoor Walk",      "A 20-minute outdoor walk combines sunlight, light cardio, and stress relief — three powerful heart-health benefits in one habit."),
    ("🫐", "Dark Chocolate",    "Dark chocolate (70%+ cocoa) in small amounts contains flavonoids that support blood vessel function and reduce inflammation."),
    ("🍵", "Green Tea",         "Green tea drinkers show lower rates of cardiovascular disease — 2–3 cups daily may help lower LDL and blood pressure."),
    ("🧄", "Garlic Benefits",   "Garlic contains allicin, shown in studies to modestly lower blood pressure and reduce arterial plaque formation."),
    ("🏊", "Swimming",          "Swimming is one of the best full-body cardiovascular workouts — it raises heart rate without stressing joints."),
    ("🫀", "Know Your Numbers", "Know your numbers: blood pressure below 120/80, LDL below 100 mg/dL, and resting heart rate 60–100 bpm are key targets."),
    ("🤝", "Social Health",     "Strong social connections are independently linked to lower heart disease risk — loneliness raises cortisol and inflammation."),
]

BMI_BANDS = [
    (0, 18.5, "underweight_bmi_less_than_18_5", "Underweight", "#3b82f6"),
    (18.5, 25, "normal_weight_bmi_18_5_to_24_9", "Normal", "#14b8a6"),
    (25, 30, "overweight_bmi_25_to_29_9", "Overweight", "#f59e0b"),
    (30, 100, "obese_bmi_30_or_more", "Obese", "#ef4444"),
]

HEART_HEALTH_FAQ = {
    "what is heart disease": {
        "keywords": ["heart disease", "cardiovascular disease", "heart condition", "heart problem", "coronary artery", "cad", "heart failure", "arrhythmia", "what is heart"],
        "answer": "Heart disease refers to several conditions affecting your heart's structure and function, including coronary artery disease, arrhythmias, and heart failure. Coronary artery disease (CAD) is the most common type, caused by plaque buildup in arteries restricting blood flow to the heart muscle.",
    },
    "what are symptoms of heart attack": {
        "keywords": ["heart attack", "heart attack symptom", "myocardial infarction", "mi symptom", "chest pain", "cardiac arrest", "sign of heart attack", "symptoms of heart"],
        "answer": "Common heart attack symptoms include chest pain or pressure, shortness of breath, pain radiating to arm/jaw/back, nausea, cold sweats, and lightheadedness. Women may experience subtler symptoms like fatigue and jaw pain. Call emergency services immediately if suspected.",
    },
    "how to lower blood pressure": {
        "keywords": ["lower blood pressure", "reduce blood pressure", "lower bp", "high blood pressure", "hypertension treatment", "reduce hypertension", "control blood pressure", "decrease blood pressure"],
        "answer": "Lower BP through: reducing sodium intake (<2300mg/day), regular aerobic exercise (30 min/day), maintaining healthy weight, limiting alcohol, quitting smoking, managing stress, and eating potassium-rich foods (bananas, sweet potatoes). Medications may also be needed — consult your doctor.",
    },
    "what is cholesterol": {
        "keywords": ["cholesterol", "ldl", "hdl", "bad cholesterol", "good cholesterol", "triglyceride", "lipid", "lipoprotein", "cholesterol level"],
        "answer": "Cholesterol is a waxy substance in your blood. LDL ('bad') cholesterol builds up in arteries; HDL ('good') cholesterol removes it. High LDL raises heart disease risk. Targets: LDL <100 mg/dL, HDL >60 mg/dL, Triglycerides <150 mg/dL. Diet, exercise, and medications can help manage levels.",
    },
    "how much exercise for heart health": {
        "keywords": ["exercise heart", "how much exercise", "workout heart", "physical activity heart", "exercise recommendation", "cardio recommendation", "how often exercise", "exercise per week"],
        "answer": "The American Heart Association recommends at least 150 minutes/week of moderate-intensity aerobic activity OR 75 minutes/week of vigorous activity, plus muscle-strengthening activities 2+ days/week. Even short 10-minute walks count and provide real cardiovascular benefits.",
    },
    "what foods are good for heart": {
        "keywords": ["heart healthy food", "food for heart", "diet heart", "heart diet", "what to eat heart", "good food heart", "nutrition heart", "eat for heart", "mediterranean diet"],
        "answer": "Heart-healthy foods: oily fish (salmon, mackerel, sardines), leafy greens (spinach, kale), whole grains, berries, avocados, nuts (walnuts, almonds), olive oil, legumes (beans, lentils), and dark chocolate (70%+ cocoa in moderation). The Mediterranean diet pattern is strongly evidence-based for heart protection.",
    },
    "what is blood pressure": {
        "keywords": ["blood pressure", "systolic", "diastolic", "mmhg", "bp reading", "blood pressure reading", "normal blood pressure", "what is bp", "measure blood pressure"],
        "answer": "Blood pressure is the force of blood against artery walls, measured as systolic/diastolic (e.g., 120/80 mmHg). Normal: below 120/80. Elevated: 120-129/<80. Stage 1 hypertension: 130-139/80-89. Stage 2: 140+/90+. Hypertensive crisis: 180+/120+. Regular monitoring is key.",
    },
    "how does smoking affect heart": {
        "keywords": ["smoking heart", "cigarette heart", "tobacco heart", "nicotine heart", "smoking affect", "smoking risk", "quit smoking heart", "smoke heart"],
        "answer": "Smoking damages blood vessel walls, reduces oxygen in blood, increases clotting risk, raises blood pressure, and accelerates atherosclerosis. Smokers have 2-4x higher risk of heart disease. Risk drops significantly within 1-2 years of quitting — at 5-15 years, risk approaches that of non-smokers.",
    },
    "what is diabetes risk for heart": {
        "keywords": ["diabetes heart", "diabetic heart", "blood sugar heart", "diabetes cardiovascular", "diabetes risk heart", "insulin heart", "diabetes affect heart"],
        "answer": "Diabetics have 2-4x higher cardiovascular risk. High blood sugar damages blood vessels and nerves controlling the heart. Managing HbA1c below 7%, controlling blood pressure, and maintaining healthy LDL together is critical for diabetics to significantly reduce heart disease risk.",
    },
    "what is bmi": {
        "keywords": ["bmi", "body mass index", "what is bmi", "bmi category", "bmi calculate", "overweight", "obese", "obesity heart", "bmi heart"],
        "answer": "BMI (Body Mass Index) = weight(kg) / height(m)². Categories: Underweight <18.5, Normal 18.5-24.9, Overweight 25-29.9, Obese ≥30. Obesity significantly raises heart disease, diabetes, and hypertension risk. Even a 5-10% weight loss in overweight individuals meaningfully reduces cardiovascular risk.",
    },
    "how does sleep affect heart": {
        "keywords": ["sleep heart", "sleep cardiovascular", "insomnia heart", "poor sleep heart", "sleep apnea", "how much sleep", "sleep hours heart", "sleep risk"],
        "answer": "Poor sleep (< 6 hours or >9 hours) is linked with hypertension, obesity, diabetes, and heart disease. Sleep apnea especially strains the heart by causing repeated drops in oxygen. Aim for 7-9 hours of quality sleep and maintain a consistent sleep schedule even on weekends.",
    },
    "what is stress and heart disease": {
        "keywords": ["stress heart", "anxiety heart", "mental health heart", "stress cardiovascular", "depression heart", "cortisol heart", "stress blood pressure", "stress affect heart"],
        "answer": "Chronic stress raises cortisol levels, increasing blood pressure and inflammation. It also promotes unhealthy behaviors (smoking, poor diet, inactivity). Stress management through mindfulness, exercise, and social connection directly benefits heart health. Depression also independently raises heart disease risk.",
    },
    "what is stroke": {
        "keywords": ["stroke", "brain attack", "what is stroke", "stroke symptom", "stroke sign", "stroke risk", "cerebrovascular", "tia", "transient ischemic"],
        "answer": "A stroke occurs when blood supply to part of the brain is cut off — either by a clot (ischemic, 87% of cases) or a burst blood vessel (hemorrhagic). Symptoms: sudden face drooping, arm weakness, speech difficulty, severe headache, vision loss. Act FAST — every minute matters. Call emergency services immediately.",
    },
    "what is kidney disease and heart": {
        "keywords": ["kidney heart", "kidney disease heart", "renal cardiovascular", "kidney cardiovascular", "ckd heart", "chronic kidney heart"],
        "answer": "Kidney disease and heart disease are closely linked. Damaged kidneys can't regulate blood pressure and fluid properly, straining the heart. Conversely, heart failure reduces blood flow to the kidneys. People with chronic kidney disease (CKD) have up to 10-20x higher cardiovascular risk. Managing both conditions together is essential.",
    },
    "what is atrial fibrillation": {
        "keywords": ["atrial fibrillation", "afib", "irregular heartbeat", "arrhythmia", "heart rhythm", "palpitation", "heart flutter", "skipped beat"],
        "answer": "Atrial fibrillation (AFib) is an irregular, often rapid heart rhythm where the upper chambers quiver instead of beating properly. AFib raises stroke risk 5x because blood can pool and clot in the heart. Symptoms include palpitations, shortness of breath, fatigue. It is manageable with medication, cardioversion, or ablation.",
    },
    "how does alcohol affect heart": {
        "keywords": ["alcohol heart", "drinking heart", "wine heart", "beer heart", "alcohol cardiovascular", "binge drinking heart", "drink affect heart", "alcohol risk heart"],
        "answer": "Moderate alcohol may have slight cardiovascular benefit for some people, but heavy or binge drinking is harmful: it raises blood pressure, causes irregular heartbeats (holiday heart syndrome), weakens heart muscle (cardiomyopathy), and increases stroke risk. The AHA advises if you don't drink, don't start for heart health.",
    },
    "what is heart failure": {
        "keywords": ["heart failure", "congestive heart failure", "chf", "weak heart", "heart pump", "ejection fraction", "heart failure symptom"],
        "answer": "Heart failure means the heart can't pump blood efficiently enough to meet the body's needs. Symptoms include shortness of breath (especially lying down), leg swelling, rapid weight gain, and fatigue. It is a chronic condition managed with medications (ACE inhibitors, beta-blockers, diuretics), lifestyle changes, and sometimes devices.",
    },
    "what is the best diet for heart": {
        "keywords": ["best diet heart", "dash diet", "mediterranean", "heart diet plan", "diet plan heart", "eating plan heart", "nutrition plan heart"],
        "answer": "The two best-evidence diets for heart health are the Mediterranean Diet (olive oil, fish, vegetables, nuts, legumes, whole grains) and the DASH Diet (Dietary Approaches to Stop Hypertension — low sodium, rich in potassium, calcium, magnesium). Both significantly reduce cardiovascular events in research studies.",
    },
    "what is ecg or ekg": {
        "keywords": ["ecg", "ekg", "electrocardiogram", "heart test", "cardiac test", "heart scan", "echocardiogram", "stress test"],
        "answer": "An ECG (electrocardiogram) records the electrical activity of your heart via electrodes on your skin. It detects arrhythmias, heart attacks (past or current), and structural issues. An echocardiogram uses ultrasound to view heart structure and pumping function. A stress test monitors your heart during exercise to detect coronary artery disease.",
    },
    "how to prevent heart disease": {
        "keywords": ["prevent heart disease", "prevent heart attack", "reduce heart risk", "heart disease prevention", "avoid heart disease", "heart health tips", "protect heart"],
        "answer": "Key prevention strategies: (1) Don't smoke, (2) Exercise 150+ min/week, (3) Eat a heart-healthy diet (Mediterranean/DASH), (4) Maintain healthy weight (BMI 18.5-24.9), (5) Control blood pressure (<120/80), (6) Manage cholesterol (LDL <100), (7) Control blood sugar if diabetic, (8) Limit alcohol, (9) Manage stress, (10) Get regular checkups.",
    },
}

RISK_RADAR_FACTORS = {
    "Smoking": {"key": "smoking_status", "bad": ["current_smoker_every_day", "current_smoker_some_days"], "warn": ["former_smoker"]},
    "Exercise": {"key": "exercise_status_in_past_30_Days", "bad": ["no"], "warn": []},
    "Sleep": {"key": "sleep_category", "bad": ["very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours"], "warn": ["very_long_sleep_11_or_more_hours"]},
    "BMI": {"key": "BMI", "bad": ["obese_bmi_30_or_more"], "warn": ["overweight_bmi_25_to_29_9", "underweight_bmi_less_than_18_5"]},
    "Alcohol": {"key": "binge_drinking_status", "bad": ["yes"], "warn": []},
    "Checkup": {"key": "length_of_time_since_last_routine_checkup", "bad": ["5+_years_ago", "never"], "warn": ["past_5_years", "past_2_years"]},
}


def load_page_icon(path):
    try:
        return Image.open(path)
    except Exception:
        return None


icon = load_page_icon(ICON_PATH)

st.set_page_config(
    layout="wide",
    page_title=PROJECT_TITLE,
    page_icon=icon,
    menu_items={
        "About": f"{PROJECT_TITLE} — {APP_VERSION}. Made with ❤️ by {CREATOR_NAME}.",
        "Get help": "https://www.who.int/health-topics/cardiovascular-diseases",
    },
)

init_theme()
st.markdown(get_theme_styles(), unsafe_allow_html=True)
render_theme_toggle()


# =============================================================================
# CACHED HELPERS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_assets():
    with open(MODEL_PATH, "rb") as model_file:
        loaded_model = pkl.load(model_file)
    with open(ENCODER_PATH, "rb") as encoder_file:
        loaded_encoder = pkl.load(encoder_file)
    return loaded_model, loaded_encoder


def pretty_value(value):
    if value is None:
        return "N/A"
    return str(value).replace("_", " ").title()


def get_feature_names(encoder, input_df, transformed):
    transformed_df = pd.DataFrame(transformed)
    try:
        names = list(encoder.get_feature_names_out())
        if len(names) == transformed_df.shape[1]:
            return names
    except Exception:
        pass
    if isinstance(transformed, pd.DataFrame):
        return list(transformed.columns)
    if transformed_df.shape[1] == len(input_df.columns):
        return list(input_df.columns)
    return [f"feature_{i}" for i in range(transformed_df.shape[1])]


def encode_input(input_data, encoder):
    input_df = pd.DataFrame([input_data])
    transformed = encoder.transform(input_df)
    feature_names = get_feature_names(encoder, input_df, transformed)
    return pd.DataFrame(transformed, columns=feature_names)


def predict_heart_disease_risk(input_data, model, encoder):
    input_encoded = encode_input(input_data, encoder)
    prediction = model.predict_proba(input_encoded)[:, 1][0] * 100
    return float(prediction), input_encoded


def get_tree_model(model):
    try:
        estimator = model.estimators_[0]
        if hasattr(estimator, "steps"):
            return estimator.steps[-1][1]
        if hasattr(estimator, "named_steps"):
            return list(estimator.named_steps.values())[-1]
        return estimator
    except Exception:
        return model


@st.cache_resource(show_spinner=False)
def get_cached_explainer(_tree_model):
    return shap.TreeExplainer(_tree_model)


def normalize_importances(values, feature_count):
    if feature_count == 0:
        return np.array([])
    values = np.asarray(values, dtype=float).reshape(-1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if values.size > feature_count:
        values = values[:feature_count]
    elif values.size < feature_count:
        values = np.pad(values, (0, feature_count - values.size))
    total = values.sum()
    if total <= 0:
        return np.round(np.ones(feature_count) * (100 / feature_count), 2)
    return np.round((values / total) * 100, 2)


def build_feature_importance_df(model, input_encoded):
    feature_count = len(input_encoded.columns)
    tree_model = get_tree_model(model)
    try:
        explainer = get_cached_explainer(tree_model)
        shap_values = explainer.shap_values(input_encoded)
        if isinstance(shap_values, list):
            shap_matrix = np.asarray(shap_values[-1])
        else:
            shap_matrix = np.asarray(shap_values)
        if shap_matrix.ndim == 3:
            shap_matrix = shap_matrix[:, :, -1]
        if shap_matrix.ndim == 1:
            shap_matrix = shap_matrix.reshape(1, -1)
        raw_importances = np.abs(shap_matrix).mean(axis=0)
    except Exception:
        raw_importances = getattr(tree_model, "feature_importances_", np.ones(feature_count))
    importances = normalize_importances(raw_importances, feature_count)
    return pd.DataFrame({
        "Feature": list(input_encoded.columns),
        "Importance": importances,
    }).sort_values(by="Importance", ascending=False)


def base_feature_name(encoded_feature):
    encoded_feature = str(encoded_feature)
    for feature in sorted(FEATURE_NAME_MAPPING.keys(), key=len, reverse=True):
        if encoded_feature == feature or encoded_feature.startswith(f"{feature}_"):
            return feature
    return encoded_feature


def feature_importance_value(feature_importance_df, feature):
    feature_names = feature_importance_df["Feature"].astype(str)
    mask = feature_names.eq(feature) | feature_names.str.startswith(f"{feature}_")
    if not mask.any():
        return 0.0
    return float(feature_importance_df.loc[mask, "Importance"].sum())


def get_recommendation_intro(risk):
    if risk > 70:
        return "Your risk of heart disease is very high. Here are recommendations to reduce your risk:"
    if risk > 40:
        return "Your risk of heart disease is high. Here are recommendations to reduce your risk:"
    if risk > 25:
        return "Your risk of heart disease is moderate. Here are recommendations to reduce your risk:"
    return "Your risk of heart disease is low. Keep maintaining a healthy lifestyle."


def get_risk_level(risk):
    if risk > 70: return "Very High"
    if risk > 40: return "High"
    if risk > 25: return "Moderate"
    return "Low"


def get_risk_class(risk):
    if risk > 70: return "risk-very-high"
    if risk > 40: return "risk-high"
    if risk > 25: return "risk-med"
    return "risk-low"


def get_risk_color(risk):
    if risk > 70: return "#7f1d1d"
    if risk > 40: return "#dc2626"
    if risk > 25: return "#f59e0b"
    return "#14b8a6"


def build_recommendations(input_data, feature_importance_df):
    important_features = set()
    cumulative_importance = 0.0
    for _, row in feature_importance_df.iterrows():
        feature = base_feature_name(row["Feature"])
        if feature in FEATURE_NAME_MAPPING:
            important_features.add(feature)
        cumulative_importance += float(row["Importance"])
        if cumulative_importance >= 50:
            break
    for rule in RECOMMENDATION_RULES:
        if rule["condition"](input_data):
            important_features.add(rule["feature"])
    rec_map = {}
    final_features = []
    for rule in RECOMMENDATION_RULES:
        feature = rule["feature"]
        if feature in important_features and rule["condition"](input_data):
            importance = feature_importance_value(feature_importance_df, feature)
            rec_map[feature] = rule["message"].format(importance=importance)
            final_features.append(feature)
    sorted_features = sorted(
        final_features,
        key=lambda f: feature_importance_value(feature_importance_df, f),
        reverse=True,
    )
    ordered_rec_map = {feature: rec_map[feature] for feature in sorted_features}
    if not sorted_features:
        return ordered_rec_map, None
    feature_values = [feature_importance_value(feature_importance_df, feature) for feature in sorted_features]
    other_factors_importance = max(0.0, round(100 - sum(feature_values), 2))
    pie_df = pd.DataFrame({
        "Feature": [FEATURE_NAME_MAPPING[feature] for feature in sorted_features] + ["Other Factors"],
        "Importance": feature_values + [other_factors_importance],
    })
    return ordered_rec_map, pie_df


# =============================================================================
# HEALTH SCORE / WELLNESS METRICS
# =============================================================================
def compute_lifestyle_score(d):
    score = 100
    if d.get("smoking_status") == "current_smoker_every_day": score -= 22
    elif d.get("smoking_status") == "current_smoker_some_days": score -= 14
    elif d.get("smoking_status") == "former_smoker": score -= 5
    if d.get("exercise_status_in_past_30_Days") == "no": score -= 15
    if d.get("binge_drinking_status") == "yes": score -= 10
    if d.get("drinks_category") in ["high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"]: score -= 10
    if d.get("sleep_category") in ["very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours"]: score -= 10
    elif d.get("sleep_category") == "very_long_sleep_11_or_more_hours": score -= 4
    if d.get("BMI") == "obese_bmi_30_or_more": score -= 12
    elif d.get("BMI") == "overweight_bmi_25_to_29_9": score -= 6
    elif d.get("BMI") == "underweight_bmi_less_than_18_5": score -= 5
    if d.get("general_health") == "poor": score -= 12
    elif d.get("general_health") == "fair": score -= 6
    if d.get("length_of_time_since_last_routine_checkup") != "past_year": score -= 6
    return max(0, min(100, score))


def heart_age_estimate(d, risk):
    age_map = {
        "Age_18_to_24": 21, "Age_25_to_29": 27, "Age_30_to_34": 32, "Age_35_to_39": 37,
        "Age_40_to_44": 42, "Age_45_to_49": 47, "Age_50_to_54": 52, "Age_55_to_59": 57,
        "Age_60_to_64": 62, "Age_65_to_69": 67, "Age_70_to_74": 72, "Age_75_to_79": 77,
        "Age_80_or_older": 82,
    }
    base = age_map.get(d.get("age_category"), 40)
    delta = (risk - 15) * 0.18
    return int(round(base + delta))


def render_risk_gauge(risk):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risk,
        number={"suffix": "%", "font": {"size": 42, "color": get_risk_color(risk)}},
        delta={"reference": 25, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#14b8a6"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(148,163,184,0.6)"},
            "bar": {"color": get_risk_color(risk), "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "rgba(20,184,166,0.22)"},
                {"range": [25, 40], "color": "rgba(245,158,11,0.22)"},
                {"range": [40, 70], "color": "rgba(220,38,38,0.22)"},
                {"range": [70, 100], "color": "rgba(127,29,29,0.30)"},
            ],
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.75, "value": risk},
        },
    ))
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "var(--md-soft)"},
    )
    return fig


# =============================================================================
# PDF
# =============================================================================
def create_heart_health_pdf(input_data, risk, feature_to_recommendation):
    try:
        from reportlab.lib.colors import HexColor, white, black
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen.canvas import Canvas
    except Exception as error:
        raise ImportError("ReportLab is required. Install it with: pip install reportlab==4.4.10") from error

    # ══════════════════════════════════════════════════════════════════════════
    # MD3 DARK THEME — Design Tokens
    # ══════════════════════════════════════════════════════════════════════════
    # Surfaces
    BG               = HexColor("#0D1117")   # Page background — near-black
    SURFACE_1        = HexColor("#141B22")   # Card level 1
    SURFACE_2        = HexColor("#1C2630")   # Card level 2 / zebra
    SURFACE_3        = HexColor("#243040")   # Elevated / header areas

    # Primary — MD3 teal
    PRIMARY          = HexColor("#4DB6AC")   # md.sys.color.primary (dark)
    ON_PRIMARY       = HexColor("#003733")   # text on primary
    PRIMARY_CONT     = HexColor("#00504A")   # primary container
    ON_PRIMARY_CONT  = HexColor("#70F6EA")   # text in container

    # Secondary
    SECONDARY        = HexColor("#80CBC4")   # md.sys.color.secondary (dark)
    SEC_CONT         = HexColor("#1E3E3B")

    # Tertiary — indigo accent
    TERTIARY         = HexColor("#7986CB")
    TERT_CONT        = HexColor("#1A237E")

    # Error / risk
    ERROR            = HexColor("#FF6B6B")
    ERROR_CONT       = HexColor("#3B1010")
    ON_ERROR_CONT    = HexColor("#FFB4AB")

    # Warning / moderate
    WARN             = HexColor("#FFD54F")
    WARN_CONT        = HexColor("#3B2E00")
    ON_WARN_CONT     = HexColor("#FFE57F")

    # Success / low risk
    SUCCESS          = HexColor("#69F0AE")
    SUCCESS_CONT     = HexColor("#00391F")
    ON_SUCCESS_CONT  = HexColor("#A5F3C8")

    # Text
    ON_BG            = HexColor("#E0ECEB")   # Primary text on dark bg
    ON_SURF_MED      = HexColor("#8BA8A4")   # Secondary / muted text
    ON_SURF_LOW      = HexColor("#4A6460")   # Very muted / labels

    # Outlines
    OUTLINE          = HexColor("#2E4040")
    OUTLINE_VAR      = HexColor("#1E3030")

    buffer = io.BytesIO()
    c = Canvas(buffer, pagesize=letter)
    width, height = letter
    ML = 44          # margin left
    MR = width - 44  # margin right
    CW = MR - ML     # content width

    now_str  = datetime.datetime.now().strftime("%B %d, %Y")
    time_str = datetime.datetime.now().strftime("%H:%M")

    # ── Risk tier ────────────────────────────────────────────────────────────
    if risk < 20:
        R_FG, R_BG, R_ON, R_LABEL, R_DESC = (
            SUCCESS, SUCCESS_CONT, ON_SUCCESS_CONT,
            "LOW RISK",
            "Your cardiovascular profile currently falls within a lower-risk range. "
            "Sustaining your current lifestyle habits is strongly encouraged."
        )
    elif risk < 50:
        R_FG, R_BG, R_ON, R_LABEL, R_DESC = (
            WARN, WARN_CONT, ON_WARN_CONT,
            "MODERATE RISK",
            "Your assessment indicates a moderate cardiovascular risk profile. "
            "Targeted lifestyle modifications may significantly improve your long-term outlook."
        )
    else:
        R_FG, R_BG, R_ON, R_LABEL, R_DESC = (
            ERROR, ERROR_CONT, ON_ERROR_CONT,
            "ELEVATED RISK",
            "Your assessment indicates an elevated cardiovascular risk profile. "
            "Prompt consultation with a qualified healthcare provider is strongly advised."
        )

    lifestyle = compute_lifestyle_score(input_data)
    heart_age = heart_age_estimate(input_data, risk)

    # ══════════════════════════════════════════════════════════════════════════
    # PRIMITIVES
    # ══════════════════════════════════════════════════════════════════════════
    def fill_bg():
        """Paint the full page dark background."""
        c.setFillColor(BG)
        c.rect(0, 0, width, height, fill=1, stroke=0)

    def draw_heart(cx, cy, size=9, color=None):
        c.setFillColor(color or ERROR)
        r = size * 0.38
        c.circle(cx - r, cy + r * 0.60, r, fill=1, stroke=0)
        c.circle(cx + r, cy + r * 0.60, r, fill=1, stroke=0)
        p = c.beginPath()
        p.moveTo(cx - size * 0.76, cy + r * 0.35)
        p.lineTo(cx + size * 0.76, cy + r * 0.35)
        p.lineTo(cx, cy - size * 0.80)
        p.close()
        c.drawPath(p, fill=1, stroke=0)

    def txt(text, x, y, font="Helvetica", size=9, color=None, align="left"):
        c.setFillColor(color or ON_BG)
        c.setFont(font, size)
        if align == "center":
            c.drawCentredString(x, y, str(text))
        elif align == "right":
            c.drawRightString(x, y, str(text))
        else:
            c.drawString(x, y, str(text))

    def para(text, y, x=None, size=9, color=None, line_h=13.5, bold=False, indent=0):
        x = (x if x is not None else ML) + indent
        avail = MR - x - 4
        wrap_w = max(28, int(avail / (size * 0.510)))
        lines = textwrap.wrap(str(text), width=wrap_w) or [""]
        font = "Helvetica-Bold" if bold else "Helvetica"
        for ln in lines:
            y = ensure(y, 18)
            c.setFillColor(color or ON_BG)
            c.setFont(font, size)
            c.drawString(x, y, ln)
            y -= line_h
        return y

    # ── Page state ────────────────────────────────────────────────────────────
    page_num = [1]

    def draw_footer():
        """Dark footer strip on every page."""
        fh = 34
        c.setFillColor(SURFACE_1)
        c.rect(0, 0, width, fh, fill=1, stroke=0)
        c.setStrokeColor(PRIMARY_CONT)
        c.setLineWidth(0.5)
        c.line(ML, fh, MR, fh)

        # Left — "Made with ♥ by Yatin"
        made = "Made with"
        by   = f" by {CREATOR_NAME}"
        mw   = c.stringWidth(made, "Helvetica", 7.5)
        gap  = 12
        fx   = ML
        txt(made, fx, 12, size=7.5, color=ON_SURF_MED)
        draw_heart(fx + mw + gap / 2, 13.5, size=5, color=ERROR)
        c.setFillColor(ON_SURF_MED)
        c.setFont("Helvetica", 7.5)
        c.drawString(fx + mw + gap, 12, by)

        # Centre — date
        txt(f"{now_str}  ·  {time_str}", width / 2, 12, size=7.5,
            color=ON_SURF_LOW, align="center")

        # Right — page
        txt(f"Page {page_num[0]}", MR, 12, size=7.5,
            color=ON_SURF_MED, align="right")

    def new_page():
        draw_footer()
        c.showPage()
        page_num[0] += 1
        fill_bg()
        # Continuation header strip
        c.setFillColor(SURFACE_3)
        c.rect(0, height - 28, width, 28, fill=1, stroke=0)
        c.setFillColor(PRIMARY)
        c.rect(0, height - 3, width, 3, fill=1, stroke=0)
        txt("Cardiovascular Risk Assessment  —  Continued",
            ML, height - 19, font="Helvetica-Bold", size=8, color=ON_SURF_MED)
        txt(f"Page {page_num[0]}", MR, height - 19,
            size=8, color=ON_SURF_MED, align="right")
        return height - 46

    def ensure(y, needed=80):
        if y < needed + 44:
            return new_page()
        return y

    # ── Section header ────────────────────────────────────────────────────────
    def section_header(label, y, accent=None, icon=None):
        y = ensure(y, 55)
        accent = accent or PRIMARY
        h = 26
        # filled pill
        c.setFillColor(accent)
        c.roundRect(ML, y - 5, CW, h, 7, fill=1, stroke=0)
        # left accent bar
        c.setFillColor(ON_PRIMARY)
        prefix = f"{icon}  " if icon else ""
        c.setFont("Helvetica-Bold", 9.5)
        c.setFillColor(ON_PRIMARY)
        c.drawString(ML + 14, y + 6, f"{prefix}{label.upper()}")
        return y - h - 8

    # ── Card background ───────────────────────────────────────────────────────
    def card(y, content_h, color=None, radius=8):
        color = color or SURFACE_1
        c.setFillColor(color)
        c.roundRect(ML, y - content_h, CW, content_h + 6, radius, fill=1, stroke=0)

    # ── Key-value table row ───────────────────────────────────────────────────
    def kv_row(label, value, y, label_w=165, zebra=False):
        y = ensure(y, 20)
        rh = 17
        bg = SURFACE_2 if zebra else SURFACE_1
        c.setFillColor(bg)
        c.rect(ML, y - 4, CW, rh, fill=1, stroke=0)
        # subtle left accent line
        c.setFillColor(PRIMARY_CONT)
        c.rect(ML, y - 4, 2, rh, fill=1, stroke=0)
        # label
        c.setFillColor(ON_SURF_MED)
        c.setFont("Helvetica", 8.5)
        c.drawString(ML + 10, y + 2, label)
        # value
        c.setFillColor(ON_BG)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(ML + label_w, y + 2, str(value))
        return y - rh

    # ── Bullet item ───────────────────────────────────────────────────────────
    def bullet_item(text, y, dot=None, bg=None):
        dot = dot or PRIMARY
        y = ensure(y, 26)
        avail = MR - (ML + 22) - 4
        wrap_w = max(28, int(avail / (9 * 0.510)))
        lines = textwrap.wrap(str(text), width=wrap_w) or [""]
        item_h = len(lines) * 13 + 8
        if bg:
            c.setFillColor(bg)
            c.roundRect(ML, y - item_h + 8, CW, item_h, 6, fill=1, stroke=0)
        # dot
        c.setFillColor(dot)
        c.circle(ML + 11, y + 3, 3.2, fill=1, stroke=0)
        for i, ln in enumerate(lines):
            y = ensure(y, 16)
            c.setFillColor(ON_BG)
            c.setFont("Helvetica", 9)
            c.drawString(ML + 22, y, ln)
            y -= 13
        return y - 3

    # ── Metric chip (small stat badge) ───────────────────────────────────────
    def metric_chip(label, value, x, y, w=120, h=44, accent=None):
        accent = accent or PRIMARY
        c.setFillColor(SURFACE_2)
        c.roundRect(x, y, w, h, 8, fill=1, stroke=0)
        c.setStrokeColor(accent)
        c.setLineWidth(1)
        c.roundRect(x, y, w, h, 8, fill=0, stroke=1)
        c.setFillColor(accent)
        c.setFont("Helvetica-Bold", 13)
        c.drawCentredString(x + w / 2, y + h - 20, str(value))
        c.setFillColor(ON_SURF_MED)
        c.setFont("Helvetica", 7.5)
        c.drawCentredString(x + w / 2, y + 7, label)

    # ── Callout / alert box ───────────────────────────────────────────────────
    def callout_box(text, y, bg, border, fg, left_bar=None):
        avail  = CW - 28
        wrap_w = max(28, int(avail / (8.5 * 0.510)))
        lines  = textwrap.wrap(str(text), width=wrap_w) or [""]
        box_h  = len(lines) * 12.5 + 18
        y      = ensure(y, box_h + 14)
        boty   = y - box_h + 10
        c.setFillColor(bg)
        c.roundRect(ML, boty, CW, box_h, 8, fill=1, stroke=0)
        c.setStrokeColor(border)
        c.setLineWidth(1)
        c.roundRect(ML, boty, CW, box_h, 8, fill=0, stroke=1)
        if left_bar:
            c.setFillColor(left_bar)
            c.roundRect(ML, boty, 4, box_h, 4, fill=1, stroke=0)
        ty = y - 4
        for ln in lines:
            c.setFillColor(fg)
            c.setFont("Helvetica", 8.5)
            c.drawString(ML + 14, ty, ln)
            ty -= 12.5
        return boty - 10

    # ══════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ══════════════════════════════════════════════════════════════════════════
    fill_bg()

    # ── Hero header band ─────────────────────────────────────────────────────
    HDR_H = 132
    c.setFillColor(SURFACE_3)
    c.rect(0, height - HDR_H, width, HDR_H, fill=1, stroke=0)

    # top edge accent stripe
    c.setFillColor(PRIMARY)
    c.rect(0, height - 4, width, 4, fill=1, stroke=0)

    # Heart icon
    draw_heart(ML + 18, height - 46, size=16, color=ERROR)

    # Report type label
    c.setFillColor(ON_SURF_LOW)
    c.setFont("Helvetica", 8)
    c.drawString(ML + 44, height - 26, "CARDIOVASCULAR HEALTH REPORT")

    # Main title
    c.setFillColor(ON_BG)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(ML + 44, height - 44, "Heart Disease Risk Assessment")

    # Subtitle / app name
    c.setFillColor(ON_SURF_MED)
    c.setFont("Helvetica", 8.5)
    c.drawString(ML + 44, height - 58, PROJECT_TITLE)

    # ── Risk badge (right side of header) ────────────────────────────────────
    BW, BH = 138, 60
    BX = MR - BW
    BY = height - HDR_H + (HDR_H - BH) / 2
    c.setFillColor(R_BG)
    c.roundRect(BX, BY, BW, BH, 10, fill=1, stroke=0)
    c.setStrokeColor(R_FG)
    c.setLineWidth(1.5)
    c.roundRect(BX, BY, BW, BH, 10, fill=0, stroke=1)
    c.setFillColor(R_FG)
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(BX + BW / 2, BY + BH - 26, f"{risk:.1f}%")
    c.setFillColor(R_ON)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(BX + BW / 2, BY + 9, R_LABEL)

    # ── Meta bar below header ─────────────────────────────────────────────────
    META_H = 22
    meta_y = height - HDR_H
    c.setFillColor(SURFACE_1)
    c.rect(0, meta_y - META_H, width, META_H, fill=1, stroke=0)
    c.setStrokeColor(OUTLINE)
    c.setLineWidth(0.4)
    c.line(0, meta_y - META_H, width, meta_y - META_H)

    c.setFillColor(ON_SURF_LOW)
    c.setFont("Helvetica", 7.5)
    c.drawString(ML, meta_y - 14, f"Generated  {now_str}  at  {time_str}")
    c.drawRightString(MR, meta_y - 14, "CONFIDENTIAL  —  FOR PERSONAL USE ONLY")

    y = meta_y - META_H - 18

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — PATIENT PROFILE
    # ══════════════════════════════════════════════════════════════════════════
    y = section_header("Patient Profile", y, accent=PRIMARY)
    y -= 2
    profile_rows = [
        ("Age Group",        pretty_value(input_data.get("age_category"))),
        ("Biological Sex",   pretty_value(input_data.get("gender"))),
        ("Ethnicity",        pretty_value(input_data.get("race"))),
    ]
    for i, (lbl, val) in enumerate(profile_rows):
        y = kv_row(lbl, val, y, zebra=(i % 2 == 0))
    y -= 16

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — CLINICAL HEALTH INDICATORS
    # ══════════════════════════════════════════════════════════════════════════
    y = section_header("Clinical Health Indicators", y, accent=SECONDARY)
    y -= 2

    bmi_raw = input_data.get("BMI", "N/A")
    try:
        bmi_val = float(bmi_raw)
        if bmi_val < 18.5:   bmi_note = "Underweight"
        elif bmi_val < 25:   bmi_note = "Normal range"
        elif bmi_val < 30:   bmi_note = "Overweight"
        else:                bmi_note = "Obese"
        bmi_display = f"{bmi_val:.1f}  ({bmi_note})"
    except (ValueError, TypeError):
        bmi_display = pretty_value(bmi_raw)

    health_rows = [
        ("Body Mass Index (BMI)",          bmi_display),
        ("Diabetes Diagnosis",             pretty_value(input_data.get("ever_told_you_had_diabetes"))),
        ("Prior Myocardial Infarction",    pretty_value(input_data.get("ever_diagnosed_with_heart_attack"))),
        ("Cerebrovascular Accident",       pretty_value(input_data.get("ever_diagnosed_with_a_stroke"))),
        ("Chronic Kidney Disease",         pretty_value(input_data.get("ever_told_you_have_kidney_disease"))),
        ("Tobacco Use",                    pretty_value(input_data.get("smoking_status"))),
        ("Physical Activity (Past 30 d)",  pretty_value(input_data.get("exercise_status_in_past_30_Days"))),
        ("Alcohol Consumption Pattern",    pretty_value(input_data.get("binge_drinking_status"))),
        ("Average Sleep Duration",         pretty_value(input_data.get("sleep_category"))),
    ]
    for i, (lbl, val) in enumerate(health_rows):
        y = kv_row(lbl, val, y, zebra=(i % 2 == 0))
    y -= 16

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 3 — RISK ASSESSMENT RESULTS
    # ══════════════════════════════════════════════════════════════════════════
    y = section_header("Risk Assessment Results", y, accent=R_FG)
    y -= 6

    # 4 metric chips in a row
    y = ensure(y, 68)
    chip_w = (CW - 18) / 4
    chips = [
        ("Risk Classification", get_risk_level(risk), R_FG),
        ("Predicted Risk Score", f"{risk:.1f}%",       R_FG),
        ("Lifestyle Score",      f"{lifestyle}/100",    PRIMARY),
        ("Est. Cardiac Age",     f"{heart_age} yrs",   TERTIARY),
    ]
    for i, (lbl, val, acc) in enumerate(chips):
        metric_chip(lbl, val, ML + i * (chip_w + 6), y - 44, w=chip_w, h=44, accent=acc)
    y -= 60

    # Risk progress bar
    y = ensure(y, 28)
    bar_full   = CW - 10
    bar_filled = bar_full * min(risk / 100, 1.0)
    # track
    c.setFillColor(SURFACE_2)
    c.roundRect(ML + 5, y, bar_full, 10, 5, fill=1, stroke=0)
    # zone ticks
    for pct, col in [(0.2, SUCCESS), (0.5, WARN), (1.0, ERROR)]:
        c.setFillColor(col)
        c.rect(ML + 5, y, bar_full * pct, 10, fill=1, stroke=0)
    # overlay dark fill right of value
    c.setFillColor(SURFACE_1)
    c.rect(ML + 5 + max(bar_filled, 8), y, bar_full - max(bar_filled, 8), 10, fill=1, stroke=0)
    # pointer line
    c.setStrokeColor(ON_BG)
    c.setLineWidth(1.5)
    px = ML + 5 + max(bar_filled, 8)
    c.line(px, y - 1, px, y + 11)
    # labels
    c.setFillColor(ON_SURF_LOW)
    c.setFont("Helvetica", 6.5)
    c.drawString(ML + 5, y - 8, "0%  Low")
    c.drawCentredString(ML + 5 + bar_full * 0.5, y - 8, "50%  Moderate")
    c.drawRightString(ML + 5 + bar_full, y - 8, "High  100%")
    y -= 22

    # ── Risk summary statement ────────────────────────────────────────────────
    y -= 4
    y = callout_box(R_DESC, y, bg=R_BG, border=R_FG, fg=R_ON, left_bar=R_FG)
    y -= 8

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 4 — CLINICAL INTERPRETATION
    # ══════════════════════════════════════════════════════════════════════════
    y = section_header("Clinical Interpretation", y, accent=SECONDARY)
    y -= 6

    interp_lines = [
        ("Risk Model",
         "This assessment employs a supervised machine-learning classifier trained on large-scale "
         "population health survey data. The model synthesises demographic, behavioural, and clinical "
         "variables to generate a probabilistic estimate of cardiovascular disease risk. The output "
         "is expressed as a percentage reflecting relative population-level risk, not absolute "
         "lifetime probability."),
        ("Score Meaning",
         f"A score of {risk:.1f}% places this individual in the {get_risk_level(risk).lower()} tier "
         f"of the model's risk distribution. The estimated cardiac age of {heart_age} years "
         f"{'exceeds' if heart_age > 50 else 'is consistent with'} the chronological age range, "
         f"which is a secondary indicator of cumulative cardiovascular burden. "
         f"The lifestyle index of {lifestyle}/100 reflects modifiable behavioural risk factors."),
        ("Limitations",
         "Predictive models of this nature carry inherent statistical uncertainty and are not "
         "equivalent to a clinical evaluation. Results may be influenced by the completeness and "
         "accuracy of self-reported data. This report should not be used as the sole basis for "
         "any clinical or therapeutic decision."),
    ]
    for heading_lbl, body in interp_lines:
        y = ensure(y, 30)
        c.setFillColor(TERTIARY)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(ML + 8, y, heading_lbl)
        y -= 13
        y = para(body, y, x=ML + 8, size=8.5, color=ON_SURF_MED, line_h=13)
        y -= 8

    y -= 4

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 5 — EVIDENCE-BASED RECOMMENDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    y = section_header("Evidence-Based Recommendations", y, accent=SUCCESS)
    y -= 6

    general_recs = [
        ("Cardioprotective Nutrition",
         "Adopt a dietary pattern consistent with Mediterranean or DASH guidelines: prioritise "
         "vegetables, legumes, whole grains, oily fish, and unsaturated fats. Restrict sodium "
         "to below 2,300 mg per day, eliminate trans fats, and limit added sugars and "
         "ultra-processed foods to reduce systemic inflammation and atherogenic lipid profiles."),
        ("Structured Physical Activity",
         "Target a minimum of 150 minutes of moderate-intensity aerobic exercise or 75 minutes of "
         "vigorous activity per week, in accordance with WHO cardiovascular guidelines. Supplement "
         "with two sessions of resistance training weekly to improve metabolic and vascular health. "
         "Avoid prolonged sedentary periods — break sitting time every 30–60 minutes."),
        ("Tobacco Cessation & Alcohol Moderation",
         "Complete tobacco cessation is the single most impactful modifiable risk reduction "
         "available. Even reduction significantly lowers endothelial damage and thrombotic risk. "
         "Limit alcohol consumption to no more than 14 units per week with alcohol-free days, "
         "as excess intake is independently associated with hypertension and arrhythmia."),
        ("Metabolic & Biometric Monitoring",
         "Maintain a BMI within the 18.5–24.9 kg/m² range. Have fasting lipids, blood glucose, "
         "HbA1c, and blood pressure reviewed annually or more frequently if indicated. "
         "Target systolic blood pressure below 130 mmHg and LDL cholesterol below 100 mg/dL "
         "in moderate-to-high risk profiles. Engage in periodic 12-lead ECG screening "
         "if recommended by your physician."),
        ("Psychosocial & Sleep Health",
         "Chronic psychological stress activates the HPA axis, raising cortisol and inflammatory "
         "markers that accelerate vascular ageing. Integrate structured stress-reduction "
         "practices such as cognitive behavioural techniques, mindfulness, or guided relaxation. "
         "Achieve 7–9 hours of quality sleep nightly; untreated sleep apnoea is a significant "
         "independent risk factor for hypertension and cardiac events."),
    ]
    for _, rec_text in general_recs:
        y = bullet_item(rec_text, y, dot=SUCCESS, bg=SUCCESS_CONT)
        y -= 4

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 6 — PERSONALISED CLINICAL ACTIONS
    # ══════════════════════════════════════════════════════════════════════════
    if feature_to_recommendation:
        y -= 4
        y = section_header("Personalised Clinical Actions", y, accent=TERTIARY)
        y -= 6
        for rec in feature_to_recommendation.values():
            clean = rec.strip().lstrip("-").strip()
            y = bullet_item(clean, y, dot=TERTIARY, bg=TERT_CONT)
            y -= 4

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 7 — MEDICAL DISCLAIMER
    # ══════════════════════════════════════════════════════════════════════════
    y -= 8
    disclaimer = (
        "MEDICAL DISCLAIMER  —  This document has been generated by an artificial intelligence "
        "system and is provided solely for informational and educational purposes. It does not "
        "constitute, and must not be interpreted as, medical advice, clinical diagnosis, or a "
        "treatment recommendation. The outputs of this model are probabilistic estimates derived "
        "from population-level data and cannot account for the full complexity of an individual's "
        "medical history, comorbidities, or clinical presentation. Users are strongly advised to "
        "share this report with a licensed healthcare professional before making any health-related "
        "decisions. Do not delay seeking medical attention or disregard professional advice on the "
        "basis of information contained in this document."
    )
    y = callout_box(disclaimer, y, bg=WARN_CONT, border=WARN, fg=ON_WARN_CONT, left_bar=WARN)

    # ══════════════════════════════════════════════════════════════════════════
    # FINAL FOOTER
    # ══════════════════════════════════════════════════════════════════════════
    draw_footer()
    c.save()
    buffer.seek(0)
    return buffer.read()


# =============================================================================
# SESSION STATE
# =============================================================================
def init_session_state():
    defaults = {
        "risk_result": None,
        "report_input_data": None,
        "ftr_recs": {},
        "pie_df": None,
        "recommendation_intro": "",
        "feature_importance_df": None,
        "history": [],
        "whatif_overrides": {},
        "active_tab": "Assessment",
        "health_goals": [],
        "goal_progress": {},
        "notes_log": [],
        "quiz_score": None,
        "quiz_answers": {},
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    raw_history = st.session_state.get("history", [])
    clean_history = []
    for entry in raw_history if isinstance(raw_history, list) else []:
        if not isinstance(entry, dict):
            continue
        try:
            entry["risk"] = float(entry["risk"])
        except (KeyError, TypeError, ValueError):
            continue
        entry.setdefault("time", "")
        entry.setdefault("input", {})
        clean_history.append(entry)
    st.session_state.history = clean_history


# =============================================================================
# STYLES  — FIX 1 included: .md-callout gets box-sizing + overflow fix
# =============================================================================
def render_styles():
    st.markdown(
        """
<style>
/* ============================================================
   GOOGLE FONTS — Outfit (display) + DM Sans (body)
   ============================================================ */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800;900&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,700;1,9..40,400&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200');

/* ============================================================
   DESIGN TOKENS
   ============================================================ */
:root {
    --md-primary: #006a6a;
    --md-primary-rgb: 0, 106, 106;
    --md-secondary: #3f5f90;
    --md-secondary-rgb: 63, 95, 144;
    --md-tertiary: #ba1a1a;
    --md-tertiary-rgb: 186, 26, 26;
    --md-success: #14b8a6;
    --md-success-rgb: 20, 184, 166;
    --md-warning: #f59e0b;
    --md-warning-rgb: 245, 158, 11;
    --md-error: #ef4444;
    --md-error-rgb: 239, 68, 68;
    --md-surface: rgba(255,255,255,0.050);
    --md-surface-container: rgba(255,255,255,0.075);
    --md-surface-container-high: rgba(255,255,255,0.105);
    --md-surface-container-highest: rgba(255,255,255,0.135);
    --md-outline: rgba(148, 163, 184, 0.28);
    --md-outline-variant: rgba(148, 163, 184, 0.18);
    --md-soft: rgba(148, 163, 184, 0.96);
    --md-shadow-1: 0 4px 14px rgba(15, 23, 42, 0.08);
    --md-shadow-2: 0 12px 34px rgba(15, 23, 42, 0.12);
    --md-shadow-3: 0 22px 58px rgba(15, 23, 42, 0.16);
    --md-shadow-glow: 0 0 50px rgba(var(--md-primary-rgb), 0.18);
    --md-shadow-success-glow: 0 0 30px rgba(var(--md-success-rgb), 0.22);
    --md-shape-xs: 8px;
    --md-shape-sm: 14px;
    --md-shape-md: 22px;
    --md-shape-lg: 28px;
    --md-shape-xl: 36px;
    --md-shape-pill: 999px;
    --md-ease-emphasized: cubic-bezier(0.2, 0.0, 0, 1.0);
    --md-ease-standard: cubic-bezier(0.2, 0, 0, 1);
    --md-dur-short: 200ms;
    --md-dur-med: 350ms;
    --md-dur-long: 500ms;
}

html, body, [class*="css"], .stMarkdown, .stText, p, span, div, label {
    font-family: 'DM Sans', sans-serif !important;
}
h1, h2, h3, h4, h5, h6,
.md-title, .md-form-title, .md-sidebar-title,
.md-metric-value, .md-result-hero h2 {
    font-family: 'Outfit', sans-serif !important;
}

html { scroll-behavior: smooth; }

html, body {
    overflow-x: hidden !important;
    max-width: 100vw !important;
}

/* ── FIX: Streamlit sidebar toggle button ligature text overflow ── */
[data-testid="stSidebarCollapsedControl"] {
    overflow: hidden !important;
    max-width: 48px !important;
}
[data-testid="stSidebarCollapsedControl"] span {
    font-family: 'Material Symbols Rounded', 'Material Symbols Outlined' !important;
    font-feature-settings: 'liga' !important;
    -webkit-font-feature-settings: 'liga' !important;
    overflow: hidden !important;
    display: inline-block !important;
    max-width: 24px !important;
    white-space: nowrap !important;
    text-overflow: clip !important;
}

@keyframes md-fade-up {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes md-fade-in {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes md-scale-in {
    from { opacity: 0; transform: scale(0.94); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes md-pulse-soft {
    0%,100% { box-shadow: 0 0 0 0 rgba(var(--md-primary-rgb),0.35); }
    50%     { box-shadow: 0 0 0 16px rgba(var(--md-primary-rgb),0.0); }
}
@keyframes md-pulse-error {
    0%,100% { box-shadow: 0 0 0 0 rgba(var(--md-error-rgb),0.35); }
    50%     { box-shadow: 0 0 0 16px rgba(var(--md-error-rgb),0.0); }
}
@keyframes md-shine {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
@keyframes md-float {
    0%,100% { transform: translateY(0) rotate(0deg); }
    33%     { transform: translateY(-5px) rotate(0.5deg); }
    66%     { transform: translateY(-2px) rotate(-0.5deg); }
}
@keyframes md-spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes md-heartbeat {
    0%,100% { transform: scale(1); }
    14%     { transform: scale(1.12); }
    28%     { transform: scale(1); }
    42%     { transform: scale(1.08); }
    70%     { transform: scale(1); }
}
@keyframes md-shimmer {
    0%   { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}
@keyframes md-progress-fill {
    from { width: 0%; }
    to   { width: var(--target-width, 0%); }
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
    max-width: 1360px;
    animation: md-fade-up 500ms var(--md-ease-emphasized) both;
}

[data-testid="stSidebar"] {
    border-right: 1px solid var(--md-outline);
    background:
        linear-gradient(180deg,
            rgba(var(--md-primary-rgb), 0.14),
            rgba(var(--md-secondary-rgb), 0.07) 40%,
            rgba(255,255,255,0.02));
    overflow: hidden !important;
}

[data-testid="stSidebarContent"] {
    overflow-x: hidden !important;
}

@media (max-width: 768px) {
    [data-testid="stSidebar"] {
        background: #12101a !important;
    }
    .md-sidebar-link:hover,
    .md-sidebar-link:active,
    .md-goal-card:hover,
    .md-goal-card:active {
        transform: none !important;
    }
}

/* ── MATERIAL SYMBOLS FONT (for sidebar toggle icon) ─────── */

/* ── SIDEBAR COLLAPSE ARROW FIX ──────────────────────────── */
/* Target Streamlit's actual sidebar toggle: [data-testid="stSidebarCollapsedControl"] */
[data-testid="stSidebarCollapsedControl"] {
    overflow: visible !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    overflow: hidden !important;
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    width: 36px !important;
    height: 36px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    position: relative !important;
}
/* Hide the raw ligature text span */
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stSidebarCollapsedControl"] button .material-symbols-rounded,
[data-testid="stSidebarCollapsedControl"] button .material-symbols-outlined {
    font-size: 0 !important;
    line-height: 0 !important;
    color: transparent !important;
    overflow: hidden !important;
    display: block !important;
    width: 0 !important;
    height: 0 !important;
    visibility: hidden !important;
}
/* Inject a clean chevron via ::after pseudo-element */
[data-testid="stSidebarCollapsedControl"] button::after {
    content: "";
    display: block !important;
    width: 20px !important;
    height: 20px !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' stroke='rgba(148,163,184,0.9)' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='13 17 18 12 13 7'/%3E%3Cpolyline points='6 17 11 12 6 7'/%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-size: contain !important;
    background-position: center !important;
    flex-shrink: 0 !important;
    visibility: visible !important;
}

button[aria-label="Close sidebar"],
button[aria-label="Open sidebar"],
button[aria-label="Collapse sidebar"],
button[aria-label="Expand sidebar"],
button[aria-label="collapse sidebar"],
button[aria-label="expand sidebar"] {
    font-size: 0 !important;
    color: transparent !important;
    letter-spacing: -9999px !important;
    overflow: hidden !important;
}
button[aria-label="Close sidebar"] span,
button[aria-label="Open sidebar"] span,
button[aria-label="Collapse sidebar"] span,
button[aria-label="Expand sidebar"] span,
button[aria-label="collapse sidebar"] span,
button[aria-label="expand sidebar"] span {
    visibility: hidden !important;
    font-size: 0 !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* ── PREVENT HORIZONTAL SCROLL ON MOBILE SIDEBAR ─────────── */
@media (max-width: 768px) {
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
}

[data-testid="stSidebar"] img {
    width: 100%; max-height: 220px; object-fit: contain !important;
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-lg);
    padding: 10px; box-shadow: var(--md-shadow-1); margin-bottom: 12px;
}

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR — MD3 EXPRESSIVE REDESIGN
   ═══════════════════════════════════════════════════════════════ */
.md-sidebar-hero {
    border: 1.5px solid rgba(var(--md-primary-rgb), 0.22);
    border-radius: 22px;
    padding: 20px 18px 18px;
    position: relative; overflow: hidden;
    background:
        radial-gradient(ellipse 260px 120px at 100% 0%,   rgba(var(--md-tertiary-rgb),0.28), transparent 70%),
        radial-gradient(ellipse 200px 140px at 0%   100%, rgba(var(--md-secondary-rgb),0.20), transparent 70%),
        linear-gradient(145deg, rgba(var(--md-primary-rgb),0.16) 0%, rgba(var(--md-secondary-rgb),0.08) 60%, transparent),
        var(--md-surface);
    box-shadow: 0 2px 12px rgba(var(--md-primary-rgb),0.10), 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 18px;
    animation: sb-hero-in 500ms cubic-bezier(0.05,0.7,0.1,1.0) both;
}
@keyframes sb-hero-in {
    from { opacity:0; transform: translateY(10px); }
    to   { opacity:1; transform: translateY(0);    }
}
.md-sidebar-hero::before {
    content: "❤️"; font-size: 80px; opacity: 0.06;
    position: absolute; right: -8px; bottom: -10px;
    line-height: 1; pointer-events: none; user-select: none;
    transform: rotate(-10deg);
}
.md-sidebar-heading {
    font-family: 'Outfit', sans-serif !important;
    font-size: 17px; font-weight: 900; letter-spacing: 0.01em; line-height: 1.2;
    margin-bottom: 6px;
    display: flex; align-items: center; gap: 9px;
}
.md-sidebar-heading-icon {
    font-size: 30px; line-height: 1; flex-shrink: 0;
    animation: md-heartbeat 2.5s ease-in-out infinite;
    display: inline-block;
}
.md-sb-avatar-box {
    width: 54px; height: 54px; min-width: 54px;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.55) 0%, rgba(var(--md-tertiary-rgb),0.35) 100%);
    border: 1px solid rgba(var(--md-primary-rgb),0.35);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px;
    box-shadow: 0 4px 16px rgba(var(--md-primary-rgb),0.28);
    flex-shrink: 0;
    position: relative;
    transform-origin: center;
    animation: md-heartbeat 2.5s ease-in-out infinite;
}
.md-sb-avatar-box::after {
    content: '';
    position: absolute; bottom: -3px; right: -3px;
    width: 13px; height: 13px; border-radius: 50%;
    background: #00c853;
    border: 2px solid var(--md-bg, #0d0d14);
    box-shadow: 0 0 6px rgba(0,200,83,0.55);
    animation: sb-online-heart 2.4s ease-in-out infinite;
}
@keyframes sb-online-heart { 0%,100%{box-shadow:0 0 6px rgba(0,200,83,0.55);} 50%{box-shadow:0 0 10px rgba(0,200,83,0.28);} }
.md-sidebar-heading-text {
    background: linear-gradient(135deg, #ffffff 30%, #14b8a6 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.md-sidebar-kicker {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 11px; border-radius: 999px;
    background: linear-gradient(90deg, rgba(var(--md-primary-rgb),0.20), rgba(var(--md-tertiary-rgb),0.15));
    border: 1.5px solid rgba(var(--md-primary-rgb),0.40);
    color: #14b8a6; font-size: 9.5px; font-weight: 900; letter-spacing: 0.10em;
    margin-bottom: 8px; text-transform: uppercase;
    box-shadow: 0 1px 6px rgba(var(--md-primary-rgb),0.12);
}
.md-sidebar-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 19px; font-weight: 900; line-height: 1.1; margin-bottom: 6px;
    background: linear-gradient(135deg, var(--md-on-surface) 40%, rgba(var(--md-primary-rgb),0.85));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.md-sidebar-text { color: var(--md-soft); font-size: 12px; line-height: 1.5; margin-top: 2px; }

/* ── Stat pills row inside hero ─────────────────────────────── */
.sb-stat-row { display: flex; gap: 7px; margin-top: 14px; flex-wrap: wrap; }
.sb-stat-pill {
    display: inline-flex; flex-direction: column; align-items: center;
    padding: 8px 12px; border-radius: 14px; flex: 1; min-width: 52px;
    background: rgba(var(--md-primary-rgb),0.10);
    border: 1px solid rgba(var(--md-primary-rgb),0.22);
}
.sb-stat-pill-val { font-size: 13px; font-weight: 900; color: var(--md-primary); line-height: 1; }
.sb-stat-pill-lbl { font-size: 9px; font-weight: 700; color: var(--md-soft); letter-spacing: 0.05em; margin-top: 3px; text-transform: uppercase; }

/* ── Section label ──────────────────────────────────────────── */
.md-sidebar-section {
    margin: 20px 0 9px 0;
    display: flex; align-items: center; gap: 7px;
    font-size: 10px; font-weight: 900;
    color: var(--md-soft); text-transform: uppercase; letter-spacing: 0.10em;
}
.md-sidebar-section::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--md-outline-variant), transparent);
}

/* ── Nav cards ──────────────────────────────────────────────── */
.md-sidebar-link {
    display: flex; align-items: center; gap: 12px; padding: 12px 13px;
    border-radius: 16px; border: 1.5px solid var(--md-outline-variant);
    background: var(--md-surface-container); margin-bottom: 8px;
    transition: transform 180ms cubic-bezier(0.05,0.7,0.1,1.0),
                background 180ms cubic-bezier(0.05,0.7,0.1,1.0),
                border-color 180ms, box-shadow 180ms;
    position: relative; overflow: hidden;
}
.md-sidebar-link::before {
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, rgba(var(--md-primary-rgb),0.7), rgba(var(--md-tertiary-rgb),0.5));
    border-radius: 3px 0 0 3px; transform: scaleY(0); transform-origin: center;
    transition: transform 180ms cubic-bezier(0.05,0.7,0.1,1.0);
}
.md-sidebar-link:hover {
    transform: translateX(5px);
    background: rgba(var(--md-primary-rgb),0.10);
    border-color: rgba(var(--md-primary-rgb),0.35);
    box-shadow: 0 3px 12px rgba(var(--md-primary-rgb),0.12);
}
.md-sidebar-link:hover::before { transform: scaleY(1); }
.md-sidebar-icon {
    width: 40px; height: 40px; min-width: 40px;
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.24), rgba(var(--md-secondary-rgb),0.16));
    border: 1px solid rgba(var(--md-primary-rgb),0.18);
    font-size: 18px;
    box-shadow: 0 2px 6px rgba(var(--md-primary-rgb),0.10);
    transition: transform 180ms cubic-bezier(0.05,0.7,0.1,1.0);
}
.md-sidebar-link:hover .md-sidebar-icon { transform: scale(1.12) rotate(-4deg); }
.md-sidebar-link-title { font-family: 'Outfit', sans-serif !important; font-size: 13.5px; font-weight: 800; }
.md-sidebar-link-sub   { color: var(--md-soft); font-size: 11.5px; line-height: 1.35; }

/* ── Daily Heart Tip card (Medibot sb-tip design) ───────────── */
.sb-tip {
    border:1.5px solid rgba(var(--md-secondary-rgb),.28);
    border-radius:22px;
    padding:14px 14px 14px 12px;
    background:
        radial-gradient(ellipse at 6% 40%,  rgba(var(--md-secondary-rgb),.18), transparent 55%),
        radial-gradient(ellipse at 92% 80%, rgba(var(--md-primary-rgb),.12),   transparent 50%),
        rgba(var(--md-secondary-rgb),.04);
    margin-top:8px;
    display:flex; gap:12px; align-items:flex-start;
    position:relative; overflow:hidden;
    box-shadow: 0 4px 20px rgba(var(--md-secondary-rgb),.10), inset 0 1px 0 rgba(255,255,255,.06);
    transition: box-shadow 180ms ease, border-color 180ms ease;
}
.sb-tip:hover {
    border-color: rgba(var(--md-secondary-rgb),.45);
    box-shadow: 0 6px 26px rgba(var(--md-secondary-rgb),.18), inset 0 1px 0 rgba(255,255,255,.08);
}
.sb-tip::before {
    content:''; position:absolute; left:0; top:0; bottom:0; width:3px;
    background: linear-gradient(180deg,
        rgba(var(--md-secondary-rgb),1) 0%,
        rgba(var(--md-primary-rgb),.7) 100%);
    border-radius:22px 0 0 22px;
}
.sb-tip-num-col {
    display:flex; flex-direction:column; align-items:center;
    gap:4px; flex-shrink:0; padding-top:1px;
}
.sb-tip-num-ring {
    position:relative; width:40px; height:40px; flex-shrink:0;
}
.sb-tip-arc-svg {
    width:40px; height:40px; overflow:visible;
}
.sb-tip-arc-bg {
    fill:none;
    stroke: rgba(var(--md-secondary-rgb),.18);
    stroke-width:4;
}
.sb-tip-arc-fill {
    fill:none;
    stroke-width:4;
    stroke-linecap:round;
}
.sb-tip-num-inner {
    position:absolute; inset:0;
    display:flex; align-items:center; justify-content:center;
}
.sb-tip-num-icon {
    font-size:16px; line-height:1;
    filter: drop-shadow(0 0 5px rgba(var(--md-secondary-rgb),.70));
}
.sb-tip-counter {
    font-size:11px; font-weight:900;
    background: linear-gradient(115deg, #14b8a6, #6366f1);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    line-height:1; letter-spacing:-.01em;
}
.sb-tip-total {
    font-size:9px; font-weight:700; opacity:.65;
}
.sb-tip-content { flex:1; min-width:0; }
.sb-tip-title {
    font-size:12px; font-weight:900; line-height:1.2;
    background: linear-gradient(115deg, #14b8a6 0%, #6366f1 100%);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    margin-bottom:5px; letter-spacing:.01em;
}
.sb-tip-body {
    font-size:11.5px; line-height:1.6;
    color:var(--md-soft);
}
.sb-next-tip-wrap { margin-top: 8px; }
.sb-next-tip-wrap .stButton > button {
    min-height: 34px !important;
    height: 34px !important;
    padding: 0 14px !important;
    font-size: 11px !important;
    font-weight: 800 !important;
    letter-spacing: .03em !important;
    border-radius: 999px !important;
    border: 1px solid rgba(var(--md-secondary-rgb),.45) !important;
    background: linear-gradient(135deg,
        rgba(var(--md-secondary-rgb),.16) 0%,
        rgba(var(--md-primary-rgb),.12) 100%) !important;
    color: #14b8a6 !important;
    box-shadow: none !important;
    transform: none !important;
    transition: transform 140ms ease, background 140ms ease,
                box-shadow 140ms ease, border-color 140ms ease !important;
}
.sb-next-tip-wrap .stButton > button:hover {
    transform: translateY(-1px) !important;
    background: linear-gradient(135deg,
        rgba(var(--md-secondary-rgb),.28) 0%,
        rgba(var(--md-primary-rgb),.20) 100%) !important;
    box-shadow: 0 3px 12px rgba(var(--md-secondary-rgb),.22) !important;
    color: white !important;
    border-color: rgba(var(--md-secondary-rgb),.70) !important;
}
.sb-next-tip-wrap .stButton > button:active { transform: translateY(0) !important; }

/* ── Medical note ───────────────────────────────────────────── */
.md-sidebar-note {
    border: 1.5px solid rgba(var(--md-warning-rgb), 0.32);
    border-radius: 16px; padding: 13px 14px;
    background:
        radial-gradient(ellipse 200px 80px at 100% 0%, rgba(var(--md-warning-rgb),0.10), transparent),
        rgba(var(--md-warning-rgb), 0.07);
    color: var(--md-soft); font-size: 12px; line-height: 1.5; margin-top: 12px;
}

/* ── Footer ─────────────────────────────────────────────────── */
.md-sidebar-footer {
    text-align: center; color: var(--md-soft); font-size: 11.5px;
    margin-top: 16px; padding: 10px 0;
    border-top: 1px solid var(--md-outline-variant);
}
.md-sidebar-footer strong { color: var(--md-primary); }

.md-hero {
    position: relative; overflow: hidden;
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-xl);
    padding: 34px;
    margin: 10px 0 20px 0;
    background:
        radial-gradient(1100px 400px at 100% 0%, rgba(var(--md-primary-rgb),0.20), transparent 60%),
        radial-gradient(800px 320px at 0% 100%, rgba(var(--md-tertiary-rgb),0.12), transparent 60%),
        linear-gradient(135deg, rgba(var(--md-primary-rgb), 0.17), rgba(var(--md-secondary-rgb), 0.10) 58%, rgba(var(--md-tertiary-rgb), 0.07)),
        var(--md-surface);
    box-shadow: var(--md-shadow-3), var(--md-shadow-glow);
}
.md-hero::before {
    content:""; position:absolute; inset:0;
    background: repeating-linear-gradient(45deg, rgba(255,255,255,0.018) 0 2px, transparent 2px 16px);
    pointer-events:none;
}
.md-hero::after {
    content:""; position:absolute; top:-60px; right:-60px;
    width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--md-primary-rgb),0.14), transparent 70%);
    pointer-events:none;
    animation: md-spin-slow 30s linear infinite;
}
.md-hero-grid {
    display: grid; grid-template-columns: minmax(0, 1fr) 300px;
    gap: 32px; align-items: center;
}
.md-hero-brand { display: flex; align-items: center; gap: 20px; min-width: 0; }

.md-hero-avatar.md-sb-avatar-box {
    width: 100px; height: 100px; min-width: 100px;
    border-radius: var(--md-shape-lg);
    font-size: 52px;
    box-shadow: var(--md-shadow-2);
    animation: none;
    margin-bottom: 0;
    flex-shrink: 0;
}
.md-hero-avatar.md-sb-avatar-box::after { display: none; }
.md-kicker {
    display: inline-flex; padding: 7px 13px;
    border-radius: var(--md-shape-pill);
    border: 1px solid rgba(var(--md-primary-rgb), 0.36);
    background: rgba(var(--md-primary-rgb), 0.13);
    color: #14b8a6; font-size: 12px; font-weight: 800;
    letter-spacing: 0.04em; margin-bottom: 11px;
    text-transform: uppercase;
}
.md-title {
    margin: 0; font-size: clamp(32px, 4.2vw, 58px);
    line-height: 0.97; font-weight: 900; letter-spacing: -0.5px;
    font-family: 'Outfit', sans-serif !important;
    background: linear-gradient(110deg, #ffffff 25%, rgba(255,255,255,0.50) 55%, #ffffff 85%);
    background-size: 200% 100%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: md-shine 8s linear infinite;
}
.md-subtitle { max-width: 780px; margin-top: 13px; color: var(--md-soft); font-size: 16px; line-height: 1.7; }

.md-pill {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-md);
    padding: 14px 16px; margin-bottom: 10px;
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-1);
    transition: transform var(--md-dur-short) var(--md-ease-emphasized),
                box-shadow var(--md-dur-short) var(--md-ease-emphasized);
}
.md-pill:hover { transform: translateY(-3px); box-shadow: var(--md-shadow-2); }
.md-pill-label { color: var(--md-soft); font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 4px; }
.md-pill-value { font-family: 'Outfit', sans-serif !important; font-size: 16px; font-weight: 800; }

.md-info-grid {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 16px; margin: 16px 0 22px 0;
}
.md-info-card {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-lg);
    padding: 20px; min-height: 148px;
    background: var(--md-surface);
    box-shadow: var(--md-shadow-1);
    transition:
        transform var(--md-dur-med) var(--md-ease-emphasized),
        box-shadow var(--md-dur-med) var(--md-ease-emphasized),
        background var(--md-dur-med) var(--md-ease-emphasized);
    position: relative; overflow: hidden;
    animation: md-fade-up 500ms var(--md-ease-emphasized) both;
}
.md-info-card:nth-child(1) { animation-delay: 80ms; }
.md-info-card:nth-child(2) { animation-delay: 160ms; }
.md-info-card:nth-child(3) { animation-delay: 240ms; }
.md-info-card::after {
    content:""; position:absolute; inset:0;
    background: linear-gradient(135deg, transparent 60%, rgba(var(--md-primary-rgb),0.10));
    opacity:0; transition: opacity var(--md-dur-med) var(--md-ease-emphasized); pointer-events:none;
}
.md-info-card:hover { transform: translateY(-4px); background: var(--md-surface-container); box-shadow: var(--md-shadow-2); }
.md-info-card:hover::after { opacity:1; }
.md-info-icon {
    width: 48px; height: 48px; border-radius: var(--md-shape-sm);
    margin-bottom: 13px;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.28), rgba(var(--md-secondary-rgb),0.20));
    display: flex; align-items: center; justify-content: center;
    font-family: 'Outfit', sans-serif !important; font-weight: 900; font-size: 18px;
}
.md-info-card strong { display: block; font-family: 'Outfit', sans-serif !important; font-size: 16px; font-weight: 800; margin-bottom: 7px; }
.md-info-card span   { display: block; color: var(--md-soft); line-height: 1.6; font-size: 14px; }

.md-form-title    { font-family: 'Outfit', sans-serif !important; font-size: 28px; font-weight: 900; margin: 6px 0 4px 0; }
.md-form-subtitle { color: var(--md-soft); font-size: 14px; line-height: 1.5; margin-bottom: 16px; }

.risk-badge {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 110px; padding: 12px 18px; border-radius: var(--md-shape-pill);
    color: white; font-family: 'Outfit', sans-serif !important; font-weight: 900; font-size: 19px;
    box-shadow: var(--md-shadow-1);
    animation: md-pulse-soft 2.6s ease-out infinite;
    letter-spacing: 0.02em;
}
.risk-low       { background: linear-gradient(135deg, #0f7b55, #14b8a6); }
.risk-med       { background: linear-gradient(135deg, #b45309, #f59e0b); }
.risk-high      { background: linear-gradient(135deg, #dc2626, #ef4444); animation: md-pulse-error 2.4s ease-out infinite; }
.risk-very-high { background: linear-gradient(135deg, #7f1d1d, #ba1a1a); animation: md-pulse-error 2.0s ease-out infinite; }

.md-muted    { color: var(--md-soft); font-size: 14px; line-height: 1.6; }

.md-rec-list { margin-top: 14px; display: grid; gap: 10px; }
.md-rec-item {
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-md);
    padding: 12px 16px;
    background: rgba(255,255,255,0.042);
    color: var(--md-soft); line-height: 1.55; font-size: 14px;
    transition:
        border-color var(--md-dur-short) var(--md-ease-emphasized),
        background var(--md-dur-short) var(--md-ease-emphasized),
        transform var(--md-dur-short) var(--md-ease-emphasized);
}
.md-rec-item:hover {
    border-color: rgba(var(--md-primary-rgb),0.38);
    background: rgba(var(--md-primary-rgb),0.09);
    transform: translateX(4px);
}

.md-metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 13px; margin: 14px 0;
}
.md-metric {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-md);
    padding: 16px;
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-1);
    transition: transform var(--md-dur-short) var(--md-ease-emphasized),
                box-shadow var(--md-dur-short) var(--md-ease-emphasized);
    animation: md-scale-in 400ms var(--md-ease-emphasized) both;
}
.md-metric:hover { transform: translateY(-3px); box-shadow: var(--md-shadow-2); }
.md-metric-label {
    color: var(--md-soft); font-size: 11px; font-weight: 800;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.md-metric-value {
    font-family: 'Outfit', sans-serif !important;
    font-size: 30px; font-weight: 900; margin-top: 5px; line-height: 1.02;
}
.md-metric-sub { color: var(--md-soft); font-size: 12px; margin-top: 5px; }

.md-progress {
    height: 10px; border-radius: var(--md-shape-pill);
    background: rgba(148,163,184,0.18); overflow: hidden; margin-top: 9px;
}
.md-progress > span {
    display: block; height: 100%; border-radius: inherit;
    transition: width 700ms var(--md-ease-emphasized);
}

.md-chip-row { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 9px; }
.md-chip {
    padding: 6px 11px; border-radius: var(--md-shape-pill);
    background: rgba(var(--md-primary-rgb),0.12);
    border: 1px solid rgba(var(--md-primary-rgb),0.32);
    color: #14b8a6; font-size: 12px; font-weight: 800;
    transition: transform var(--md-dur-short) var(--md-ease-emphasized);
}
.md-chip:hover { transform: translateY(-2px); }
.md-chip.warn { background: rgba(var(--md-warning-rgb),0.12); border-color: rgba(var(--md-warning-rgb),0.32); color: #f59e0b; }
.md-chip.bad  { background: rgba(var(--md-error-rgb),0.12); border-color: rgba(var(--md-error-rgb),0.32); color: #ef4444; }
.md-chip.good { background: rgba(var(--md-success-rgb),0.12); border-color: rgba(var(--md-success-rgb),0.32); color: #14b8a6; }

.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] button {
    border-radius: var(--md-shape-pill) !important;
    min-height: 48px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    border: 1px solid rgba(var(--md-primary-rgb), 0.40) !important;
    background: linear-gradient(135deg, #006a6a, #14b8a6) !important;
    color: white !important;
    box-shadow: var(--md-shadow-1);
    transition:
        transform var(--md-dur-short) var(--md-ease-emphasized),
        box-shadow var(--md-dur-short) var(--md-ease-emphasized) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-3px) !important;
    box-shadow: var(--md-shadow-2) !important;
}
.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0px) !important;
}

.btn-secondary .stButton > button {
    background: var(--md-surface-container) !important;
    border-color: var(--md-outline) !important;
    color: var(--md-soft) !important;
}

[data-baseweb="select"], textarea, input[type="text"], input[type="number"] {
    border-radius: var(--md-shape-sm) !important;
}
[data-testid="stForm"] .stSelectbox label,
[data-testid="stForm"] .stTextInput label,
[data-testid="stForm"] .stNumberInput label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.92rem; font-weight: 700;
}
div[data-testid="stExpander"] { border-radius: var(--md-shape-md) !important; }
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: var(--md-shape-lg) !important;
    border-color: var(--md-outline) !important;
    background: var(--md-surface) !important;
    box-shadow: var(--md-shadow-1);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 6px; background: var(--md-surface-container);
    padding: 7px; border-radius: var(--md-shape-pill);
    border: 1px solid var(--md-outline-variant);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--md-shape-pill) !important;
    padding: 9px 20px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 14px !important;
    color: var(--md-soft) !important;
    transition: all var(--md-dur-short) var(--md-ease-emphasized) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.24), rgba(var(--md-secondary-rgb),0.20)) !important;
    color: white !important;
    box-shadow: var(--md-shadow-1);
}

.md-result-hero {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-lg);
    padding: 20px;
    background:
        radial-gradient(600px 240px at 100% 0%, rgba(var(--md-primary-rgb),0.18), transparent 60%),
        var(--md-surface-container);
    box-shadow: var(--md-shadow-2);
    margin-bottom: 14px;
    animation: md-scale-in 400ms var(--md-ease-emphasized) both;
}
.md-result-hero h2 {
    margin: 0; font-family: 'Outfit', sans-serif !important;
    font-size: 46px; font-weight: 900; line-height: 1.0;
}
.md-result-hero .lvl { font-size: 13px; font-weight: 900; }
.md-result-hero p   { color: var(--md-soft); font-size: 14px; line-height: 1.55; margin-top: 7px; }

.md-history-item {
    display: flex; align-items: center; gap: 13px;
    padding: 11px 14px; border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-md); background: var(--md-surface);
    margin-bottom: 9px;
    transition: transform var(--md-dur-short) var(--md-ease-emphasized),
                background var(--md-dur-short) var(--md-ease-emphasized);
}
.md-history-item:hover { transform: translateX(5px); background: var(--md-surface-container); }
.md-history-dot {
    width: 13px; height: 13px; border-radius: 999px; flex: none;
    box-shadow: 0 0 0 4px rgba(255,255,255,0.07);
}

.md-goal-card {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-md);
    padding: 14px 16px;
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-1);
    margin-bottom: 10px;
    transition: transform var(--md-dur-short) var(--md-ease-emphasized);
}
.md-goal-card:hover { transform: translateX(3px); }
.md-goal-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px; font-weight: 800; margin-bottom: 6px;
}
.md-goal-meta { color: var(--md-soft); font-size: 12px; margin-bottom: 8px; }
.md-goal-bar {
    height: 8px; border-radius: var(--md-shape-pill);
    background: rgba(148,163,184,0.18); overflow: hidden;
}
.md-goal-bar-fill {
    height: 100%; border-radius: inherit;
    background: linear-gradient(90deg, #006a6a, #14b8a6);
    transition: width 600ms var(--md-ease-emphasized);
}
.md-goal-pct {
    font-size: 12px; color: #14b8a6; font-weight: 800; margin-top: 4px; text-align: right;
}

.md-quiz-q {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-lg);
    padding: 20px;
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-1);
    margin-bottom: 14px;
    animation: md-fade-up 400ms var(--md-ease-emphasized) both;
}
.md-quiz-q-text {
    font-family: 'Outfit', sans-serif !important;
    font-size: 17px; font-weight: 800; margin-bottom: 14px; line-height: 1.4;
}
.md-quiz-score {
    border: 2px solid rgba(var(--md-primary-rgb), 0.50);
    border-radius: var(--md-shape-lg);
    padding: 24px; text-align: center;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.14), rgba(var(--md-secondary-rgb),0.09));
    box-shadow: var(--md-shadow-glow);
    animation: md-scale-in 400ms var(--md-ease-emphasized) both;
}
.md-quiz-score-num {
    font-family: 'Outfit', sans-serif !important;
    font-size: 56px; font-weight: 900; color: #14b8a6; line-height: 1;
}

.md-chat-bubble {
    padding: 12px 16px;
    border-radius: var(--md-shape-md);
    margin-bottom: 10px;
    font-size: 14px; line-height: 1.6;
    max-width: 88%;
    animation: md-fade-up 300ms var(--md-ease-emphasized) both;
}
.md-chat-user {
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.22), rgba(var(--md-secondary-rgb),0.16));
    border: 1px solid rgba(var(--md-primary-rgb),0.30);
    margin-left: auto; text-align: right;
    border-bottom-right-radius: var(--md-shape-xs);
}
.md-chat-bot {
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    margin-right: auto;
    border-bottom-left-radius: var(--md-shape-xs);
}
.md-chat-bot-label {
    font-size: 11px; font-weight: 900; color: #14b8a6;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 5px;
}

.md-note-item {
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-md);
    padding: 13px 16px;
    background: var(--md-surface);
    margin-bottom: 9px;
    transition: transform var(--md-dur-short) var(--md-ease-emphasized);
    animation: md-fade-up 300ms var(--md-ease-emphasized) both;
}
.md-note-item:hover { transform: translateX(4px); }
.md-note-time { color: var(--md-soft); font-size: 11px; margin-bottom: 4px; }
.md-note-text { font-size: 14px; line-height: 1.55; }
.md-note-tag  {
    display: inline-block; padding: 3px 9px;
    border-radius: var(--md-shape-pill);
    background: rgba(var(--md-primary-rgb),0.12);
    border: 1px solid rgba(var(--md-primary-rgb),0.26);
    color: #14b8a6; font-size: 11px; font-weight: 800;
    margin-top: 6px;
}

.md-radar-legend {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin-top: 12px;
}
.md-radar-item {
    display: flex; align-items: center; gap: 8px;
    font-size: 13px; color: var(--md-soft);
}
.md-radar-dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }

.md-section-label {
    display: flex; align-items: center; gap: 12px;
    margin: 22px 0 14px 0;
}
.md-section-label-text {
    font-family: 'Outfit', sans-serif !important;
    font-size: 20px; font-weight: 900; white-space: nowrap;
}
.md-section-label-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--md-outline), transparent);
}

.md-stat-card {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-lg);
    padding: 22px; text-align: center;
    background:
        linear-gradient(135deg, rgba(var(--md-primary-rgb),0.12), rgba(var(--md-secondary-rgb),0.07)),
        var(--md-surface);
    box-shadow: var(--md-shadow-1);
    transition: transform var(--md-dur-med) var(--md-ease-emphasized),
                box-shadow var(--md-dur-med) var(--md-ease-emphasized);
    animation: md-scale-in 450ms var(--md-ease-emphasized) both;
}
.md-stat-card:hover { transform: translateY(-5px); box-shadow: var(--md-shadow-2), var(--md-shadow-glow); }
.md-stat-icon { font-size: 32px; margin-bottom: 10px; animation: md-heartbeat 2.5s ease-in-out infinite; }
.md-stat-num {
    font-family: 'Outfit', sans-serif !important;
    font-size: 38px; font-weight: 900; line-height: 1;
}
.md-stat-label { color: var(--md-soft); font-size: 13px; margin-top: 6px; line-height: 1.45; }

/* ============================================================
   FIX 1 — Callout overflow fix (constrain inside bordered containers)
   ============================================================ */
.md-callout {
    border-radius: var(--md-shape-md);
    padding: 14px 16px;
    margin: 10px 0;
    display: flex; align-items: flex-start; gap: 12px;
    font-size: 14px; line-height: 1.55;
    /* FIX: prevent overflow out of parent container */
    box-sizing: border-box;
    width: 100%;
    overflow: hidden;
    word-break: break-word;
    min-width: 0;
}
.md-callout-info {
    background: rgba(var(--md-secondary-rgb),0.10);
    border: 1px solid rgba(var(--md-secondary-rgb),0.28);
    color: var(--md-soft);
}
.md-callout-success {
    background: rgba(var(--md-success-rgb),0.10);
    border: 1px solid rgba(var(--md-success-rgb),0.28);
    color: var(--md-soft);
}
.md-callout-warn {
    background: rgba(var(--md-warning-rgb),0.10);
    border: 1px solid rgba(var(--md-warning-rgb),0.28);
    color: var(--md-soft);
}
.md-callout-icon { font-size: 18px; flex: none; margin-top: 1px; }

@media print {
    [data-testid="stSidebar"], .stButton, .stDownloadButton, .stTabs [data-baseweb="tab-list"] { display: none !important; }
    .md-hero { box-shadow: none; }
    body { background: white !important; color: black !important; }
}

@media (max-width: 980px) {
    .md-hero-grid, .md-info-grid { grid-template-columns: 1fr; }
    .md-hero-brand { align-items: flex-start; flex-direction: column; }
    .md-hero { padding: 22px; border-radius: var(--md-shape-lg); }
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
@media (max-width: 600px) {
    .md-title { font-size: 32px; }
    .md-subtitle { font-size: 15px; }
    .risk-badge { width: 100%; }
    .md-result-hero h2 { font-size: 38px; }
    .md-metric-grid { grid-template-columns: 1fr 1fr; }
    .md-radar-legend { grid-template-columns: 1fr; }
}

/* ── LIGHT MODE OVERRIDES ─────────────────────────────────── */
[data-theme="light"] {
    --md-surface: rgba(103,106,0,0.00);
    --md-surface-container: rgba(0,106,106,0.08);
    --md-surface-container-high: rgba(0,106,106,0.12);
    --md-surface-container-highest: rgba(0,106,106,0.16);
    --md-outline: rgba(0,106,106,0.22);
    --md-outline-variant: rgba(0,106,106,0.14);
    --md-soft: rgba(10,60,60,0.70);
    --md-shadow-1: 0 4px 14px rgba(0,106,106,0.10);
    --md-shadow-2: 0 12px 34px rgba(0,106,106,0.14);
    --md-shadow-3: 0 22px 58px rgba(0,106,106,0.18);
}

[data-theme="light"] .stApp,
[data-theme="light"] [data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #f0fbfb 0%, #e8f5f4 55%, #eef5fb 100%) !important;
    color: #0a1f1f !important;
}
[data-theme="light"] [data-testid="stMain"] { background: transparent !important; }

/* Sidebar (non-mobile) */
[data-theme="light"] [data-testid="stSidebar"] {
    background:
        linear-gradient(180deg,
            rgba(0,106,106,0.12),
            rgba(63,95,144,0.07) 40%,
            rgba(255,255,255,0.02)) !important;
    border-right: 1px solid rgba(0,106,106,0.18) !important;
}
@media (max-width: 768px) {
    [data-theme="light"] [data-testid="stSidebar"] {
        background: #12101a !important;
    }
}

/* Sidebar components */
[data-theme="light"] .md-sidebar-hero {
    background:
        radial-gradient(ellipse 260px 120px at 100% 0%, rgba(0,106,106,0.18), transparent 70%),
        linear-gradient(135deg, rgba(0,106,106,0.12), rgba(63,95,144,0.08)),
        rgba(255,255,255,0.90) !important;
    border-color: rgba(0,106,106,0.22) !important;
}
[data-theme="light"] .md-sidebar-kicker {
    background: rgba(0,106,106,0.11) !important;
    border-color: rgba(0,106,106,0.30) !important;
    color: #004f4f !important;
}
[data-theme="light"] .md-sidebar-title   { color: #0a1f1f !important; }
[data-theme="light"] .md-sidebar-text    { color: #3a6060 !important; }
[data-theme="light"] .md-sidebar-section { color: #5a8080 !important; }
[data-theme="light"] .md-sidebar-link {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(0,106,106,0.16) !important;
}
[data-theme="light"] .md-sidebar-link:hover {
    background: rgba(0,106,106,0.08) !important;
    border-color: rgba(0,106,106,0.34) !important;
}
[data-theme="light"] .md-sidebar-link-title { color: #0a1f1f !important; }
[data-theme="light"] .md-sidebar-link-sub   { color: #3a6060 !important; }
[data-theme="light"] .md-sidebar-note {
    background: rgba(245,158,11,0.07) !important;
    border-color: rgba(245,158,11,0.26) !important;
    color: #3a6060 !important;
}
[data-theme="light"] .md-sidebar-tip-card {
    background: rgba(0,106,106,0.06) !important;
    border-color: rgba(0,106,106,0.20) !important;
}
[data-theme="light"] .md-sidebar-tip-header { background: rgba(0,106,106,0.08) !important; }
[data-theme="light"] .md-sidebar-tip-body   { color: #3a6060 !important; }
[data-theme="light"] .md-sidebar-tip-label  { color: #004f4f !important; }
[data-theme="light"] .md-sidebar-footer { color: #5a8080 !important; border-color: rgba(0,106,106,0.15) !important; }

/* Hero */
[data-theme="light"] .md-hero {
    background:
        linear-gradient(135deg, rgba(0,106,106,0.12), rgba(63,95,144,0.07) 58%, rgba(186,26,26,0.04)),
        rgba(255,255,255,0.80) !important;
    border-color: rgba(0,106,106,0.20) !important;
}
[data-theme="light"] .md-kicker {
    background: rgba(0,106,106,0.10) !important;
    border-color: rgba(0,106,106,0.28) !important;
    color: #004f4f !important;
}

[data-theme="light"] .md-title    { color: #0a1f1f !important; }
[data-theme="light"] .md-subtitle { color: #3a6060 !important; }
[data-theme="light"] .md-pill {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(0,106,106,0.18) !important;
}
[data-theme="light"] .md-pill-label { color: #5a8080 !important; }
[data-theme="light"] .md-pill-value { color: #0a1f1f !important; }
[data-theme="light"] .md-chip {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(0,106,106,0.20) !important;
    color: #003c3c !important;
}

/* Info cards */
[data-theme="light"] .md-info-card {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(0,106,106,0.16) !important;
}
[data-theme="light"] .md-info-card:hover {
    background: rgba(255,255,255,0.94) !important;
    border-color: rgba(0,106,106,0.32) !important;
}
[data-theme="light"] .md-info-card strong { color: #0a1f1f !important; }
[data-theme="light"] .md-info-card span   { color: #3a6060 !important; }
[data-theme="light"] .md-info-icon {
    background: linear-gradient(135deg, rgba(0,106,106,0.18), rgba(63,95,144,0.12)) !important;
    border-color: rgba(0,106,106,0.18) !important;
}

/* Form */
[data-theme="light"] .md-form-title    { color: #0a1f1f !important; }
[data-theme="light"] .md-form-subtitle { color: #3a6060 !important; }
[data-theme="light"] .md-muted         { color: #3a6060 !important; }

/* Recommendation list */
[data-theme="light"] .md-rec-item {
    background: rgba(255,255,255,0.80) !important;
    border-color: rgba(0,106,106,0.14) !important;
    color: #0a1f1f !important;
}
[data-theme="light"] .md-rec-item:hover {
    background: rgba(0,106,106,0.08) !important;
    border-color: rgba(0,106,106,0.30) !important;
}

/* Metrics */
[data-theme="light"] .md-metric {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(0,106,106,0.16) !important;
}
[data-theme="light"] .md-metric:hover {
    background: rgba(255,255,255,0.94) !important;
}
[data-theme="light"] .md-metric-label { color: #5a8080 !important; }
[data-theme="light"] .md-metric-value { color: #0a1f1f !important; }
[data-theme="light"] .md-metric-sub   { color: #5a8080 !important; }
[data-theme="light"] .md-progress {
    background: rgba(0,106,106,0.12) !important;
}

/* Result hero */
[data-theme="light"] .md-result-hero h2 { color: #0a1f1f !important; }

/* Callout boxes */
[data-theme="light"] .md-callout-ok   { background: rgba(20,184,166,0.08) !important; border-color: rgba(20,184,166,0.24) !important; color: #3a6060 !important; }
[data-theme="light"] .md-callout-warn { background: rgba(245,158,11,0.07) !important; border-color: rgba(245,158,11,0.24) !important; color: #5a5270 !important; }

/* Risk badge */
[data-theme="light"] .risk-badge       { color: white !important; }

/* Footer */
[data-theme="light"] .md-footer { border-top-color: rgba(0,106,106,0.16) !important; }
[data-theme="light"] .md-footer-brand-name { color: #0a1f1f !important; }
[data-theme="light"] .md-footer-brand-sub  { color: #5a8080 !important; }
[data-theme="light"] .md-footer-link {
    background: rgba(255,255,255,0.78) !important;
    border-color: rgba(0,106,106,0.14) !important;
    color: #3a6060 !important;
}
[data-theme="light"] .md-footer-link:hover { color: #0a1f1f !important; }
[data-theme="light"] .md-footer-meta       { color: #5a8080 !important; }
[data-theme="light"] .md-footer-version {
    background: rgba(0,106,106,0.09) !important;
    border-color: rgba(0,106,106,0.22) !important;
    color: #004f4f !important;
}
[data-theme="light"] .md-footer-disclaimer { color: #3a6060 !important; }

/* Tabs */
[data-theme="light"] .stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.82) !important;
    border-color: rgba(0,106,106,0.14) !important;
}
[data-theme="light"] .stTabs [data-baseweb="tab"]  { color: #003c3c !important; }
[data-theme="light"] .stTabs [aria-selected="true"] { color: white !important; }

/* Inputs */
[data-theme="light"] input,
[data-theme="light"] textarea,
[data-theme="light"] [data-baseweb="select"] {
    background: rgba(255,255,255,0.90) !important;
    border-color: rgba(0,106,106,0.24) !important;
    color: #0a1f1f !important;
}
[data-theme="light"] label,
[data-theme="light"] p,
[data-theme="light"] li { color: #0a1f1f !important; }
</style>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================
def render_sidebar_card(icon_text, title, subtitle):
    st.markdown(
        f"""
<div class="md-sidebar-link">
  <div class="md-sidebar-icon">{escape(icon_text)}</div>
  <div>
    <div class="md-sidebar-link-title">{escape(title)}</div>
    <div class="md-sidebar-link-sub">{escape(subtitle)}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _get_heart_tip_html(tip_index):
    tip_icon, tip_title, tip_body = HEART_TIPS[tip_index % len(HEART_TIPS)]
    display_num = (tip_index % len(HEART_TIPS)) + 1
    total       = len(HEART_TIPS)
    arc_len     = round(2 * 3.14159 * 16 * (display_num / total), 2)
    return f"""<div class="sb-tip">
    <div class="sb-tip-num-col">
        <div class="sb-tip-num-ring">
            <svg class="sb-tip-arc-svg" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="sb-tip-grad-{display_num}" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%"   stop-color="#14b8a6"/>
                        <stop offset="100%" stop-color="#6366f1"/>
                    </linearGradient>
                </defs>
                <circle class="sb-tip-arc-bg"   cx="20" cy="20" r="16" />
                <circle class="sb-tip-arc-fill" cx="20" cy="20" r="16"
                    stroke="url(#sb-tip-grad-{display_num})"
                    stroke-dasharray="{arc_len} 100.53"
                    transform="rotate(-90 20 20)" />
            </svg>
            <div class="sb-tip-num-inner">
                <span class="sb-tip-num-icon">{escape(tip_icon)}</span>
            </div>
        </div>
        <div class="sb-tip-counter">{display_num}<span class="sb-tip-total">/{total}</span></div>
    </div>
    <div class="sb-tip-content">
        <div class="sb-tip-title">{escape(tip_title)}</div>
        <div class="sb-tip-body">{escape(tip_body)}</div>
    </div>
</div>"""


def render_sidebar():

    # ── Tips cycle sequentially 1→20 then wrap back to 1 ─────────────────
    if "sidebar_tip_idx" not in st.session_state:
        st.session_state.sidebar_tip_idx = 0

    with st.sidebar:
        if Path(SIDEBAR_IMAGE_PATH).exists():
            st.image(str(SIDEBAR_IMAGE_PATH), use_container_width=True)

        # ── Hero card ────────────────────────────────────────────────────
        st.markdown(
            """
<div class="md-sidebar-hero">
  <div style="display:flex; align-items:flex-start; gap:12px; margin-bottom:10px;">
    <div class="md-sb-avatar-box" style="margin-bottom:0; flex-shrink:0;">🫀</div>
    <div style="display:flex; flex-direction:column; justify-content:center; padding-top:4px;">
      <div class="md-sidebar-kicker">✦ AI Heart Health</div>
      <div class="md-sidebar-heading"><span class="md-sidebar-heading-text">Heart Disease Risk Assessment</span></div>
    </div>
  </div>
  <div class="md-sidebar-text">Estimate heart disease risk using guided health inputs, explainable AI insights, and a downloadable PDF report.</div>
  <div class="sb-stat-row">
    <div class="sb-stat-pill"><span class="sb-stat-pill-val">AI</span><span class="sb-stat-pill-lbl">Powered</span></div>
    <div class="sb-stat-pill"><span class="sb-stat-pill-val">SHAP</span><span class="sb-stat-pill-lbl">Insights</span></div>
    <div class="sb-stat-pill"><span class="sb-stat-pill-val">PDF</span><span class="sb-stat-pill-lbl">Report</span></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ── Workflow nav ─────────────────────────────────────────────────
        st.markdown('<div class="md-sidebar-section">Workflow</div>', unsafe_allow_html=True)
        render_sidebar_card("👤", "Demographics", "Age, gender, and background")
        render_sidebar_card("🩺", "Medical History", "Conditions and checkups")
        render_sidebar_card("🏃", "Lifestyle", "Sleep, exercise, smoking, alcohol")
        render_sidebar_card("📄", "PDF Report", "Download your result summary")

        # ── Scan Studio nav ──────────────────────────────────────────────
        st.markdown('<div class="md-sidebar-section">Scan Enhancement Studio</div>', unsafe_allow_html=True)
        render_sidebar_card("🎨", "Scan Studio", "Upload & enhance cardiac images")
        render_sidebar_card("🌈", "6-Panel View", "Contrast, edges, heatmap & more")
        render_sidebar_card("💾", "Download Panels", "Save any enhanced version")

        # ── Daily Heart Tip — sequential cycling ─────────────────────────
        st.markdown('<div class="md-sidebar-section">💡 Daily Heart Tip</div>', unsafe_allow_html=True)
        st.markdown(_get_heart_tip_html(st.session_state.sidebar_tip_idx), unsafe_allow_html=True)
        st.markdown("<div class='sb-next-tip-wrap'>", unsafe_allow_html=True)
        if st.button("✦ Next Tip", use_container_width=True, key="new_tip_btn"):
            st.session_state.sidebar_tip_idx = (st.session_state.sidebar_tip_idx + 1) % len(HEART_TIPS)
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Recent Results ────────────────────────────────────────────────
        if st.session_state.get("history"):
            st.markdown('<div class="md-sidebar-section">Recent Results</div>', unsafe_allow_html=True)
            for entry in st.session_state.history[-3:][::-1]:
                if not isinstance(entry, dict) or "risk" not in entry:
                    continue
                risk_value = entry.get("risk")
                time_value = entry.get("time", "")
                try:
                    risk_value = float(risk_value)
                except (TypeError, ValueError):
                    continue
                color = get_risk_color(risk_value)
                st.markdown(
                    f"""
<div class="md-history-item">
  <div class="md-history-dot" style="background:{color};"></div>
  <div>
    <div class="md-sidebar-link-title">{risk_value:.1f}% — {get_risk_level(risk_value)}</div>
    <div class="md-sidebar-link-sub">{escape(str(time_value))}</div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            if st.button("Clear history", use_container_width=True):
                st.session_state.history = []
                st.rerun()

        # ── Active Goals ──────────────────────────────────────────────────
        active_goals = [g for g in st.session_state.get("health_goals", []) if not g.get("done")]
        if active_goals:
            st.markdown('<div class="md-sidebar-section">Active Goals</div>', unsafe_allow_html=True)
            for goal in active_goals[:3]:
                pct = goal.get("progress", 0)
                st.markdown(
                    f"""
<div class="md-goal-card" style="margin-bottom:7px;">
  <div class="md-goal-title">{escape(goal['title'])}</div>
  <div class="md-goal-bar"><div class="md-goal-bar-fill" style="width:{pct}%;"></div></div>
  <div class="md-goal-pct">{pct}%</div>
</div>
""",
                    unsafe_allow_html=True,
                )

        # ── Safety note + footer ──────────────────────────────────────────
        st.markdown(
            """
<div class="md-sidebar-section">Safety</div>
<div class="md-sidebar-note">
  <strong>⚕️ Medical Note</strong><br/>
  This app is educational support only. It is not a diagnosis or replacement for professional medical care. Always consult a qualified healthcare provider.
</div>
<div class="md-sidebar-footer">
  Made with ❤️ by <strong>Yatin Sharma</strong>
</div>
""",
            unsafe_allow_html=True,
        )


# =============================================================================
# HERO + INFO CARDS
# =============================================================================
def render_hero():
    st.markdown(
        f"""
<div class="md-hero">
  <div class="md-hero-grid">
    <div>
      <div class="md-hero-brand">
        <div class="md-sb-avatar-box md-hero-avatar">🫀</div>
        <div>
          <div class="md-kicker">Heart Risk Assessment</div>
          <h1 class="md-title">Heart Disease<br/>Risk Assessment</h1>
          <p class="md-subtitle">Early heart disease risk assessment powered by machine learning, explainable insights, and personalized prevention guidance.</p>
        </div>
      </div>
    </div>
    <div>
      <div class="md-pill"><div class="md-pill-label">Model</div><div class="md-pill-value">LightGBM Ensemble</div></div>
      <div class="md-pill"><div class="md-pill-label">Output</div><div class="md-pill-value">Risk Score + PDF Report</div></div>
      <div class="md-pill"><div class="md-pill-label">Insights</div><div class="md-pill-value">SHAP-Based Factors</div></div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_info_cards():
    st.markdown(
        """
<div class="md-info-grid">
  <div class="md-info-card">
    <div class="md-info-icon">1</div>
    <strong>Enter Health Details</strong>
    <span>Share demographics, medical history, and lifestyle information through a guided form.</span>
  </div>
  <div class="md-info-card">
    <div class="md-info-icon">2</div>
    <strong>AI Risk Analysis</strong>
    <span>The model estimates your heart disease risk and highlights important contributing factors.</span>
  </div>
  <div class="md-info-card">
    <div class="md-info-icon">3</div>
    <strong>Download Report</strong>
    <span>Generate a clean PDF report with your result, recommendations, and health summary.</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_recommendations(saved_recs):
    if not saved_recs:
        st.markdown(
            """
<div class="md-rec-list">
  <div class="md-rec-item">Keep focusing on heart-healthy habits, regular physical activity, balanced nutrition, and routine medical checkups.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        return
    rec_html = '<div class="md-rec-list">'
    for rec in saved_recs.values():
        clean_rec = escape(rec.strip().lstrip("-").strip())
        rec_html += f'<div class="md-rec-item">{clean_rec}</div>'
    rec_html += "</div>"
    st.markdown(rec_html, unsafe_allow_html=True)


# =============================================================================
# METRIC TILES & DASHBOARD
# =============================================================================
def render_metrics(input_data, risk):
    lifestyle = compute_lifestyle_score(input_data)
    heart_age = heart_age_estimate(input_data, risk)
    avg_pop = 8.5
    diff = risk - avg_pop
    diff_label = f"{'+' if diff >= 0 else ''}{diff:.1f}% vs avg"
    diff_color = "#ef4444" if diff > 0 else "#14b8a6"

    st.markdown(
        f"""
<div class="md-metric-grid">
  <div class="md-metric">
    <div class="md-metric-label">Risk Score</div>
    <div class="md-metric-value" style="color:{get_risk_color(risk)};">{risk:.1f}%</div>
    <div class="md-metric-sub">{get_risk_level(risk)} risk</div>
  </div>
  <div class="md-metric">
    <div class="md-metric-label">Lifestyle Score</div>
    <div class="md-metric-value">{lifestyle}/100</div>
    <div class="md-progress"><span style="width:{lifestyle}%; background:linear-gradient(90deg,#14b8a6,#006a6a);"></span></div>
  </div>
  <div class="md-metric">
    <div class="md-metric-label">Estimated Heart Age</div>
    <div class="md-metric-value">{heart_age}</div>
    <div class="md-metric-sub">years (heuristic)</div>
  </div>
  <div class="md-metric">
    <div class="md-metric-label">Vs Population Avg</div>
    <div class="md-metric-value" style="color:{diff_color};">{diff_label}</div>
    <div class="md-metric-sub">baseline ~{avg_pop}%</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_factor_chips(input_data):
    chips = []
    if input_data.get("smoking_status") != "never_smoked":
        chips.append(("🚬 Smoking", "bad"))
    if input_data.get("exercise_status_in_past_30_Days") == "no":
        chips.append(("🛋️ Sedentary", "warn"))
    if input_data.get("BMI") in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]:
        chips.append(("⚖️ BMI", "warn"))
    if input_data.get("sleep_category") in ["very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours"]:
        chips.append(("😴 Poor Sleep", "warn"))
    if input_data.get("binge_drinking_status") == "yes":
        chips.append(("🍺 Binge Drinking", "bad"))
    if input_data.get("ever_told_you_had_diabetes") == "yes":
        chips.append(("💉 Diabetes", "bad"))
    if input_data.get("general_health") in ["fair", "poor"]:
        chips.append(("📉 Low Health", "warn"))
    if input_data.get("exercise_status_in_past_30_Days") == "yes" and input_data.get("smoking_status") == "never_smoked":
        chips.append(("✅ Active Non-Smoker", "good"))
    if not chips:
        chips.append(("✅ Healthy Profile", "good"))
    html = '<div class="md-chip-row">'
    for label, kind in chips:
        cls = f"md-chip {kind}".strip()
        html += f'<span class="{cls}">{escape(label)}</span>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# WHAT-IF SIMULATOR
# =============================================================================
def render_whatif(model, encoder, base_input, base_risk):
    st.markdown(
        """
<div style="margin-bottom:12px;">
  <div style="font-family:'Outfit',sans-serif!important; font-size:18px; font-weight:900; line-height:1.2; margin-bottom:4px;">🧪 What-If Simulator</div>
  <div style="color:var(--md-soft); font-size:13px; line-height:1.5;">Adjust modifiable factors to see how your risk could change. Original answers stay untouched.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    sim = base_input.copy()
    c1, c2 = st.columns(2)
    with c1:
        sim["smoking_status"] = st.selectbox(
            "🚬 Smoking Status",
            options=["never_smoked", "former_smoker", "current_smoker_some_days", "current_smoker_every_day"],
            index=["never_smoked", "former_smoker", "current_smoker_some_days", "current_smoker_every_day"].index(
                sim.get("smoking_status", "never_smoked")
            ),
            format_func=pretty_value, key="wi_smoke",
        )
        sim["exercise_status_in_past_30_Days"] = st.radio(
            "🏃 Exercise (past 30 days)", ["yes", "no"],
            index=0 if sim.get("exercise_status_in_past_30_Days", "yes") == "yes" else 1,
            horizontal=True, format_func=pretty_value, key="wi_ex",
        )
        sim["BMI"] = st.selectbox(
            "⚖️ BMI Category",
            options=["underweight_bmi_less_than_18_5", "normal_weight_bmi_18_5_to_24_9",
                     "overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"],
            index=["underweight_bmi_less_than_18_5", "normal_weight_bmi_18_5_to_24_9",
                   "overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"].index(
                sim.get("BMI", "normal_weight_bmi_18_5_to_24_9")
            ),
            format_func=pretty_value, key="wi_bmi",
        )
    with c2:
        sim["sleep_category"] = st.selectbox(
            "😴 Sleep Category",
            options=["very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours",
                     "normal_sleep_6_to_8_hours", "long_sleep_9_to_10_hours",
                     "very_long_sleep_11_or_more_hours"],
            index=["very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours",
                   "normal_sleep_6_to_8_hours", "long_sleep_9_to_10_hours",
                   "very_long_sleep_11_or_more_hours"].index(
                sim.get("sleep_category", "normal_sleep_6_to_8_hours")
            ),
            format_func=pretty_value, key="wi_sleep",
        )
        sim["binge_drinking_status"] = st.radio(
            "🍺 Binge Drinking (past 30 days)", ["yes", "no"],
            index=0 if sim.get("binge_drinking_status", "no") == "yes" else 1,
            horizontal=True, format_func=pretty_value, key="wi_binge",
        )
        sim["drinks_category"] = st.selectbox(
            "🍷 Alcohol per Week",
            options=["did_not_drink", "very_low_consumption_0.01_to_1_drinks",
                     "low_consumption_1.01_to_5_drinks", "moderate_consumption_5.01_to_10_drinks",
                     "high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"],
            index=["did_not_drink", "very_low_consumption_0.01_to_1_drinks",
                   "low_consumption_1.01_to_5_drinks", "moderate_consumption_5.01_to_10_drinks",
                   "high_consumption_10.01_to_20_drinks", "very_high_consumption_more_than_20_drinks"].index(
                sim.get("drinks_category", "did_not_drink")
            ),
            format_func=pretty_value, key="wi_drinks",
        )

    if st.button("Recalculate What-If", use_container_width=True, key="wi_btn"):
        try:
            new_risk, _ = predict_heart_disease_risk(sim, model, encoder)
            delta = new_risk - base_risk
            arrow = "▼" if delta < 0 else ("▲" if delta > 0 else "■")
            color = "#14b8a6" if delta < 0 else ("#ef4444" if delta > 0 else "#94a3b8")
            compare_df = pd.DataFrame({
                "Scenario": ["Original Risk", "Simulated Risk"],
                "Risk %": [base_risk, new_risk],
            })
            fig = px.bar(
                compare_df, x="Scenario", y="Risk %",
                color="Scenario",
                color_discrete_sequence=["#94a3b8", color],
                text=compare_df["Risk %"].map(lambda v: f"{v:.1f}%"),
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(
                height=220, margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8"),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="wi_chart")
            st.markdown(
                f"""
<div class="md-pill" style="margin-top:10px;">
  <div class="md-pill-label">Simulated Risk vs Original</div>
  <div class="md-pill-value" style="font-size:22px;">
    {new_risk:.1f}% &nbsp;
    <span style="color:{color}; font-size:16px;">{arrow} {abs(delta):.1f}%</span>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            if delta < 0:
                st.markdown(
                    f'<div class="md-callout md-callout-success"><span class="md-callout-icon">✅</span>These lifestyle changes could reduce your risk by approximately <strong>{abs(delta):.1f}%</strong>. Small consistent steps add up.</div>',
                    unsafe_allow_html=True,
                )
            elif delta > 0:
                st.markdown(
                    f'<div class="md-callout md-callout-warn"><span class="md-callout-icon">⚠️</span>This combination of factors would increase estimated risk by <strong>{delta:.1f}%</strong>.</div>',
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Simulation failed: {e}")


# =============================================================================
# BMI CALCULATOR
# =============================================================================
def render_bmi_calculator():
    st.markdown("#### 🧮 BMI Calculator")
    st.caption("Compute your BMI and auto-fill the matching category in the form.")
    unit = st.radio("Units", ["Metric (kg, cm)", "Imperial (lb, in)"], horizontal=True, key="bmi_unit")
    c1, c2 = st.columns(2)
    if unit.startswith("Metric"):
        with c1: weight = st.number_input("Weight (kg)", 30.0, 250.0, 70.0, 0.5, key="bmi_w")
        with c2: height = st.number_input("Height (cm)", 100.0, 230.0, 170.0, 0.5, key="bmi_h")
        bmi_val = weight / ((height / 100) ** 2) if height > 0 else 0
    else:
        with c1: weight = st.number_input("Weight (lb)", 60.0, 550.0, 154.0, 0.5, key="bmi_w")
        with c2: height = st.number_input("Height (in)", 40.0, 90.0, 67.0, 0.5, key="bmi_h")
        bmi_val = (weight / (height ** 2)) * 703 if height > 0 else 0

    band = next((b for b in BMI_BANDS if b[0] <= bmi_val < b[1]), BMI_BANDS[1])

    bmi_capped = min(max(bmi_val, 0), 40)
    bmi_pct = bmi_capped / 40 * 100
    st.markdown(
        f"""
<div class="md-pill" style="margin-top:10px;">
  <div class="md-pill-label">Your BMI</div>
  <div class="md-pill-value" style="font-size:24px; color:{band[4]};">
    {bmi_val:.1f} — {band[3]}
  </div>
</div>
<div style="position:relative; height:14px; border-radius:999px; overflow:hidden; margin-top:10px;
  background: linear-gradient(90deg, #3b82f6 0%, #14b8a6 25%, #f59e0b 50%, #ef4444 75%, #7f1d1d 100%);">
  <div style="position:absolute; top:0; left:calc({bmi_pct:.1f}% - 7px);
    width:14px; height:14px; border-radius:50%;
    background:white; border:2px solid {band[4]}; box-shadow:0 0 6px rgba(0,0,0,0.4);"></div>
</div>
<div style="display:flex; justify-content:space-between; font-size:11px; color:var(--md-soft); margin-top:4px;">
  <span>Underweight</span><span>Normal</span><span>Overweight</span><span>Obese</span>
</div>
""",
        unsafe_allow_html=True,
    )

    implications = {
        "underweight_bmi_less_than_18_5": "Being underweight may indicate malnutrition and can increase risk of cardiovascular issues, weakened immunity, and bone loss.",
        "normal_weight_bmi_18_5_to_24_9": "Your BMI is in the healthy range. Maintain this with balanced nutrition and regular physical activity.",
        "overweight_bmi_25_to_29_9": "Overweight BMI increases risk of hypertension, type 2 diabetes, and heart disease. Even a 5-10% weight reduction helps.",
        "obese_bmi_30_or_more": "Obesity significantly raises risk of heart disease, stroke, type 2 diabetes, and certain cancers. Consult a healthcare provider for a personalized plan.",
    }
    impl = implications.get(band[2], "")
    callout_class = "md-callout-success" if band[2] == "normal_weight_bmi_18_5_to_24_9" else "md-callout-warn"
    callout_icon = "✅" if band[2] == "normal_weight_bmi_18_5_to_24_9" else "⚠️"
    st.markdown(
        f'<div class="md-callout {callout_class}" style="margin-top:10px;"><span class="md-callout-icon">{callout_icon}</span>{escape(impl)}</div>',
        unsafe_allow_html=True,
    )
    return band[2]


# =============================================================================
# COMPARISON CHART
# =============================================================================
def render_comparison_chart(risk):
    pop_data = pd.DataFrame({
        "Group": ["Population avg", "Same age (rough)", "Healthy lifestyle", "You"],
        "Risk %": [8.5, max(5, risk * 0.7), 4.2, risk],
    })
    fig = px.bar(
        pop_data, x="Risk %", y="Group", orientation="h",
        color="Risk %", color_continuous_scale=[[0, "#14b8a6"], [0.5, "#f59e0b"], [1, "#ef4444"]],
        text=pop_data["Risk %"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False), yaxis=dict(showgrid=False),
        font=dict(color="#94a3b8"),
    )
    return fig


# =============================================================================
# TOP DRIVERS BAR
# =============================================================================
def render_top_drivers(feature_importance_df):
    df = feature_importance_df.copy()
    df["Pretty"] = df["Feature"].apply(lambda f: FEATURE_NAME_MAPPING.get(base_feature_name(f), pretty_value(base_feature_name(f))))
    df = df.groupby("Pretty", as_index=False)["Importance"].sum().sort_values("Importance", ascending=True).tail(8)
    fig = px.bar(
        df, x="Importance", y="Pretty", orientation="h",
        color="Importance", color_continuous_scale="Tealrose",
        text=df["Importance"].map(lambda v: f"{v:.1f}%"),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, title=""), yaxis=dict(title=""),
        font=dict(color="#94a3b8"),
    )
    return fig


# =============================================================================
# LIFESTYLE RADAR CHART
# =============================================================================
def render_lifestyle_radar(input_data):
    factor_scores = {
        "Smoking": 100 if input_data.get("smoking_status") == "never_smoked"
                   else (60 if input_data.get("smoking_status") == "former_smoker"
                         else (30 if input_data.get("smoking_status") == "current_smoker_some_days" else 5)),
        "Exercise": 95 if input_data.get("exercise_status_in_past_30_Days") == "yes" else 15,
        "Sleep": 90 if input_data.get("sleep_category") == "normal_sleep_6_to_8_hours"
                 else (70 if input_data.get("sleep_category") == "long_sleep_9_to_10_hours"
                       else (30 if input_data.get("sleep_category") == "short_sleep_4_to_5_hours" else 10)),
        "BMI": 90 if input_data.get("BMI") == "normal_weight_bmi_18_5_to_24_9"
               else (55 if input_data.get("BMI") == "overweight_bmi_25_to_29_9"
                     else (60 if input_data.get("BMI") == "underweight_bmi_less_than_18_5" else 20)),
        "Alcohol": 95 if input_data.get("binge_drinking_status") == "no" else 20,
        "Checkup": 95 if input_data.get("length_of_time_since_last_routine_checkup") == "past_year"
                   else (70 if input_data.get("length_of_time_since_last_routine_checkup") == "past_2_years" else 30),
        "Mental": 90 if input_data.get("mental_health_status") == "zero_days_not_good"
                  else (50 if input_data.get("mental_health_status") == "1_to_13_days_not_good" else 20),
        "Physical": 90 if input_data.get("physical_health_status") == "zero_days_not_good"
                    else (50 if input_data.get("physical_health_status") == "1_to_13_days_not_good" else 20),
    }

    cats = list(factor_scores.keys())
    vals = list(factor_scores.values())
    cats_closed = cats + [cats[0]]
    vals_closed = vals + [vals[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=cats_closed,
        fill="toself",
        fillcolor="rgba(20,184,166,0.18)",
        line=dict(color="#14b8a6", width=2.5),
        name="Your Profile",
    ))
    fig.add_trace(go.Scatterpolar(
        r=[80] * (len(cats) + 1), theta=cats_closed,
        fill="toself",
        fillcolor="rgba(148,163,184,0.06)",
        line=dict(color="rgba(148,163,184,0.30)", width=1, dash="dash"),
        name="Healthy Target",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10, color="#94a3b8"), gridcolor="rgba(148,163,184,0.15)"),
            angularaxis=dict(tickfont=dict(size=12, color="#cbd5e1"), gridcolor="rgba(148,163,184,0.15)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=12, color="#94a3b8")),
        height=380, margin=dict(l=30, r=30, t=20, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
    )
    return fig


# =============================================================================
# RISK TIMELINE FORECAST
# =============================================================================
def render_risk_forecast(risk, input_data):
    base = risk
    optimistic_reductions = []
    pessimistic_increases = []

    for year in range(1, 6):
        mod_reduction = 0
        if input_data.get("smoking_status") != "never_smoked": mod_reduction += 1.8
        if input_data.get("exercise_status_in_past_30_Days") == "no": mod_reduction += 1.2
        if input_data.get("BMI") in ["overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]: mod_reduction += 0.9
        if input_data.get("binge_drinking_status") == "yes": mod_reduction += 0.7
        optimistic_reductions.append(max(2, base - (mod_reduction * year)))

        age_drift = 1.1 if input_data.get("age_category") in AGE_RISK_CATEGORIES else 0.7
        pessimistic_increases.append(min(95, base + (age_drift * year)))

    years = list(range(1, 6))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0] + years, y=[base] + optimistic_reductions,
        mode="lines+markers", name="With improvements",
        line=dict(color="#14b8a6", width=2.5),
        marker=dict(size=8, color="#14b8a6"),
        fill="tonexty", fillcolor="rgba(20,184,166,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=[0] + years, y=[base] + pessimistic_increases,
        mode="lines+markers", name="Without changes",
        line=dict(color="#ef4444", width=2.5, dash="dash"),
        marker=dict(size=8, color="#ef4444"),
    ))
    fig.add_hline(y=25, line_dash="dot", line_color="rgba(245,158,11,0.5)", annotation_text="Moderate threshold")
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5, font=dict(size=12, color="#94a3b8")),
        xaxis=dict(title="Years", showgrid=False, tickvals=[0,1,2,3,4,5], ticktext=["Now","Yr1","Yr2","Yr3","Yr4","Yr5"], color="#94a3b8"),
        yaxis=dict(title="Risk %", showgrid=True, gridcolor="rgba(148,163,184,0.12)", color="#94a3b8"),
        font=dict(color="#94a3b8"),
    )
    return fig


# =============================================================================
# KNOWLEDGE Q&A WIDGET
# =============================================================================
def render_heart_qa():
    st.markdown(
        """
<div style="margin-bottom:10px;">
  <div style="font-family:'Outfit',sans-serif!important; font-size:18px; font-weight:900; line-height:1.2; margin-bottom:4px;">💬 Heart Health Q&A</div>
  <div style="color:var(--md-soft); font-size:13px; line-height:1.5;">Ask a heart health question. Answers are pre-loaded educational content — not personalized medical advice.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="md-chat-bubble md-chat-user">{escape(msg["content"])}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="md-chat-bubble md-chat-bot">'
                f'<div class="md-chat-bot-label">🫀 Heart Health Info</div>'
                f'{escape(msg["content"])}'
                f'<div style="margin-top:10px; padding-top:8px; border-top:1px solid rgba(148,163,184,0.15); font-size:11px; color:var(--md-soft);">'
                f'⚕️ Educational info only — not medical advice. Consult your doctor for personal health decisions.'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Quick questions — 3+3 layout with more topics
    st.markdown(
        '<div style="font-size:11px; font-weight:800; color:var(--md-soft); text-transform:uppercase; letter-spacing:0.07em; margin-bottom:8px;">Quick Questions</div>',
        unsafe_allow_html=True,
    )
    quick_qs = [
        ("💉 Blood Pressure",       "What is blood pressure and what are the normal ranges?"),
        ("🚬 Smoking & Heart",      "How does smoking affect the heart?"),
        ("🥗 Heart-Healthy Foods",  "What foods are good for heart health?"),
        ("⚖️ BMI & Heart",         "What is BMI and how does it affect heart disease risk?"),
        ("🏃 Exercise Target",      "How much exercise is recommended for heart health?"),
        ("😴 Sleep & Heart",        "How does sleep affect heart health and what is sleep apnea?"),
        ("🧬 Cholesterol",          "What is cholesterol, LDL, and HDL?"),
        ("💊 Diabetes & Heart",     "How does diabetes increase heart disease risk?"),
        ("🧘 Stress & Heart",       "How does chronic stress affect heart disease?"),
        ("🫀 Heart Failure",        "What is heart failure and what are its symptoms?"),
        ("🧠 Stroke",               "What is a stroke and what are the warning signs?"),
        ("🍷 Alcohol & Heart",      "How does alcohol affect the heart?"),
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    row3 = st.columns(3)
    row4 = st.columns(3)
    cols = row1 + row2 + row3 + row4
    for i, (label, question) in enumerate(quick_qs):
        with cols[i]:
            if st.button(label, key=f"qq_{i}", use_container_width=True):
                _process_qa(question)
                st.rerun()

    st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
    with st.form("qa_form", clear_on_submit=True):
        user_q = st.text_input(
            "Your question",
            placeholder="e.g. What causes a heart attack? How do I lower cholesterol?",
            label_visibility="collapsed",
        )
        ask_col, clear_col = st.columns([4, 1])
        with ask_col:
            ask_clicked = st.form_submit_button("💬 Ask", use_container_width=True)
        if ask_clicked and user_q.strip():
            _process_qa(user_q.strip())
            st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def _process_qa(question: str):
    st.session_state.chat_history.append({"role": "user", "content": question})
    q_lower = question.lower().strip()

    best_score = 0
    best_answer = None

    for _topic_key, entry in HEART_HEALTH_FAQ.items():
        keywords = entry["keywords"]
        score = 0
        for kw in keywords:
            kw_l = kw.lower()
            if kw_l in q_lower:
                score += len(kw_l.split()) * 2
        if score > best_score:
            best_score = score
            best_answer = entry["answer"]

    # Fallback: single important word match (skip stopwords)
    if best_score == 0:
        stopwords = {"what", "is", "are", "how", "does", "do", "the", "a", "an", "my", "i", "me", "to", "for", "of", "in", "can", "tell", "explain", "about"}
        q_words = [w for w in q_lower.split() if w not in stopwords and len(w) > 3]
        for _topic_key, entry in HEART_HEALTH_FAQ.items():
            keywords = entry["keywords"]
            score = 0
            for kw in keywords:
                kw_words = kw.lower().split()
                for qw in q_words:
                    for kw_word in kw_words:
                        if qw in kw_word or kw_word in qw:
                            score += 1
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

    if not best_answer or best_score == 0:
        best_answer = (
            "I don't have a specific answer for that in my knowledge base. "
            "Try asking about: heart disease, blood pressure, cholesterol, smoking, exercise, diet, "
            "BMI, sleep, stress, diabetes, stroke, AFib, or heart failure. "
            "For personalized medical advice, consult a qualified healthcare professional "
            "or visit WHO (who.int) or the American Heart Association (heart.org)."
        )
    st.session_state.chat_history.append({"role": "bot", "content": best_answer})


# =============================================================================
# HEART HEALTH KNOWLEDGE QUIZ
# =============================================================================
QUIZ_QUESTIONS = [
    {
        "q": "What is the recommended weekly moderate-intensity aerobic exercise for adults?",
        "options": ["60 minutes", "100 minutes", "150 minutes", "200 minutes"],
        "answer": "150 minutes",
        "explanation": "The AHA recommends at least 150 minutes/week of moderate-intensity aerobic exercise, such as brisk walking.",
    },
    {
        "q": "Which cholesterol type is considered 'good' because it removes cholesterol from arteries?",
        "options": ["LDL", "HDL", "VLDL", "Triglycerides"],
        "answer": "HDL",
        "explanation": "HDL (High-Density Lipoprotein) is called 'good' cholesterol because it transports cholesterol away from arteries to the liver.",
    },
    {
        "q": "What is the normal blood pressure range (systolic/diastolic)?",
        "options": ["100/60 mmHg", "Below 120/80 mmHg", "130/90 mmHg", "140/80 mmHg"],
        "answer": "Below 120/80 mmHg",
        "explanation": "Normal blood pressure is below 120/80 mmHg. Readings above 130/80 are considered elevated or hypertensive.",
    },
    {
        "q": "Which of these is NOT a modifiable heart disease risk factor?",
        "options": ["Smoking", "Age", "BMI", "Diet"],
        "answer": "Age",
        "explanation": "Age is a non-modifiable risk factor. Smoking, BMI, and diet are all modifiable through lifestyle changes.",
    },
    {
        "q": "How many hours of sleep per night is generally recommended for adults?",
        "options": ["4-5 hours", "5-6 hours", "7-9 hours", "10-12 hours"],
        "answer": "7-9 hours",
        "explanation": "Adults need 7-9 hours of quality sleep. Chronic sleep deprivation raises blood pressure and cardiovascular risk.",
    },
    {
        "q": "Which food is richest in heart-healthy omega-3 fatty acids?",
        "options": ["Chicken breast", "Salmon", "White rice", "Cheddar cheese"],
        "answer": "Salmon",
        "explanation": "Fatty fish like salmon, mackerel, and sardines are rich in omega-3 fatty acids that reduce inflammation and heart disease risk.",
    },
    {
        "q": "Approximately how much does quitting smoking reduce heart disease risk in 1-2 years?",
        "options": ["No change", "5% reduction", "Significantly reduces", "Doubles the risk"],
        "answer": "Significantly reduces",
        "explanation": "Within 1-2 years of quitting, the risk of heart disease drops significantly. After 5-15 years, risk approaches that of non-smokers.",
    },
]


def render_quiz():
    total = len(QUIZ_QUESTIONS)

    # ── Quiz CSS ────────────────────────────────────────────────────────────────
    st.markdown(
        """
<style>
/* ── Quiz wrapper ── */
.md-quiz-wrap {
    background: var(--md-surface);
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-xl);
    overflow: hidden;
    box-shadow: var(--md-shadow-2);
    animation: md-fade-up 450ms var(--md-ease-emphasized) both;
}

/* ── Quiz hero header ── */
.md-quiz-header {
    position: relative; overflow: hidden;
    padding: 28px 28px 24px 28px;
    background:
        radial-gradient(800px 260px at 100% 0%, rgba(var(--md-primary-rgb),0.22), transparent 60%),
        radial-gradient(500px 200px at 0% 100%, rgba(var(--md-secondary-rgb),0.14), transparent 55%),
        linear-gradient(135deg, rgba(var(--md-primary-rgb),0.18), rgba(var(--md-secondary-rgb),0.10) 60%, transparent),
        var(--md-surface-container);
    border-bottom: 1px solid var(--md-outline);
}
.md-quiz-header::before {
    content:""; position:absolute; inset:0;
    background: repeating-linear-gradient(55deg, rgba(255,255,255,0.016) 0 1px, transparent 1px 16px);
    pointer-events:none;
}
.md-quiz-header-inner { position:relative; z-index:1; display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
.md-quiz-header-icon {
    width:60px; height:60px; border-radius:var(--md-shape-md);
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.36), rgba(var(--md-secondary-rgb),0.24));
    border: 1px solid rgba(var(--md-primary-rgb),0.38);
    display:flex; align-items:center; justify-content:center;
    font-size:28px; flex-shrink:0; box-shadow:var(--md-shadow-1);
}
.md-quiz-header-title {
    font-family:'Outfit',sans-serif!important;
    font-size:clamp(20px,2.6vw,26px); font-weight:900; line-height:1.1; margin:0 0 5px 0;
}
.md-quiz-header-sub { color:var(--md-soft); font-size:13px; line-height:1.5; }
.md-quiz-badge {
    margin-left:auto; padding:7px 15px; border-radius:var(--md-shape-pill);
    background:rgba(var(--md-primary-rgb),0.14); border:1px solid rgba(var(--md-primary-rgb),0.38);
    color:#14b8a6; font-size:12px; font-weight:900; letter-spacing:0.06em; text-transform:uppercase;
    white-space:nowrap;
}

/* ── Progress bar strip ── */
.md-quiz-progress-bar {
    height:5px; width:100%;
    background:rgba(148,163,184,0.18);
    position:relative;
}
.md-quiz-progress-fill {
    height:100%; border-radius:0;
    background: linear-gradient(90deg, #006a6a, #14b8a6);
    transition: width 600ms var(--md-ease-emphasized);
}

/* ── Quiz body ── */
.md-quiz-body { padding: 24px 28px 28px 28px; }

/* ── Question card ── */
.md-quiz-card {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-lg);
    overflow: hidden;
    margin-bottom: 16px;
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-1);
    transition: transform var(--md-dur-short) var(--md-ease-emphasized),
                box-shadow var(--md-dur-short) var(--md-ease-emphasized);
    animation: md-fade-up 400ms var(--md-ease-emphasized) both;
}
.md-quiz-card:hover { transform: translateY(-2px); box-shadow: var(--md-shadow-2); }
.md-quiz-card-header {
    padding: 16px 20px 0 20px;
    display: flex; align-items: center; gap: 12px;
}
.md-quiz-card-num {
    width:32px; height:32px; border-radius:var(--md-shape-pill);
    background:linear-gradient(135deg,rgba(var(--md-primary-rgb),0.3),rgba(var(--md-secondary-rgb),0.2));
    border:1px solid rgba(var(--md-primary-rgb),0.38);
    color:#14b8a6; font-size:13px; font-weight:900;
    display:flex; align-items:center; justify-content:center; flex-shrink:0;
}
.md-quiz-card-q {
    font-family:'Outfit',sans-serif!important;
    font-size:15px; font-weight:800; line-height:1.4; flex:1;
}
.md-quiz-card-opts { padding: 12px 20px 18px 20px; }

/* ── Result score card ── */
.md-quiz-result-hero {
    position:relative; overflow:hidden;
    border-radius:var(--md-shape-xl);
    padding:36px 28px;
    text-align:center;
    background:
        radial-gradient(700px 280px at 50% 0%, rgba(var(--md-primary-rgb),0.20), transparent 60%),
        linear-gradient(135deg, rgba(var(--md-primary-rgb),0.16), rgba(var(--md-secondary-rgb),0.10)),
        var(--md-surface-container);
    border:2px solid rgba(var(--md-primary-rgb),0.42);
    box-shadow: var(--md-shadow-3), var(--md-shadow-glow);
    margin-bottom:24px;
    animation: md-scale-in 500ms var(--md-ease-emphasized) both;
}
.md-quiz-result-hero::before {
    content:""; position:absolute; inset:0;
    background:repeating-linear-gradient(45deg,rgba(255,255,255,0.015) 0 1px,transparent 1px 14px);
    pointer-events:none;
}
.md-quiz-result-icon { font-size:52px; margin-bottom:12px; animation:md-heartbeat 2.5s ease-in-out infinite; position:relative; z-index:1; }
.md-quiz-result-score {
    font-family:'Outfit',sans-serif!important;
    font-size:72px; font-weight:900; color:#14b8a6; line-height:0.9;
    position:relative; z-index:1;
}
.md-quiz-result-denom {
    font-family:'Outfit',sans-serif!important;
    font-size:28px; font-weight:700; color:rgba(20,184,166,0.60);
}
.md-quiz-result-grade {
    font-family:'Outfit',sans-serif!important;
    font-size:22px; font-weight:900; margin-top:12px; position:relative; z-index:1;
}
.md-quiz-result-pct {
    color:var(--md-soft); font-size:14px; margin-top:6px; position:relative; z-index:1;
}
.md-quiz-result-bar {
    height:10px; border-radius:var(--md-shape-pill);
    background:rgba(148,163,184,0.18); overflow:hidden;
    margin:16px auto 0 auto; max-width:280px; position:relative; z-index:1;
}
.md-quiz-result-bar-fill {
    height:100%; border-radius:inherit;
    background:linear-gradient(90deg,#006a6a,#14b8a6);
    transition:width 900ms var(--md-ease-emphasized);
}

/* ── Review answer cards ── */
.md-quiz-review-card {
    border-radius:var(--md-shape-lg);
    padding:16px 18px;
    margin-bottom:12px;
    animation:md-fade-up 380ms var(--md-ease-emphasized) both;
    transition:transform var(--md-dur-short) var(--md-ease-emphasized);
}
.md-quiz-review-card:hover { transform:translateX(4px); }
.md-quiz-review-card.correct {
    background:rgba(20,184,166,0.07);
    border:1px solid rgba(20,184,166,0.30);
}
.md-quiz-review-card.wrong {
    background:rgba(239,68,68,0.07);
    border:1px solid rgba(239,68,68,0.28);
}
.md-quiz-review-header { display:flex; align-items:center; gap:10px; margin-bottom:8px; }
.md-quiz-review-icon { font-size:18px; flex-shrink:0; }
.md-quiz-review-q {
    font-family:'Outfit',sans-serif!important;
    font-size:14px; font-weight:800; line-height:1.35; flex:1;
}
.md-quiz-review-answers { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
.md-quiz-ans-pill {
    padding:4px 12px; border-radius:var(--md-shape-pill);
    font-size:12px; font-weight:800;
}
.md-quiz-ans-pill.your-correct {
    background:rgba(20,184,166,0.18); border:1px solid rgba(20,184,166,0.40); color:#14b8a6;
}
.md-quiz-ans-pill.your-wrong {
    background:rgba(239,68,68,0.14); border:1px solid rgba(239,68,68,0.38); color:#ef4444;
}
.md-quiz-ans-pill.correct-ans {
    background:rgba(20,184,166,0.12); border:1px solid rgba(20,184,166,0.30); color:#14b8a6;
}
.md-quiz-review-exp {
    color:var(--md-soft); font-size:12.5px; line-height:1.55; font-style:italic;
    border-left:2px solid rgba(148,163,184,0.30); padding-left:10px;
}
</style>
""",
        unsafe_allow_html=True,
    )

    # ── Quiz container ──────────────────────────────────────────────────────────
    st.markdown('<div class="md-quiz-wrap">', unsafe_allow_html=True)

    # Header
    st.markdown(
        f"""
<div class="md-quiz-header">
  <div class="md-quiz-header-inner">
    <div class="md-quiz-header-icon">🧠</div>
    <div style="flex:1; min-width:0;">
      <div class="md-quiz-header-title">Heart Health Knowledge Quiz</div>
      <div class="md-quiz-header-sub">Test your cardiovascular knowledge across {total} evidence-based questions.</div>
    </div>
    <div class="md-quiz-badge">{total} Questions</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Results view ─────────────────────────────────────────────────────────
    if st.session_state.quiz_score is not None:
        score = st.session_state.quiz_score
        pct = int(score / total * 100)
        emoji = "🏆" if pct >= 80 else ("👏" if pct >= 60 else "📚")
        grade = "Excellent!" if pct >= 80 else ("Great Job!" if pct >= 60 else "Keep Learning!")
        grade_color = "#14b8a6" if pct >= 80 else ("#f59e0b" if pct >= 60 else "#ef4444")

        st.markdown(
            f"""
<div class="md-quiz-progress-bar">
  <div class="md-quiz-progress-fill" style="width:100%;"></div>
</div>
<div class="md-quiz-body">
  <div class="md-quiz-result-hero">
    <div class="md-quiz-result-icon">{emoji}</div>
    <div>
      <span class="md-quiz-result-score">{score}</span>
      <span class="md-quiz-result-denom">/{total}</span>
    </div>
    <div class="md-quiz-result-grade" style="color:{grade_color};">{grade}</div>
    <div class="md-quiz-result-pct">{pct}% correct</div>
    <div class="md-quiz-result-bar">
      <div class="md-quiz-result-bar-fill" style="width:{pct}%; background:linear-gradient(90deg,{'#006a6a,#14b8a6' if pct>=80 else ('#b45309,#f59e0b' if pct>=60 else '#dc2626,#ef4444')});"></div>
    </div>
  </div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-family:\'Outfit\',sans-serif!important; font-size:15px; font-weight:900; margin-bottom:14px; display:flex; align-items:center; gap:8px;">📋 <span>Answer Review</span></div>',
            unsafe_allow_html=True,
        )

        for i, q in enumerate(QUIZ_QUESTIONS):
            user_ans = st.session_state.quiz_answers.get(i)
            is_correct = user_ans == q["answer"]
            icon = "✅" if is_correct else "❌"
            card_cls = "correct" if is_correct else "wrong"

            your_pill_cls = "your-correct" if is_correct else "your-wrong"
            your_label = "Your answer" if is_correct else "Your answer (wrong)"

            st.markdown(
                f"""
<div class="md-quiz-review-card {card_cls}" style="animation-delay:{i*50}ms;">
  <div class="md-quiz-review-header">
    <span class="md-quiz-review-icon">{icon}</span>
    <div class="md-quiz-review-q">Q{i+1}. {escape(q['q'])}</div>
  </div>
  <div class="md-quiz-review-answers">
    <span class="md-quiz-ans-pill {your_pill_cls}">{escape(str(user_ans or 'Not answered'))}</span>
    {'<span class="md-quiz-ans-pill correct-ans">✓ ' + escape(q["answer"]) + '</span>' if not is_correct else ''}
  </div>
  <div class="md-quiz-review-exp">{escape(q['explanation'])}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown('</div></div>', unsafe_allow_html=True)  # close body + wrap

        st.button("🔄 Retake Quiz", use_container_width=True, key="retake_quiz",
                  on_click=lambda: st.session_state.update({"quiz_score": None, "quiz_answers": {}}))
        return

    # ── Quiz form view ────────────────────────────────────────────────────────
    st.markdown('<div class="md-quiz-progress-bar"><div class="md-quiz-progress-fill" style="width:0%;"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="md-quiz-body">', unsafe_allow_html=True)

    with st.form("quiz_form"):
        answers = {}
        for i, q in enumerate(QUIZ_QUESTIONS):
            st.markdown(
                f"""
<div class="md-quiz-card" style="animation-delay:{i*55}ms;">
  <div class="md-quiz-card-header">
    <div class="md-quiz-card-num">{i+1}</div>
    <div class="md-quiz-card-q">{escape(q["q"])}</div>
  </div>
  <div class="md-quiz-card-opts">
""",
                unsafe_allow_html=True,
            )
            ans = st.radio(
                f"Answer Q{i+1}", q["options"],
                key=f"quiz_q_{i}", index=None,
                label_visibility="collapsed",
            )
            st.markdown('</div></div>', unsafe_allow_html=True)
            answers[i] = ans

        submitted = st.form_submit_button("🧠 Submit Quiz", use_container_width=True)
        if submitted:
            score = sum(1 for i, q in enumerate(QUIZ_QUESTIONS) if answers.get(i) == q["answer"])
            st.session_state.quiz_score = score
            st.session_state.quiz_answers = answers
            st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)  # close body + wrap


# =============================================================================
# HEALTH GOAL TRACKER
# =============================================================================
PRESET_GOALS = [
    {"title": "Exercise 3x this week", "category": "Exercise", "target_days": 7},
    {"title": "Sleep 7+ hours for 5 days", "category": "Sleep", "target_days": 7},
    {"title": "No alcohol for 2 weeks", "category": "Alcohol", "target_days": 14},
    {"title": "Eat vegetables at every meal for 1 week", "category": "Diet", "target_days": 7},
    {"title": "Walk 10,000 steps daily for 5 days", "category": "Exercise", "target_days": 5},
    {"title": "Meditate 10 minutes daily for 7 days", "category": "Mental", "target_days": 7},
    {"title": "Quit smoking for 30 days", "category": "Smoking", "target_days": 30},
    {"title": "Drink 8 glasses of water daily for 1 week", "category": "Hydration", "target_days": 7},
]

GOAL_CATEGORY_ICONS = {
    "Exercise": "🏃", "Sleep": "😴", "Alcohol": "🍷",
    "Diet": "🥗", "Mental": "🧘", "Smoking": "🚭",
    "Hydration": "💧", "Custom": "🎯",
}


def render_goal_tracker():
    st.markdown("#### 🎯 Health Goal Tracker")
    st.caption("Set personal heart-health goals and track your progress.")

    goals = st.session_state.health_goals

    with st.expander("➕ Add a New Goal", expanded=False):
        use_preset = st.toggle("Use a preset goal", value=True, key="goal_preset_toggle")
        if use_preset:
            preset_titles = [g["title"] for g in PRESET_GOALS]
            chosen = st.selectbox("Choose a preset", preset_titles, key="goal_preset_sel")
            preset = next(g for g in PRESET_GOALS if g["title"] == chosen)
            if st.button("Add This Goal", key="add_preset_goal"):
                new_goal = {
                    "id": datetime.datetime.now().isoformat(),
                    "title": preset["title"],
                    "category": preset["category"],
                    "target_days": preset["target_days"],
                    "progress": 0,
                    "done": False,
                    "added": datetime.datetime.now().strftime("%Y-%m-%d"),
                }
                st.session_state.health_goals.append(new_goal)
                st.success("Goal added!")
                st.rerun()
        else:
            g_title = st.text_input("Goal title", placeholder="e.g. Run 5km three times this week", key="goal_custom_title")
            g_cat = st.selectbox("Category", list(GOAL_CATEGORY_ICONS.keys()), key="goal_cat")
            g_days = st.number_input("Target duration (days)", 1, 90, 7, key="goal_days")
            if st.button("Add Custom Goal", key="add_custom_goal") and g_title.strip():
                new_goal = {
                    "id": datetime.datetime.now().isoformat(),
                    "title": g_title.strip(),
                    "category": g_cat,
                    "target_days": int(g_days),
                    "progress": 0,
                    "done": False,
                    "added": datetime.datetime.now().strftime("%Y-%m-%d"),
                }
                st.session_state.health_goals.append(new_goal)
                st.success("Custom goal added!")
                st.rerun()

    if not goals:
        st.markdown(
            '<div class="md-callout md-callout-info"><span class="md-callout-icon">💡</span>No goals yet. Add a goal above to start tracking your heart-health journey.</div>',
            unsafe_allow_html=True,
        )
        return

    active = [g for g in goals if not g.get("done")]
    completed = [g for g in goals if g.get("done")]

    if active:
        st.markdown("**Active Goals**")
        for idx, goal in enumerate(goals):
            if goal.get("done"):
                continue
            icon = GOAL_CATEGORY_ICONS.get(goal.get("category", "Custom"), "🎯")
            pct = goal.get("progress", 0)
            col_g, col_p, col_btn = st.columns([4, 2, 1])
            with col_g:
                st.markdown(
                    f"""
<div class="md-goal-card">
  <div class="md-goal-title">{icon} {escape(goal['title'])}</div>
  <div class="md-goal-meta">Category: {escape(goal.get('category','—'))} · Target: {goal.get('target_days','—')} days · Added: {goal.get('added','—')}</div>
  <div class="md-goal-bar"><div class="md-goal-bar-fill" style="width:{pct}%;"></div></div>
  <div class="md-goal-pct">{pct}% complete</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            with col_p:
                new_pct = st.slider(
                    "Progress %", 0, 100, pct,
                    key=f"goal_pct_{goal['id']}",
                    label_visibility="collapsed",
                )
                if new_pct != pct:
                    goal["progress"] = new_pct
                    if new_pct == 100:
                        goal["done"] = True
                    st.rerun()
            with col_btn:
                if st.button("✓", key=f"goal_done_{goal['id']}", help="Mark complete"):
                    goal["progress"] = 100
                    goal["done"] = True
                    st.rerun()

    if completed:
        with st.expander(f"✅ Completed Goals ({len(completed)})", expanded=False):
            for goal in completed:
                icon = GOAL_CATEGORY_ICONS.get(goal.get("category", "Custom"), "🎯")
                st.markdown(
                    f'<div class="md-note-item"><div class="md-note-text">✅ {icon} <strong>{escape(goal["title"])}</strong></div><div class="md-note-tag">{escape(goal.get("category","—"))}</div></div>',
                    unsafe_allow_html=True,
                )
            if st.button("Clear completed goals", key="clear_done_goals"):
                st.session_state.health_goals = [g for g in goals if not g.get("done")]
                st.rerun()


# =============================================================================
# HEALTH JOURNAL / NOTES LOG
# =============================================================================
NOTE_TAGS = ["General", "Diet", "Exercise", "Sleep", "Medication", "Symptom", "Mood", "Goal"]


def render_notes_log():
    st.markdown("#### 📓 Health Journal")
    st.caption("Keep a personal log of health notes, observations, or reminders.")

    with st.form("note_form", clear_on_submit=True):
        nc1, nc2 = st.columns([4, 1])
        with nc1:
            note_text = st.text_area("New entry", placeholder="e.g. Felt short of breath after climbing stairs. Scheduled a checkup.", height=90, label_visibility="collapsed")
        with nc2:
            note_tag = st.selectbox("Tag", NOTE_TAGS, label_visibility="collapsed")
        submitted_note = st.form_submit_button("Add Entry", use_container_width=True)
        if submitted_note and note_text.strip():
            st.session_state.notes_log.append({
                "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "text": note_text.strip(),
                "tag": note_tag,
            })
            st.success("Entry saved.")
            st.rerun()

    notes = st.session_state.notes_log
    if not notes:
        st.markdown(
            '<div class="md-callout md-callout-info"><span class="md-callout-icon">📓</span>No journal entries yet. Add your first health note above.</div>',
            unsafe_allow_html=True,
        )
        return

    tags_present = sorted(set(n["tag"] for n in notes))
    filter_tag = st.selectbox("Filter by tag", ["All"] + tags_present, key="notes_filter")

    filtered = [n for n in reversed(notes) if filter_tag == "All" or n["tag"] == filter_tag]

    for i, note in enumerate(filtered):
        tag_color_map = {
            "Symptom": "#ef4444", "Diet": "#14b8a6", "Exercise": "#3b82f6",
            "Sleep": "#8b5cf6", "Medication": "#f59e0b", "Mood": "#ec4899",
            "Goal": "#006a6a", "General": "#94a3b8",
        }
        tag_color = tag_color_map.get(note["tag"], "#94a3b8")
        st.markdown(
            f"""
<div class="md-note-item">
  <div class="md-note-time">{escape(note['time'])}</div>
  <div class="md-note-text">{escape(note['text'])}</div>
  <span class="md-note-tag" style="border-color:{tag_color}30; color:{tag_color}; background:{tag_color}12;">{escape(note['tag'])}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    if notes:
        notes_df = pd.DataFrame(notes)
        csv_bytes = notes_df.to_csv(index=False).encode("utf-8")
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Export journal (CSV)", data=csv_bytes,
                file_name="health_journal.csv", mime="text/csv",
                use_container_width=True, key="notes_csv",
            )
        with dl2:
            if st.button("🗑️ Clear all entries", key="clear_notes"):
                st.session_state.notes_log = []
                st.rerun()


# =============================================================================
# GLOBAL HEART STATS SECTION
# =============================================================================
def render_global_stats():
    st.markdown(
        """
<div class="md-section-label">
  <div class="md-section-label-text">📊 Heart Disease — Key Facts</div>
  <div class="md-section-label-line"></div>
</div>
<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap:14px; margin-bottom:18px;">
  <div class="md-stat-card" style="animation-delay:0ms;">
    <div class="md-stat-icon">❤️</div>
    <div class="md-stat-num" style="color:#ef4444;">#1</div>
    <div class="md-stat-label">Leading cause of death globally (WHO, 2023)</div>
  </div>
  <div class="md-stat-card" style="animation-delay:60ms;">
    <div class="md-stat-icon">🌍</div>
    <div class="md-stat-num" style="color:#f59e0b;">17.9M</div>
    <div class="md-stat-label">Deaths from CVD annually worldwide</div>
  </div>
  <div class="md-stat-card" style="animation-delay:120ms;">
    <div class="md-stat-icon">⏱️</div>
    <div class="md-stat-num" style="color:#14b8a6;">80%</div>
    <div class="md-stat-label">Of premature CVD deaths are preventable</div>
  </div>
  <div class="md-stat-card" style="animation-delay:180ms;">
    <div class="md-stat-icon">🏃</div>
    <div class="md-stat-num" style="color:#3b82f6;">35%</div>
    <div class="md-stat-label">Risk reduction from regular exercise</div>
  </div>
  <div class="md-stat-card" style="animation-delay:240ms;">
    <div class="md-stat-icon">🚭</div>
    <div class="md-stat-num" style="color:#8b5cf6;">2–4×</div>
    <div class="md-stat-label">Higher CVD risk for smokers vs non-smokers</div>
  </div>
  <div class="md-stat-card" style="animation-delay:300ms;">
    <div class="md-stat-icon">💊</div>
    <div class="md-stat-num" style="color:#06b6d4;">50%</div>
    <div class="md-stat-label">Of heart attacks occur in people with normal cholesterol</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


# =============================================================================
# MAIN
# =============================================================================
model, encoder = load_assets()
init_session_state()
render_styles()
render_sidebar()
render_hero()
render_info_cards()

tab_assess, tab_insights, tab_tools, tab_wellness, tab_history, tab_about, tab_scan = st.tabs(
    ["🩺 Assessment", "📊 Insights", "🛠️ Tools", "💪 Wellness", "🕓 History", "ℹ️ About", "🎨 Scan Studio"]
)

# =================== ASSESSMENT TAB ===================
with tab_assess:
    col_left, col_right = st.columns([4, 2], gap="large")

    with col_left:
        st.markdown(
            """
<div class="md-form-title">Heart Disease Risk Assessment</div>
<div class="md-form-subtitle">Complete the form below. Your information is used only for this local assessment.</div>
""",
            unsafe_allow_html=True,
        )

        with st.form(key="assessment_form", clear_on_submit=False):
            with st.expander("👤 Demographics", expanded=True):
                r1, r2 = st.columns([1.3, 1])
                gender = r1.selectbox("Gender", ["female", "male", "nonbinary"], index=1, format_func=pretty_value)
                race = r2.selectbox(
                    "Race / Ethnicity",
                    ["white_only_non_hispanic", "black_only_non_hispanic", "asian_only_non_hispanic",
                     "american_indian_or_alaskan_native_only_non_hispanic", "multiracial_non_hispanic",
                     "hispanic", "native_hawaiian_or_other_pacific_islander_only_non_hispanic"],
                    index=0, format_func=pretty_value,
                )
                age_category = st.selectbox(
                    "Age group",
                    ["Age_18_to_24", "Age_25_to_29", "Age_30_to_34", "Age_35_to_39",
                     "Age_40_to_44", "Age_45_to_49", "Age_50_to_54", "Age_55_to_59",
                     "Age_60_to_64", "Age_65_to_69", "Age_70_to_74", "Age_75_to_79", "Age_80_or_older"],
                    index=4, format_func=pretty_value,
                )

            with st.expander("🩺 Medical History", expanded=False):
                m1, m2, m3 = st.columns([1.2, 1, 1])
                general_health = m1.selectbox("Overall health", ["excellent", "very_good", "good", "fair", "poor"], index=0, format_func=pretty_value)
                heart_attack = m1.selectbox("Diagnosed with heart attack?", ["yes", "no"], index=1, format_func=pretty_value)
                stroke = m1.selectbox("Diagnosed with stroke?", ["yes", "no"], index=1, format_func=pretty_value)
                kidney_disease = m1.selectbox("Kidney disease?", ["yes", "no"], index=1, format_func=pretty_value)
                diabetes = m1.selectbox("Diabetes?", ["yes", "no", "no_prediabetes", "yes_during_pregnancy"], index=1, format_func=pretty_value)

                asthma = m2.selectbox("Asthma status", ["never_asthma", "current_asthma", "former_asthma"], index=0, format_func=pretty_value)
                depressive_disorder = m2.selectbox("Depressive disorder?", ["yes", "no"], index=1, format_func=pretty_value)
                physical_health = m2.selectbox("Physical health not good", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], index=0, format_func=pretty_value)
                mental_health = m2.selectbox("Mental health not good", ["zero_days_not_good", "1_to_13_days_not_good", "14_plus_days_not_good"], index=0, format_func=pretty_value)
                walking = m2.selectbox("Difficulty walking/stairs?", ["yes", "no"], index=1, format_func=pretty_value)

                health_care_provider = m3.selectbox("Primary healthcare provider?", ["yes_only_one", "more_than_one", "no"], index=0, format_func=pretty_value)
                could_not_afford_to_see_doctor = m3.selectbox("Could not afford doctor?", ["yes", "no"], index=1, format_func=pretty_value)
                default_bmi_idx = 1
                bmi_options = ["underweight_bmi_less_than_18_5", "normal_weight_bmi_18_5_to_24_9",
                               "overweight_bmi_25_to_29_9", "obese_bmi_30_or_more"]
                if st.session_state.get("bmi_suggestion") in bmi_options:
                    default_bmi_idx = bmi_options.index(st.session_state["bmi_suggestion"])
                bmi = m3.selectbox("BMI category", bmi_options, index=default_bmi_idx, format_func=pretty_value)
                length_of_time_since_last_routine_checkup = m3.selectbox(
                    "Last routine checkup",
                    ["past_year", "past_2_years", "past_5_years", "5+_years_ago", "never"],
                    index=0, format_func=pretty_value,
                )

            with st.expander("🏃 Lifestyle", expanded=False):
                l1, l2 = st.columns([1.2, 1])
                smoking_status = l1.selectbox(
                    "Smoking status",
                    ["never_smoked", "former_smoker", "current_smoker_some_days", "current_smoker_every_day"],
                    index=0, format_func=pretty_value,
                )
                sleep_category = l1.selectbox(
                    "Sleep",
                    ["very_short_sleep_0_to_3_hours", "short_sleep_4_to_5_hours", "normal_sleep_6_to_8_hours",
                     "long_sleep_9_to_10_hours", "very_long_sleep_11_or_more_hours"],
                    index=2, format_func=pretty_value,
                )
                drinks_category = l2.selectbox(
                    "Alcohol per week",
                    ["did_not_drink", "very_low_consumption_0.01_to_1_drinks", "low_consumption_1.01_to_5_drinks",
                     "moderate_consumption_5.01_to_10_drinks", "high_consumption_10.01_to_20_drinks",
                     "very_high_consumption_more_than_20_drinks"],
                    index=0, format_func=pretty_value,
                )
                binge_drinking_status = l2.selectbox("Binge drinking past 30 days?", ["yes", "no"], index=1, format_func=pretty_value)
                exercise_status = st.selectbox("Exercised past 30 days?", ["yes", "no"], index=0, format_func=pretty_value)

            st.info("Your data is used only to calculate this assessment inside the app.")
            submit = st.form_submit_button("🔍 Assess My Risk", use_container_width=True)

        input_data = {
            "gender": gender, "race": race, "general_health": general_health,
            "health_care_provider": health_care_provider,
            "could_not_afford_to_see_doctor": could_not_afford_to_see_doctor,
            "length_of_time_since_last_routine_checkup": length_of_time_since_last_routine_checkup,
            "ever_diagnosed_with_heart_attack": heart_attack,
            "ever_diagnosed_with_a_stroke": stroke,
            "ever_told_you_had_a_depressive_disorder": depressive_disorder,
            "ever_told_you_have_kidney_disease": kidney_disease,
            "ever_told_you_had_diabetes": diabetes, "BMI": bmi,
            "difficulty_walking_or_climbing_stairs": walking,
            "physical_health_status": physical_health, "mental_health_status": mental_health,
            "asthma_Status": asthma, "smoking_status": smoking_status,
            "binge_drinking_status": binge_drinking_status,
            "exercise_status_in_past_30_Days": exercise_status,
            "age_category": age_category, "sleep_category": sleep_category,
            "drinks_category": drinks_category,
        }

        if submit:
            try:
                with st.spinner("Analyzing your heart health risk..."):
                    risk, input_encoded = predict_heart_disease_risk(input_data, model, encoder)
                    feature_importance_df = build_feature_importance_df(model, input_encoded)
                    feature_to_recommendation, pie_df = build_recommendations(input_data, feature_importance_df)
                    st.session_state.risk_result = risk
                    st.session_state.report_input_data = input_data.copy()
                    st.session_state.ftr_recs = feature_to_recommendation
                    st.session_state.pie_df = pie_df
                    st.session_state.recommendation_intro = get_recommendation_intro(risk)
                    st.session_state.feature_importance_df = feature_importance_df
                    st.session_state.history.append({
                        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "risk": risk,
                        "input": input_data.copy(),
                    })
                st.success("Assessment complete.")
                st.balloons()
            except Exception as e:
                st.error(e)

    # ── FIX 1: Results column — callout is now inside a constrained wrapper ──
    with col_right:
        with st.container(border=True):
            st.write("### Your Results")
            if st.session_state.risk_result is not None:
                saved_risk = st.session_state.risk_result
                saved_input_data = st.session_state.report_input_data or input_data
                saved_recs = st.session_state.ftr_recs
                level = get_risk_level(saved_risk)
                risk_class = get_risk_class(saved_risk)

                st.markdown(
                    f"""
<div class="md-result-hero">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:14px; flex-wrap:wrap;">
    <h2 style="color:{get_risk_color(saved_risk)}; margin:0;">{saved_risk:.1f}%</h2>
    <div class="risk-badge {risk_class}">{level} Risk</div>
  </div>
  <p>{escape(st.session_state.recommendation_intro)}</p>
</div>
""",
                    unsafe_allow_html=True,
                )

                st.plotly_chart(render_risk_gauge(saved_risk), use_container_width=True, config={"displayModeBar": False}, key="plchart_1")
                render_factor_chips(saved_input_data)

                ls = compute_lifestyle_score(saved_input_data)
                ls_color = "#14b8a6" if ls >= 70 else ("#f59e0b" if ls >= 45 else "#ef4444")
                st.markdown(
                    f"""
<div class="md-metric" style="margin-top:12px;">
  <div class="md-metric-label">Lifestyle Score</div>
  <div class="md-metric-value" style="color:{ls_color};">{ls}/100</div>
  <div class="md-progress"><span style="width:{ls}%; background:linear-gradient(90deg,{ls_color},{ls_color}99);"></span></div>
</div>
""",
                    unsafe_allow_html=True,
                )

                if saved_risk > 25:
                    render_recommendations(saved_recs)
                else:
                    st.markdown(
                        '<div class="md-callout md-callout-success"><span class="md-callout-icon">✅</span>Your estimated risk is low. Continue healthy routines and regular checkups.</div>',
                        unsafe_allow_html=True,
                    )

                if input_data != saved_input_data:
                    st.warning("You changed an answer after the last assessment. Click 'Assess My Risk' again before downloading a new report.")

                exp_c1, exp_c2 = st.columns(2)
                with exp_c1:
                    try:
                        pdf_bytes = create_heart_health_pdf(saved_input_data, saved_risk, saved_recs)
                        st.download_button(
                            label="📄 Download PDF",
                            data=pdf_bytes,
                            file_name="heart_health_report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                    except ImportError as e:
                        st.error(str(e))
                        st.code("pip install reportlab==4.4.10")
                with exp_c2:
                    json_payload = {
                        "generated": datetime.datetime.now().isoformat(),
                        "risk_percent": round(saved_risk, 2),
                        "risk_level": level,
                        "lifestyle_score": compute_lifestyle_score(saved_input_data),
                        "estimated_heart_age": heart_age_estimate(saved_input_data, saved_risk),
                        "input": saved_input_data,
                        "recommendations": list(saved_recs.values()),
                    }
                    st.download_button(
                        label="📦 Export JSON",
                        data=json.dumps(json_payload, indent=2),
                        file_name="heart_assessment.json",
                        mime="application/json",
                        use_container_width=True,
                    )

                st.divider()
                st.caption("This app is not a replacement for professional medical advice, diagnosis, or treatment.")
            else:
                # FIX 1 — constrained wrapper prevents overflow
                st.markdown(
                    """
<div style="overflow:hidden; width:100%; box-sizing:border-box; padding:2px 0;">
  <div class="md-callout md-callout-info" style="margin:0;">
    <span class="md-callout-icon">🩺</span>
    <div>No assessment yet. Complete the form and click <strong>Assess My Risk</strong> to see your results here.</div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

# =================== INSIGHTS TAB ===================
with tab_insights:
    if st.session_state.risk_result is None:
        st.markdown(
            '<div class="md-callout md-callout-info"><span class="md-callout-icon">📊</span>Run an assessment first to unlock the full insights dashboard.</div>',
            unsafe_allow_html=True,
        )
    else:
        saved_risk = st.session_state.risk_result
        saved_input_data = st.session_state.report_input_data

        render_metrics(saved_input_data, saved_risk)

        ic1, ic2 = st.columns([1, 1], gap="large")
        with ic1:
            with st.container(border=True):
                st.markdown("#### 🎯 Risk Gauge")
                st.plotly_chart(render_risk_gauge(saved_risk), use_container_width=True, config={"displayModeBar": False}, key="plchart_2")
            with st.container(border=True):
                st.markdown("#### 🧬 Top Risk Drivers")
                if st.session_state.feature_importance_df is not None:
                    st.plotly_chart(render_top_drivers(st.session_state.feature_importance_df), use_container_width=True, config={"displayModeBar": False}, key="plchart_3")

        with ic2:
            with st.container(border=True):
                st.markdown("#### 📈 You vs Reference Groups")
                st.plotly_chart(render_comparison_chart(saved_risk), use_container_width=True, config={"displayModeBar": False}, key="plchart_4")
            with st.container(border=True):
                st.markdown("#### 🥧 Contribution to Risk")
                if st.session_state.get("pie_df") is not None and not st.session_state.pie_df.empty:
                    pie_palette = ["#14b8a6", "#3f5f90", "#f59e0b", "#ef4444",
                                   "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16",
                                   "#f97316", "#94a3b8"]
                    fig = px.pie(
                        st.session_state.pie_df, names="Feature", values="Importance",
                        hole=0.52, color_discrete_sequence=pie_palette,
                    )
                    fig.update_traces(
                        textposition="inside", textinfo="percent",
                        insidetextorientation="radial",
                        textfont=dict(size=12, color="white"),
                        marker=dict(line=dict(color="rgba(15,23,42,0.6)", width=2)),
                        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
                    )
                    fig.update_layout(
                        height=380, margin=dict(l=10, r=10, t=20, b=60),
                        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5, font=dict(size=11, color="#cbd5e1")),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=True,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="plchart_5")
                else:
                    st.caption("Pie chart will appear after running an assessment with notable risk factors.")

        st.markdown('<div class="md-section-label"><div class="md-section-label-text">🕸️ Lifestyle Radar & Risk Forecast</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
        ra1, ra2 = st.columns(2, gap="large")
        with ra1:
            with st.container(border=True):
                st.markdown("#### 🕸️ Lifestyle Factor Radar")
                st.caption("Your profile vs healthy targets across 8 dimensions.")
                st.plotly_chart(render_lifestyle_radar(saved_input_data), use_container_width=True, config={"displayModeBar": False}, key="radar_chart")
        with ra2:
            with st.container(border=True):
                st.markdown("#### 📉 5-Year Risk Forecast")
                st.caption("Illustrative projection — with and without lifestyle improvements.")
                st.plotly_chart(render_risk_forecast(saved_risk, saved_input_data), use_container_width=True, config={"displayModeBar": False}, key="forecast_chart")
                st.markdown(
                    '<div class="md-callout md-callout-warn"><span class="md-callout-icon">⚠️</span>This forecast is a heuristic illustration only, not a clinical prediction. Actual risk depends on many factors.</div>',
                    unsafe_allow_html=True,
                )

        render_global_stats()

# =================== TOOLS TAB ===================
with tab_tools:
    t1, t2 = st.columns([1, 1], gap="large")
    with t1:
        with st.container(border=True):
            suggested = render_bmi_calculator()
            if st.button("Use this BMI in the form", use_container_width=True):
                st.session_state.bmi_suggestion = suggested
                st.success("BMI category set. It will pre-fill on the Assessment tab.")
    with t2:
        with st.container(border=True):
            if st.session_state.risk_result is None:
                st.markdown(
                    '<div class="md-callout md-callout-info"><span class="md-callout-icon">🧪</span>Run an assessment first to use the What-If Simulator.</div>',
                    unsafe_allow_html=True,
                )
            else:
                render_whatif(model, encoder, st.session_state.report_input_data, st.session_state.risk_result)

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        render_heart_qa()

    with st.container(border=True):
        st.markdown("#### 📚 Heart-Healthy Quick Guide")
        guide_cols = st.columns(3)
        guides = [
            ("🥗 Diet", "Mediterranean-style eating: vegetables, fruit, whole grains, legumes, fish, olive oil. Minimize ultra-processed foods."),
            ("🏃 Movement", "150 min/week moderate cardio + 2 strength sessions. Even short walks help."),
            ("😴 Sleep", "Aim for 7-9 hours. Consistent schedule matters more than perfection."),
            ("🚭 Tobacco", "Quitting at any age improves heart outcomes within months."),
            ("🍷 Alcohol", "Less is better. Avoid binge drinking entirely."),
            ("🧘 Stress", "Daily breathing, walking, social connection — chronic stress raises BP."),
        ]
        for i, (title, body) in enumerate(guides):
            with guide_cols[i % 3]:
                st.markdown(
                    f"""
<div class="md-info-card" style="min-height:160px;">
  <strong>{escape(title)}</strong>
  <span>{escape(body)}</span>
</div>
""",
                    unsafe_allow_html=True,
                )

# =================== WELLNESS TAB ===================
with tab_wellness:
    w1, w2 = st.columns([1, 1], gap="large")
    with w1:
        with st.container(border=True):
            render_goal_tracker()
    with w2:
        with st.container(border=True):
            render_notes_log()

    st.markdown('<div style="margin-top:8px;"></div>', unsafe_allow_html=True)
    with st.container(border=True):
        render_quiz()

# =================== HISTORY TAB ===================
with tab_history:
    if not st.session_state.history:
        st.markdown(
            '<div class="md-callout md-callout-info"><span class="md-callout-icon">🕓</span>No assessments saved yet. Complete one on the Assessment tab — entries appear here automatically.</div>',
            unsafe_allow_html=True,
        )
    else:
        hist_df = pd.DataFrame([
            {"Time": h.get("time", ""), "Risk %": round(float(h["risk"]), 2), "Level": get_risk_level(float(h["risk"]))}
            for h in st.session_state.history
            if isinstance(h, dict) and "risk" in h
        ])
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("#### 📈 Risk Trend Over Time")
                fig = px.line(
                    hist_df, x="Time", y="Risk %", markers=True,
                    color_discrete_sequence=["#14b8a6"],
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=10))
                fig.add_hrect(y0=0,  y1=25,  fillcolor="rgba(20,184,166,0.10)",  line_width=0)
                fig.add_hrect(y0=25, y1=40,  fillcolor="rgba(245,158,11,0.10)",  line_width=0)
                fig.add_hrect(y0=40, y1=70,  fillcolor="rgba(220,38,38,0.10)",   line_width=0)
                fig.add_hrect(y0=70, y1=100, fillcolor="rgba(127,29,29,0.15)",   line_width=0)
                fig.update_layout(
                    height=320, margin=dict(l=10, r=10, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key="plchart_6")

            if len(hist_df) > 1:
                with st.container(border=True):
                    st.markdown("#### 🥧 Risk Level Distribution")
                    level_counts = hist_df["Level"].value_counts().reset_index()
                    level_counts.columns = ["Level", "Count"]
                    level_colors = {"Low": "#14b8a6", "Moderate": "#f59e0b", "High": "#ef4444", "Very High": "#7f1d1d"}
                    fig2 = px.pie(
                        level_counts, names="Level", values="Count",
                        color="Level", color_discrete_map=level_colors, hole=0.45,
                    )
                    fig2.update_layout(
                        height=260, margin=dict(l=10, r=10, t=10, b=10),
                        paper_bgcolor="rgba(0,0,0,0)", showlegend=True,
                        legend=dict(font=dict(color="#94a3b8")),
                        font=dict(color="#94a3b8"),
                    )
                    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False}, key="hist_pie")

        with c2:
            with st.container(border=True):
                st.markdown("#### 🗂️ Assessment Log")
                st.dataframe(hist_df, use_container_width=True, hide_index=True)
                csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download history (CSV)",
                    data=csv_bytes,
                    file_name="heart_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            if not hist_df.empty:
                best = hist_df["Risk %"].min()
                worst = hist_df["Risk %"].max()
                avg = hist_df["Risk %"].mean()
                with st.container(border=True):
                    st.markdown("#### 📐 Risk Statistics")
                    st.markdown(
                        f"""
<div class="md-metric-grid" style="grid-template-columns:1fr;">
  <div class="md-metric">
    <div class="md-metric-label">Best (Lowest) Risk</div>
    <div class="md-metric-value" style="color:#14b8a6;">{best:.1f}%</div>
  </div>
  <div class="md-metric">
    <div class="md-metric-label">Worst (Highest) Risk</div>
    <div class="md-metric-value" style="color:#ef4444;">{worst:.1f}%</div>
  </div>
  <div class="md-metric">
    <div class="md-metric-label">Average Risk</div>
    <div class="md-metric-value" style="color:{get_risk_color(avg)};">{avg:.1f}%</div>
  </div>
  <div class="md-metric">
    <div class="md-metric-label">Total Assessments</div>
    <div class="md-metric-value">{len(hist_df)}</div>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )

# =================== ABOUT TAB — FIX 3: Full M3 Expressive redesign ===================
with tab_about:
    about_logo_html = '<div class="md-sb-avatar-box md-about-avatar">🫀</div>'

    st.markdown(f"""
<style>
/* ── About Hero ─────────────────────────────────────── */
.about-hero {{
    position:relative; overflow:hidden;
    border:1px solid var(--md-outline);
    border-radius:var(--md-shape-xl);
    padding:40px 36px;
    margin-bottom:28px;
    background:
        radial-gradient(900px 360px at 90% 10%, rgba(var(--md-tertiary-rgb),0.15), transparent 55%),
        radial-gradient(600px 280px at 10% 90%, rgba(var(--md-secondary-rgb),0.12), transparent 55%),
        linear-gradient(135deg, rgba(var(--md-primary-rgb),0.18) 0%, rgba(var(--md-secondary-rgb),0.10) 60%, transparent),
        var(--md-surface);
    box-shadow:var(--md-shadow-3), var(--md-shadow-glow);
    animation:md-fade-up 500ms var(--md-ease-emphasized) both;
}}
.about-hero::before {{
    content:""; position:absolute; inset:0;
    background:repeating-linear-gradient(60deg,rgba(255,255,255,0.014) 0 1px,transparent 1px 18px);
    pointer-events:none;
}}
.about-hero-grid {{
    display:grid; grid-template-columns:auto 1fr; gap:28px; align-items:center;
    position:relative; z-index:1;
}}
.about-badge {{
    display:inline-flex; padding:6px 14px; border-radius:999px;
    background:rgba(var(--md-primary-rgb),0.14); border:1px solid rgba(var(--md-primary-rgb),0.38);
    color:#14b8a6; font-size:11px; font-weight:900; letter-spacing:0.07em;
    text-transform:uppercase; margin-bottom:10px;
}}
.about-hero-title {{
    font-family:'Outfit',sans-serif!important;
    font-size:clamp(26px,3.5vw,42px); font-weight:900; line-height:1.05; margin:0 0 10px 0;
    background:linear-gradient(110deg,#ffffff 30%,rgba(255,255,255,0.55));
    -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}}
.about-hero-sub {{
    color:var(--md-soft); font-size:15px; line-height:1.65; max-width:640px;
}}
/* ── Feature grid ───────────────────────────────────── */
.about-features {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:13px; margin:0 0 28px 0;
}}
.about-feat {{
    border:1px solid var(--md-outline-variant);
    border-radius:var(--md-shape-md);
    padding:16px;
    background:var(--md-surface-container);
    box-shadow:var(--md-shadow-1);
    transition:transform var(--md-dur-short) var(--md-ease-emphasized),
               background var(--md-dur-short) var(--md-ease-emphasized);
    animation:md-fade-up 400ms var(--md-ease-emphasized) both;
}}
.about-feat:hover {{
    transform:translateY(-3px); background:var(--md-surface-container-high);
}}
.about-feat-icon {{ font-size:22px; margin-bottom:8px; }}
.about-feat-title {{
    font-family:'Outfit',sans-serif!important;
    font-size:14px; font-weight:800; margin-bottom:5px;
}}
.about-feat-desc {{ color:var(--md-soft); font-size:13px; line-height:1.5; }}
/* ── Creator card ───────────────────────────────────── */
.about-creator {{
    border:1px solid var(--md-outline);
    border-radius:var(--md-shape-lg);
    padding:24px;
    background:linear-gradient(135deg,rgba(var(--md-primary-rgb),0.13),rgba(var(--md-secondary-rgb),0.08)),
               var(--md-surface-container);
    box-shadow:var(--md-shadow-2);
    display:flex; gap:20px; align-items:flex-start;
    margin-bottom:20px;
}}
.about-creator-avatar {{
    width:72px; height:72px; min-width:72px;
    border-radius:var(--md-shape-md);
    background:linear-gradient(135deg,rgba(var(--md-primary-rgb),0.3),rgba(var(--md-secondary-rgb),0.2));
    display:flex; align-items:center; justify-content:center;
    font-size:30px; border:1px solid var(--md-outline-variant);
    box-shadow:var(--md-shadow-1);
}}
.about-creator-name {{
    font-family:'Outfit',sans-serif!important; font-size:20px; font-weight:900; margin-bottom:4px;
}}
.about-creator-bio {{ color:var(--md-soft); font-size:13px; line-height:1.6; margin-bottom:12px; }}
.about-creator-links {{ display:flex; gap:8px; flex-wrap:wrap; }}
.about-creator-link {{
    padding:7px 14px; border-radius:999px;
    border:1px solid var(--md-outline-variant);
    background:var(--md-surface);
    color:var(--md-soft)!important; font-size:13px; font-weight:700;
    text-decoration:none;
    transition:all var(--md-dur-short) var(--md-ease-emphasized);
    display:inline-block;
}}
.about-creator-link:hover {{
    background:rgba(var(--md-primary-rgb),0.12);
    border-color:rgba(var(--md-primary-rgb),0.38);
    color:#14b8a6!important; transform:translateY(-2px);
    text-decoration:none;
}}
/* ── Tech stack pills ───────────────────────────────── */
.about-tech {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
.about-tech-pill {{
    padding:6px 13px; border-radius:999px;
    border:1px solid var(--md-outline-variant);
    background:var(--md-surface-container);
    font-size:12px; font-weight:800; color:var(--md-soft);
    transition:transform var(--md-dur-short) var(--md-ease-emphasized);
}}
.about-tech-pill:hover {{ transform:translateY(-2px); }}
/* ── Resource cards ─────────────────────────────────── */
.about-resources {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:10px; margin-top:12px;
}}
.about-resource {{
    border:1px solid var(--md-outline-variant);
    border-radius:var(--md-shape-md);
    padding:14px 16px;
    background:var(--md-surface);
    text-decoration:none!important; color:inherit!important;
    display:flex; align-items:center; gap:12px;
    transition:all var(--md-dur-short) var(--md-ease-emphasized);
    box-shadow:var(--md-shadow-1);
}}
.about-resource:hover {{
    background:rgba(var(--md-primary-rgb),0.10);
    border-color:rgba(var(--md-primary-rgb),0.34);
    transform:translateY(-3px); box-shadow:var(--md-shadow-2);
    text-decoration:none!important;
}}
.about-resource-icon {{
    width:40px; height:40px; min-width:40px; border-radius:var(--md-shape-sm);
    background:linear-gradient(135deg,rgba(var(--md-primary-rgb),0.24),rgba(var(--md-secondary-rgb),0.16));
    display:flex; align-items:center; justify-content:center; font-size:18px;
}}
.about-resource-name {{ font-weight:800; font-size:13px; line-height:1.3; }}
.about-resource-sub  {{ color:var(--md-soft); font-size:11px; margin-top:2px; }}
@media(max-width:700px){{
    .about-hero-grid {{ grid-template-columns:1fr; }}
    .about-creator {{ flex-direction:column; }}
    .about-hero {{ padding:22px 18px; }}
}}
</style>

<div class="about-hero">
  <div class="about-hero-grid">
    <div>{about_logo_html}</div>
    <div>
      <div class="about-badge">Open Source · Educational</div>
      <h2 class="about-hero-title">SmartHealthCare<br/>for Early Diagnosis</h2>
      <p class="about-hero-sub">
        An AI-powered heart disease risk assessment platform combining a LightGBM ensemble model with
        SHAP-based explanations, Material 3 Expressive design, and actionable personalised health insights.
      </p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Features grid ─────────────────────────────────────────────────────────
    st.markdown('<div class="md-section-label"><div class="md-section-label-text">✨ Features</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
    features = [
        ("🤖", "LightGBM Ensemble", "Trained tree-based model with cross-validated accuracy for heart disease classification."),
        ("🔍", "SHAP Explainability", "Feature contributions ranked by SHAP values — understand exactly what drives your risk."),
        ("🕸️", "Lifestyle Radar", "Spider chart across 8 health dimensions versus healthy targets."),
        ("📉", "5-Year Forecast", "Illustrative risk trajectory with and without lifestyle improvements."),
        ("🧪", "What-If Simulator", "Adjust modifiable factors and see real-time risk recalculation."),
        ("🧮", "BMI Calculator", "Metric & imperial, with visual scale and health implications."),
        ("🎯", "Goal Tracker", "Preset and custom health goals with progress tracking."),
        ("📓", "Health Journal", "Tagged daily entries with CSV export."),
        ("💬", "Q&A Widget", "Pre-loaded heart health educational content."),
        ("🧠", "Knowledge Quiz", "7-question quiz with explanations and scoring."),
        ("📄", "PDF & JSON Export", "Download your full report or raw assessment data."),
        ("🎨", "Material 3 Expressive", "Outfit + DM Sans, motion, soft shadows, expressive shape scale."),
    ]
    feat_html = '<div class="about-features">'
    for delay_idx, (fi, ft, fd) in enumerate(features):
        feat_html += f'<div class="about-feat" style="animation-delay:{delay_idx*35}ms;"><div class="about-feat-icon">{fi}</div><div class="about-feat-title">{escape(ft)}</div><div class="about-feat-desc">{escape(fd)}</div></div>'
    feat_html += '</div>'
    st.markdown(feat_html, unsafe_allow_html=True)

    # ── Two-column section ────────────────────────────────────────────────────
    ab1, ab2 = st.columns([1.2, 1], gap="large")

    with ab1:
        st.markdown('<div class="md-section-label"><div class="md-section-label-text">👤 Creator</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
        st.markdown(f"""
<div class="about-creator">
  <div class="about-creator-avatar">👨‍💻</div>
  <div>
    <div class="about-creator-name">{escape(CREATOR_NAME)}</div>
    <div class="about-creator-bio">
      Passionate about applying AI to healthcare for real-world impact. This project explores
      early-stage cardiovascular risk identification through interpretable machine learning,
      accessible to anyone without medical expertise.
    </div>
    <div class="about-creator-links">
      <a class="about-creator-link" href="https://github.com/YatinSharma1303/" target="_blank">🐙 GitHub</a>
      <a class="about-creator-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 LinkedIn</a>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown('<div class="md-section-label"><div class="md-section-label-text">🛠️ Tech Stack</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
        tech_items = ["Python 3.11", "Streamlit", "LightGBM", "SHAP", "Plotly", "Pandas", "NumPy", "ReportLab", "Scikit-learn", "Pillow"]
        pills_html = '<div class="about-tech">' + "".join(f'<span class="about-tech-pill">{t}</span>' for t in tech_items) + '</div>'
        st.markdown(pills_html, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="md-section-label"><div class="md-section-label-text">⚠️ Disclaimer</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="md-callout md-callout-warn" style="margin-top:0;">
  <span class="md-callout-icon">⚕️</span>
  <div>This app is for <strong>educational purposes only</strong>. It is not a medical device, diagnosis, or clinical decision-support tool. Always consult a qualified healthcare professional for advice, diagnosis, and treatment decisions.</div>
</div>
""", unsafe_allow_html=True)

    with ab2:
        st.markdown('<div class="md-section-label"><div class="md-section-label-text">🌐 Resources</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
        resources_data = [
            ("🌐", "WHO", "Cardiovascular Diseases", "https://www.who.int/health-topics/cardiovascular-diseases"),
            ("❤️", "AHA", "American Heart Association", "https://www.heart.org"),
            ("🏥", "CDC", "Heart Disease Info", "https://www.cdc.gov/heart-disease/"),
            ("📚", "NHLBI", "Heart Health Topics", "https://www.nhlbi.nih.gov/health-topics/heart-disease"),
        ]
        res_html = '<div class="about-resources">'
        for r_icon, r_name, r_sub, r_url in resources_data:
            res_html += f'<a class="about-resource" href="{r_url}" target="_blank"><div class="about-resource-icon">{r_icon}</div><div><div class="about-resource-name">{escape(r_name)}</div><div class="about-resource-sub">{escape(r_sub)}</div></div></a>'
        res_html += '</div>'
        st.markdown(res_html, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px;"></div>', unsafe_allow_html=True)
        st.markdown('<div class="md-section-label"><div class="md-section-label-text">💬 Feedback</div><div class="md-section-label-line"></div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            with st.form("feedback_form"):
                rating = st.slider("How useful was this app?", 1, 5, 4)
                stars = "⭐" * rating
                comment = st.text_area("Comments (optional)", placeholder="What worked well? What could be better?", height=100)
                sent = st.form_submit_button("Send Feedback", use_container_width=True)
                if sent:
                    st.success(f"{stars} Thanks for the {rating}/5 rating! Feedback recorded for this session.")



    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)
    render_global_stats()


# =================== FOOTER — FIX 2: Full M3 Expressive footer ===================
st.markdown("""
<style>
.md-footer {
    margin-top: 52px;
    border-top: 1px solid var(--md-outline);
    padding: 32px 0 28px 0;
}
.md-footer-top {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
.md-footer-brand {
    display: flex;
    align-items: center;
    gap: 14px;
}
.md-footer-avatar.md-sb-avatar-box {
    width: 52px; height: 52px; min-width: 52px;
    border-radius: var(--md-shape-md);
    font-size: 24px;
    box-shadow: var(--md-shadow-1);
    animation: none;
    margin-bottom: 0;
    flex-shrink: 0;
}
.md-footer-avatar.md-sb-avatar-box::after { display: none; }
.md-footer-brand-name {
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px; font-weight: 900; line-height: 1.2;
    max-width: 200px;
}
.md-footer-brand-sub {
    font-size: 12px; color: var(--md-soft); margin-top: 3px;
}
.md-footer-links {
    display: flex; flex-wrap: wrap; gap: 8px;
    justify-content: center;
    margin-bottom: 14px;
}
.md-footer-link {
    padding: 8px 16px; border-radius: var(--md-shape-pill);
    border: 1px solid var(--md-outline-variant);
    background: var(--md-surface);
    color: var(--md-soft) !important;
    font-size: 13px; font-weight: 700;
    text-decoration: none !important;
    transition: all var(--md-dur-short) var(--md-ease-emphasized);
    display: inline-flex; align-items: center; gap: 5px;
}
.md-footer-link:hover {
    background: rgba(var(--md-primary-rgb), 0.12);
    border-color: rgba(var(--md-primary-rgb), 0.38);
    color: #14b8a6 !important;
    transform: translateY(-2px);
    text-decoration: none !important;
}
.md-footer-meta {
    text-align: center;
    color: var(--md-soft);
    font-size: 12px;
    line-height: 1.75;
    margin-bottom: 16px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.md-footer-heart {
    color: #ef4444;
    display: inline-block;
    animation: md-heartbeat 2.5s ease-in-out infinite;
}
.md-footer-version {
    display: inline-block;
    padding: 3px 10px;
    border-radius: var(--md-shape-pill);
    background: rgba(var(--md-primary-rgb),0.12);
    border: 1px solid rgba(var(--md-primary-rgb),0.26);
    font-size: 11px; font-weight: 800; color: #14b8a6;
    vertical-align: middle;
}
.md-footer-disclaimer {
    margin-top: 20px;
    padding: 13px 18px;
    border-radius: var(--md-shape-md);
    background: rgba(var(--md-warning-rgb), 0.07);
    border: 1px solid rgba(var(--md-warning-rgb), 0.22);
    color: var(--md-soft);
    font-size: 12px; line-height: 1.6;
    text-align: center;
}
@media (max-width: 760px) {
    .md-footer-brand { justify-content: center; }
    .md-footer-meta  { white-space: normal; }
}
</style>
""", unsafe_allow_html=True)

# ── NEW: Additional M3 Expressive feature CSS ──────────────────────────────────
st.markdown("""
<style>
/* ══════════════════════════════════════════════════════
   DAILY TIP BANNER — dismissible floating ribbon
   ══════════════════════════════════════════════════════ */
.md-tip-banner {
    position: relative;
    padding: 14px 52px 14px 18px;
    border-radius: var(--md-shape-md);
    background: linear-gradient(110deg,
        rgba(var(--md-primary-rgb),0.18) 0%,
        rgba(var(--md-secondary-rgb),0.12) 60%,
        rgba(var(--md-primary-rgb),0.08) 100%);
    border: 1px solid rgba(var(--md-primary-rgb),0.32);
    box-shadow: var(--md-shadow-1);
    margin-bottom: 16px;
    animation: md-fade-up 500ms var(--md-ease-emphasized) both;
    overflow: hidden;
}
.md-tip-banner::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
    background: linear-gradient(180deg, #14b8a6, #006a6a);
    border-radius: 0 0 0 0;
}
.md-tip-banner-label {
    font-size: 10px; font-weight: 900; color: #14b8a6;
    text-transform: uppercase; letter-spacing: 0.09em; margin-bottom: 5px;
}
.md-tip-banner-text {
    font-size: 14px; line-height: 1.55; color: var(--md-soft);
}
.md-tip-shine {
    position: absolute; inset: 0;
    background: repeating-linear-gradient(90deg, transparent 0, rgba(255,255,255,0.025) 1px, transparent 2px 28px);
    pointer-events: none;
}

/* ══════════════════════════════════════════════════════
   FORM STEP PROGRESS STEPPER
   ══════════════════════════════════════════════════════ */
.md-stepper {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 22px;
    padding: 14px 20px;
    border-radius: var(--md-shape-md);
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    box-shadow: var(--md-shadow-1);
}
.md-step {
    display: flex;
    align-items: center;
    gap: 9px;
    flex: 1;
    min-width: 0;
}
.md-step-circle {
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px; font-weight: 900;
    flex-shrink: 0;
    transition: all var(--md-dur-med) var(--md-ease-emphasized);
}
.md-step-circle.active {
    background: linear-gradient(135deg, #006a6a, #14b8a6);
    color: white;
    box-shadow: 0 0 0 4px rgba(20,184,166,0.22);
}
.md-step-circle.done {
    background: rgba(20,184,166,0.18);
    border: 1.5px solid rgba(20,184,166,0.50);
    color: #14b8a6;
}
.md-step-circle.idle {
    background: rgba(148,163,184,0.10);
    border: 1.5px solid rgba(148,163,184,0.22);
    color: var(--md-soft);
}
.md-step-label {
    font-size: 12px; font-weight: 700;
    color: var(--md-soft);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.md-step-label.active { color: #14b8a6; font-weight: 900; }
.md-step-connector {
    height: 2px;
    flex: 1;
    background: rgba(148,163,184,0.18);
    margin: 0 8px;
    border-radius: 1px;
    min-width: 12px;
}
.md-step-connector.done {
    background: linear-gradient(90deg, #14b8a6, rgba(20,184,166,0.40));
}
@media (max-width: 680px) {
    .md-step-label { display: none; }
    .md-step-connector { min-width: 8px; margin: 0 4px; }
}

/* ══════════════════════════════════════════════════════
   HEART HEALTH DAILY CHECKLIST
   ══════════════════════════════════════════════════════ */
.md-checklist-wrap {
    border: 1px solid var(--md-outline);
    border-radius: var(--md-shape-lg);
    overflow: hidden;
    box-shadow: var(--md-shadow-1);
    animation: md-fade-up 420ms var(--md-ease-emphasized) both;
}
.md-checklist-header {
    padding: 16px 20px 14px 20px;
    background: linear-gradient(110deg,
        rgba(var(--md-primary-rgb),0.18),
        rgba(var(--md-secondary-rgb),0.10) 70%,
        transparent);
    border-bottom: 1px solid var(--md-outline-variant);
    display: flex; align-items: center; gap: 12px;
}
.md-checklist-header-icon {
    width: 44px; height: 44px;
    border-radius: var(--md-shape-sm);
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.30), rgba(var(--md-secondary-rgb),0.18));
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
}
.md-checklist-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 17px; font-weight: 900; line-height: 1.2;
}
.md-checklist-sub { color: var(--md-soft); font-size: 12px; margin-top: 2px; }
.md-checklist-body { padding: 14px 20px 20px 20px; }
.md-checklist-score {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 14px;
    border-radius: var(--md-shape-md);
    background: var(--md-surface);
    border: 1px solid var(--md-outline-variant);
    margin-bottom: 14px;
}
.md-checklist-score-num {
    font-family: 'Outfit', sans-serif !important;
    font-size: 28px; font-weight: 900; line-height: 1;
    color: #14b8a6;
}
.md-checklist-score-label { color: var(--md-soft); font-size: 12px; line-height: 1.4; }
.md-checklist-score-bar {
    flex: 1; height: 8px;
    border-radius: var(--md-shape-pill);
    background: rgba(148,163,184,0.18); overflow: hidden;
}
.md-checklist-score-fill {
    height: 100%; border-radius: inherit;
    background: linear-gradient(90deg, #006a6a, #14b8a6);
    transition: width 700ms var(--md-ease-emphasized);
}

/* ══════════════════════════════════════════════════════
   ANIMATED SECTION DIVIDER (gradient pulse line)
   ══════════════════════════════════════════════════════ */
.md-gradient-divider {
    height: 2px;
    border-radius: 1px;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(var(--md-primary-rgb),0.50) 25%,
        rgba(var(--md-secondary-rgb),0.60) 50%,
        rgba(var(--md-primary-rgb),0.50) 75%,
        transparent 100%);
    background-size: 200% 100%;
    animation: md-shine 5s linear infinite;
    margin: 22px 0;
}

/* ══════════════════════════════════════════════════════
   PULSE RING around risk result percentage
   ══════════════════════════════════════════════════════ */
@keyframes md-ring-pulse {
    0%   { transform: scale(1);   opacity: 0.60; }
    50%  { transform: scale(1.10); opacity: 0.25; }
    100% { transform: scale(1);   opacity: 0.60; }
}
.md-risk-ring-wrap {
    position: relative; display: inline-flex;
    align-items: center; justify-content: center;
}
.md-risk-ring {
    position: absolute; inset: -10px;
    border-radius: 50%;
    border: 3px solid currentColor;
    animation: md-ring-pulse 2.4s ease-in-out infinite;
    pointer-events: none;
}

/* ══════════════════════════════════════════════════════
   HISTORY DELTA BADGE
   ══════════════════════════════════════════════════════ */
.md-delta-badge {
    display: inline-flex; align-items: center; gap: 3px;
    padding: 3px 9px; border-radius: var(--md-shape-pill);
    font-size: 11px; font-weight: 900;
}
.md-delta-up   { background: rgba(239,68,68,0.14);   border: 1px solid rgba(239,68,68,0.32);   color: #ef4444; }
.md-delta-down { background: rgba(20,184,166,0.14);  border: 1px solid rgba(20,184,166,0.32);  color: #14b8a6; }
.md-delta-same { background: rgba(148,163,184,0.12); border: 1px solid rgba(148,163,184,0.24); color: #94a3b8; }

/* ══════════════════════════════════════════════════════
   QUICK SNAPSHOT CARD (sidebar or below results)
   ══════════════════════════════════════════════════════ */
.md-snapshot {
    border-radius: var(--md-shape-md);
    padding: 14px 16px;
    background:
        radial-gradient(400px 160px at 100% 0%, rgba(var(--md-primary-rgb),0.16), transparent 60%),
        var(--md-surface-container);
    border: 1px solid var(--md-outline);
    box-shadow: var(--md-shadow-1);
    margin-bottom: 12px;
    animation: md-scale-in 400ms var(--md-ease-emphasized) both;
}
.md-snapshot-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px; font-weight: 900;
    text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--md-soft); margin-bottom: 10px;
}
.md-snapshot-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 6px 0;
    border-bottom: 1px solid var(--md-outline-variant);
    font-size: 13px;
}
.md-snapshot-row:last-child { border-bottom: none; }
.md-snapshot-key  { color: var(--md-soft); }
.md-snapshot-val  { font-weight: 800; font-family: 'Outfit', sans-serif !important; }

/* ══════════════════════════════════════════════════════
   MINI STAT STRIP (top of Insights tab)
   ══════════════════════════════════════════════════════ */
.md-strip {
    display: flex; flex-wrap: wrap; gap: 10px;
    margin-bottom: 18px;
}
.md-strip-item {
    flex: 1; min-width: 120px;
    padding: 13px 16px;
    border-radius: var(--md-shape-md);
    border: 1px solid var(--md-outline-variant);
    background: var(--md-surface-container);
    box-shadow: var(--md-shadow-1);
    display: flex; align-items: center; gap: 12px;
    transition: transform var(--md-dur-short) var(--md-ease-emphasized),
                box-shadow var(--md-dur-short) var(--md-ease-emphasized);
    animation: md-fade-up 420ms var(--md-ease-emphasized) both;
}
.md-strip-item:hover { transform: translateY(-3px); box-shadow: var(--md-shadow-2); }
.md-strip-icon {
    width: 40px; height: 40px; border-radius: var(--md-shape-sm); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 19px;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.24), rgba(var(--md-secondary-rgb),0.14));
}
.md-strip-val  { font-family: 'Outfit', sans-serif !important; font-size: 20px; font-weight: 900; line-height: 1; }
.md-strip-lbl  { font-size: 11px; color: var(--md-soft); font-weight: 700; margin-top: 2px; }

/* ══════════════════════════════════════════════════════
   IMPROVED SIDEBAR MINI CHECKLIST RING
   ══════════════════════════════════════════════════════ */
@keyframes ring-spin {
    from { stroke-dashoffset: 220; }
    to   { stroke-dashoffset: 0; }
}
.md-ring-svg circle.track { stroke: rgba(148,163,184,0.18); }
.md-ring-svg circle.fill  {
    stroke-linecap: round;
    transition: stroke-dashoffset 800ms var(--md-ease-emphasized);
}

/* ══════════════════════════════════════════════════════
   ANIMATED GRADIENT HERO STAT NUMBERS
   ══════════════════════════════════════════════════════ */
.md-animated-num {
    font-family: 'Outfit', sans-serif !important;
    font-size: 38px; font-weight: 900; line-height: 1;
    background: linear-gradient(110deg, #14b8a6 0%, #ffffff 50%, #14b8a6 100%);
    background-size: 200% 100%;
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: md-shine 4s linear infinite;
}

/* ══════════════════════════════════════════════════════
   RISK INDICATOR ARROWS in History
   ══════════════════════════════════════════════════════ */
.md-hist-delta-row {
    display: flex; align-items: center; gap: 8px;
    padding: 9px 14px;
    border: 1px solid var(--md-outline-variant);
    border-radius: var(--md-shape-md);
    margin-bottom: 8px;
    background: var(--md-surface);
    transition: background var(--md-dur-short) var(--md-ease-emphasized),
                transform var(--md-dur-short) var(--md-ease-emphasized);
    animation: md-fade-up 350ms var(--md-ease-emphasized) both;
}
.md-hist-delta-row:hover { transform: translateX(4px); background: var(--md-surface-container); }
.md-hist-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
.md-hist-time { color: var(--md-soft); font-size: 12px; flex: 1; }
.md-hist-risk { font-family: 'Outfit', sans-serif !important; font-size: 17px; font-weight: 900; }

/* ══════════════════════════════════════════════════════
   BLOOD PRESSURE REFERENCE CARD
   ══════════════════════════════════════════════════════ */
.md-bp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin-top: 10px;
}
.md-bp-table th {
    text-align: left;
    padding: 8px 12px;
    color: var(--md-soft);
    font-size: 11px;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--md-outline-variant);
}
.md-bp-table td {
    padding: 9px 12px;
    border-bottom: 1px solid rgba(148,163,184,0.08);
    line-height: 1.4;
}
.md-bp-table tr:last-child td { border-bottom: none; }
.md-bp-table tr:hover td { background: rgba(var(--md-primary-rgb),0.04); }
.md-bp-dot {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}

/* ══════════════════════════════════════════════════════
   WELLNESS SCORE RING (SVG donut)
   ══════════════════════════════════════════════════════ */
.md-wellness-ring-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 18px;
    border-radius: var(--md-shape-lg);
    background: var(--md-surface-container);
    border: 1px solid var(--md-outline-variant);
    box-shadow: var(--md-shadow-1);
    margin-bottom: 14px;
    animation: md-scale-in 450ms var(--md-ease-emphasized) both;
}
.md-wellness-ring-label {
    font-family: 'Outfit', sans-serif !important;
    font-size: 15px; font-weight: 900; text-align: center;
}
.md-wellness-ring-sub { color: var(--md-soft); font-size: 12px; text-align: center; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)



# ───────────────────────── Floating Quick Widget ───────────────────────────

def render_floating_widget():
    components.html(
        """
<script>
(function() {
    const parentDoc = window.parent.document;
    const parentWin = window.parent;
    const widgetId = "mm-widget-root-heart-risk";
    const styleId = widgetId + "-style";
    parentDoc.querySelectorAll('[id^="mm-widget-root"], #mm-widget-style').forEach((element) => element.remove());

    const style = parentDoc.createElement("style");
    style.id = styleId;
    style.textContent = `
        #${widgetId} {
            position: fixed;
            z-index: 2147483000;
            right: 24px;
            bottom: 26px;
            font-family: "Space Grotesk", Inter, system-ui, sans-serif;
            touch-action: none;
        }
        #${widgetId}.mm-dragging, #${widgetId}.mm-dragging * {
            cursor: grabbing !important;
            user-select: none !important;
        }
        #${widgetId} .mm-widget-panel {
            position: absolute;
            right: 0;
            bottom: 74px;
            width: min(340px, calc(100vw - 28px));
            max-height: min(520px, calc(100vh - 28px));
            overflow-y: auto !important;
            overflow-x: hidden;
            overscroll-behavior: contain;
            touch-action: pan-y;
            -webkit-overflow-scrolling: touch;
            padding: 12px;
            border-radius: 24px;
            border: 1px solid rgba(255,255,255,.16);
            background:
                radial-gradient(circle at 12% 0%, rgba(124,77,255,.28), transparent 38%),
                radial-gradient(circle at 92% 18%, rgba(0,188,212,.20), transparent 42%),
                rgba(13, 13, 20, .94);
            box-shadow: 0 18px 60px rgba(0,0,0,.38), 0 0 0 1px rgba(124,77,255,.10);
            backdrop-filter: blur(18px);
            transform: translateY(12px) scale(.96);
            opacity: 0;
            pointer-events: none;
            transition: opacity 320ms cubic-bezier(0.34,1.56,0.64,1), transform 320ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #${widgetId} .mm-widget-panel::-webkit-scrollbar { width: 7px; }
        #${widgetId} .mm-widget-panel::-webkit-scrollbar-track { background: transparent; }
        #${widgetId} .mm-widget-panel::-webkit-scrollbar-thumb {
            background: rgba(255,255,255,.28);
            border-radius: 999px;
        }
        #${widgetId}.mm-open .mm-widget-panel {
            opacity: 1;
            pointer-events: auto;
            transform: translateY(0) scale(1);
        }
        #${widgetId}.mm-left .mm-widget-panel { left: 0; right: auto; }
        #${widgetId}.mm-down .mm-widget-panel { top: 74px; bottom: auto; transform-origin: top right; }
        #${widgetId}.mm-left.mm-down .mm-widget-panel { transform-origin: top left; }
        #${widgetId}.mm-up .mm-widget-panel { bottom: 74px; top: auto; transform-origin: bottom right; }
        #${widgetId}.mm-left.mm-up .mm-widget-panel { transform-origin: bottom left; }

        @keyframes mm-head-in {
            0%   { opacity: 0; transform: translateY(-6px) scale(.97); }
            100% { opacity: 1; transform: translateY(0)    scale(1);   }
        }
        @keyframes mm-drag-pulse {
            0%,100% { box-shadow: 0 0 0 0   rgba(var(--mm-fab-primary-rgb),.45); }
            55%     { box-shadow: 0 0 0 5px rgba(var(--mm-fab-primary-rgb),.0);  }
        }
            100% { background-position:  200% center; }
        }
        #${widgetId} .mm-widget-head {
            position: sticky;
            top: -12px;
            z-index: 2;
            display: flex;
            flex-direction: column;
            gap: 0;
            padding: 14px 14px 12px;
            margin: -12px -12px 10px;
            border-radius: 20px 20px 0 0;
            background:
                linear-gradient(135deg,
                    rgba(var(--mm-fab-primary-rgb),.18) 0%,
                    rgba(var(--mm-fab-secondary-rgb),.10) 55%,
                    rgba(13,13,20,.96) 100%);
            border-bottom: 1px solid rgba(var(--mm-fab-primary-rgb),.18);
            backdrop-filter: blur(20px) saturate(1.4);
            box-shadow:
                0 4px 24px rgba(var(--mm-fab-primary-rgb),.10),
                inset 0 1px 0 rgba(255,255,255,.10);
            animation: mm-head-in 340ms cubic-bezier(0.34,1.56,0.64,1) both;
            overflow: hidden;
        }
        /* subtle gloss streak across header */
        #${widgetId} .mm-widget-head::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(ellipse at 18% 0%, rgba(255,255,255,.13), transparent 60%);
            pointer-events: none;
        }
        /* bottom edge glow line */
        #${widgetId} .mm-widget-head::after {
            content: "";
            position: absolute;
            bottom: 0; left: 12%; right: 12%;
            height: 1px;
            background: linear-gradient(90deg,
                transparent,
                rgba(var(--mm-fab-primary-rgb),.55) 40%,
                rgba(var(--mm-fab-secondary-rgb),.45) 60%,
                transparent);
        }
        #${widgetId} .mm-widget-head-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 8px;
            margin-bottom: 7px;
        }
        #${widgetId} .mm-widget-draghint {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 4px 10px 4px 7px;
            border-radius: 999px;
            background: rgba(var(--mm-fab-primary-rgb),.18);
            border: 1px solid rgba(var(--mm-fab-primary-rgb),.36);
            color: rgba(220,210,255,.90);
            font-size: 9.5px;
            font-weight: 900;
            letter-spacing: .10em;
            text-transform: uppercase;
            animation: mm-drag-pulse 2.6s ease-in-out infinite 1.2s;
            transition: background 160ms ease, border-color 160ms ease;
            cursor: grab;
        }
        #${widgetId} .mm-widget-draghint:hover {
            background: rgba(var(--mm-fab-primary-rgb),.28);
            border-color: rgba(var(--mm-fab-primary-rgb),.55);
        }
        #${widgetId} .mm-widget-draghint-dot {
            width: 6px; height: 6px;
            border-radius: 50%;
            background: rgba(var(--mm-fab-primary-rgb),1);
            box-shadow: 0 0 6px 2px rgba(var(--mm-fab-primary-rgb),.60);
            flex-shrink: 0;
        }
        #${widgetId} .mm-widget-head-right {
            display: flex;
            align-items: center;
            gap: 6px;
        }
        #${widgetId} .mm-widget-head-icon {
            width: 30px; height: 30px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg,
                rgba(var(--mm-fab-primary-rgb),.30),
                rgba(var(--mm-fab-secondary-rgb),.20));
            border: 1px solid rgba(var(--mm-fab-primary-rgb),.28);
            box-shadow:
                0 2px 10px rgba(var(--mm-fab-primary-rgb),.22),
                inset 0 1px 0 rgba(255,255,255,.14);
            font-size: 14px;
            flex-shrink: 0;
        }
        #${widgetId} .mm-widget-title {
            font-size: 13.5px;
            font-weight: 900;
            line-height: 1.15;
            letter-spacing: .01em;
            color: #f0ebff;
            text-shadow:
                0 0 18px rgba(var(--mm-fab-primary-rgb),.55),
                0 1px 0 rgba(0,0,0,.18);
        }
        #${widgetId} .mm-widget-sub {
            margin-top: 1px;
            font-size: 10.5px;
            font-weight: 700;
            color: rgba(200,195,230,.62);
            letter-spacing: .02em;
            line-height: 1.3;
        }
        /* divider below title row */
        #${widgetId} .mm-widget-head-divider {
            height: 1px;
            margin: 0 -2px;
            background: linear-gradient(90deg,
                transparent 0%,
                rgba(var(--mm-fab-primary-rgb),.22) 30%,
                rgba(var(--mm-fab-secondary-rgb),.18) 70%,
                transparent 100%);
            border-radius: 999px;
        }

                #${widgetId} .mm-widget-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            padding-bottom: 2px;
        }
        #${widgetId} .mm-tool {
            min-height: 68px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 7px;
            padding: 11px;
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,.11);
            background: rgba(255,255,255,.07);
            color: #f7f3ff;
            cursor: pointer;
            touch-action: manipulation;
            text-align: left;
            transition: transform 150ms ease, background 150ms ease, border-color 150ms ease;
        }
        #${widgetId} .mm-tool:hover {
            transform: translateY(-2px);
            background: rgba(124,77,255,.22);
            border-color: rgba(124,77,255,.42);
        }
        #${widgetId} .mm-tool-code {
            width: 34px;
            height: 28px;
            display: grid;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg,#7c4dff,#00bcd4);
            color: #fff;
            font-size: 11px;
            font-weight: 1000;
            box-shadow: 0 8px 18px rgba(124,77,255,.28);
        }
        #${widgetId} .mm-tool-label { font-size: 12px; line-height: 1.2; font-weight: 900; }
        #${widgetId} .mm-tool-note { color: rgba(240,238,255,.60); font-size: 10px; line-height: 1.2; font-weight: 700; }
        /* ── M3 EXPRESSIVE FAB ─────────────────────────────────────── */
        @keyframes mm-pulse-ring {
            0%   { transform: scale(1);   opacity: .55; }
            60%  { transform: scale(1.55); opacity: .18; }
            100% { transform: scale(1.55); opacity: 0;   }
        }
        @keyframes mm-breathe {
            0%,100% { transform: scale(1);    opacity: .38; }
            50%      { transform: scale(1.32); opacity: .15; }
        }
        @keyframes mm-shimmer {
            0%   { transform: translateX(-130%) skewX(-18deg); }
            100% { transform: translateX(230%)  skewX(-18deg); }
        }
        @keyframes mm-orbit {
            from { transform: rotate(0deg)   translateX(34px) rotate(0deg); }
            to   { transform: rotate(360deg) translateX(34px) rotate(-360deg); }
        }
        @keyframes mm-orbit-rev {
            from { transform: rotate(0deg)   translateX(28px) rotate(0deg); }
            to   { transform: rotate(-360deg) translateX(28px) rotate(360deg); }
        }
        @keyframes mm-orbit-mid {
            from { transform: rotate(120deg)  translateX(31px) rotate(-120deg); }
            to   { transform: rotate(480deg)  translateX(31px) rotate(-480deg); }
        }
        @keyframes mm-ripple-out {
            0%   { transform: scale(0);   opacity: .55; }
            80%  { transform: scale(2.8); opacity: .12; }
            100% { transform: scale(2.8); opacity: 0;   }
        }
        @keyframes mm-dot-appear {
            0%  { transform: scale(0) translateX(0); opacity: 0; }
            50% { opacity: 1; }
            100%{ transform: scale(1); opacity: 1; }
        }
        @keyframes mm-grid-dot-in {
            0%   { transform: scale(0) rotate(-45deg); opacity: 0; }
            60%  { transform: scale(1.25) rotate(8deg); opacity: 1; }
            100% { transform: scale(1) rotate(0deg); opacity: 1; }
        }
        @keyframes mm-x-in {
            0%   { transform: translate(-50%,-50%) rotate(0deg)   scale(0); opacity: 0; }
            60%  { transform: translate(-50%,-50%) rotate(52deg)  scale(1.12); opacity: 1; }
            100% { transform: translate(-50%,-50%) rotate(45deg)  scale(1); opacity: 1; }
        }
        @keyframes mm-x-in2 {
            0%   { transform: translate(-50%,-50%) rotate(0deg)    scale(0); opacity: 0; }
            60%  { transform: translate(-50%,-50%) rotate(-52deg)  scale(1.12); opacity: 1; }
            100% { transform: translate(-50%,-50%) rotate(-45deg)  scale(1); opacity: 1; }
        }

        #${widgetId} .mm-fab {
            --mm-fab-primary-rgb: var(--md-primary-rgb, 103,80,164);
            --mm-fab-secondary-rgb: var(--md-secondary-rgb, 0,106,106);
            --mm-fab-size: clamp(58px, 5.8vw, 66px);
            width:  var(--mm-fab-size);
            height: var(--mm-fab-size);
            position: relative;
            display: grid;
            place-items: center;
            isolation: isolate;
            border-radius: 24px;
            border: 1.5px solid rgba(255,255,255,.20);
            background:
                linear-gradient(145deg,
                    rgba(var(--mm-fab-primary-rgb),.36) 0%,
                    rgba(var(--mm-fab-secondary-rgb),.22) 60%,
                    rgba(20,18,34,.92) 100%);
            color: #fff;
            box-shadow:
                0 0 0 0 rgba(var(--mm-fab-primary-rgb),.50),
                0 16px 38px rgba(var(--mm-fab-primary-rgb),.28),
                0 6px 18px rgba(0,0,0,.26),
                inset 0 1.5px 0 rgba(255,255,255,.22),
                inset 0 -1px 0 rgba(0,0,0,.18);
            cursor: grab;
            transition:
                border-radius   380ms cubic-bezier(0.34,1.56,0.64,1),
                box-shadow      280ms cubic-bezier(0.2,0,0,1),
                transform       280ms cubic-bezier(0.34,1.56,0.64,1),
                background      280ms cubic-bezier(0.2,0,0,1);
            user-select: none;
            overflow: visible;
        }

        /* Breathing pulse ring */
        #${widgetId} .mm-fab-pulse {
            position: absolute;
            inset: -6px;
            border-radius: inherit;
            border: 2px solid rgba(var(--mm-fab-primary-rgb), .60);
            pointer-events: none;
            animation: mm-breathe 2.8s ease-in-out infinite;
            border-radius: 30px;
            transition: border-radius 380ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #${widgetId} .mm-fab-pulse2 {
            position: absolute;
            inset: -12px;
            border-radius: 36px;
            border: 1.5px solid rgba(var(--mm-fab-primary-rgb), .35);
            pointer-events: none;
            animation: mm-breathe 2.8s ease-in-out infinite .6s;
            transition: border-radius 380ms cubic-bezier(0.34,1.56,0.64,1);
        }

        /* Ripple layer (triggered on click via JS class) */
        #${widgetId} .mm-fab-ripple {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            pointer-events: none;
            overflow: hidden;
        }
        #${widgetId} .mm-fab-ripple::after {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: 50%;
            background: rgba(var(--mm-fab-primary-rgb), .45);
            transform: scale(0);
            opacity: 0;
            transition: none;
        }
        #${widgetId}.mm-rippling .mm-fab-ripple::after {
            animation: mm-ripple-out 520ms cubic-bezier(0.2,0,0,1) forwards;
        }

        /* Shimmer sweep */
        #${widgetId} .mm-fab-shimmer {
            position: absolute;
            inset: 0;
            border-radius: inherit;
            overflow: hidden;
            pointer-events: none;
        }
        #${widgetId} .mm-fab-shimmer::before {
            content: "";
            position: absolute;
            top: -20%;
            left: -20%;
            width: 50%;
            height: 140%;
            background: linear-gradient(105deg,
                transparent 30%,
                rgba(255,255,255,.28) 50%,
                transparent 70%);
            animation: mm-shimmer 3.2s cubic-bezier(0.4,0,0.6,1) infinite 1.1s;
        }

        /* Surface gloss */
        #${widgetId} .mm-fab::before {
            content: "";
            position: absolute;
            inset: 0;
            border-radius: inherit;
            background:
                radial-gradient(circle at 28% 20%, rgba(255,255,255,.28), transparent 44%),
                radial-gradient(circle at 76% 82%, rgba(var(--mm-fab-secondary-rgb),.22), transparent 40%);
            pointer-events: none;
            z-index: 0;
        }

        /* Orbital particle dots */
        #${widgetId} .mm-fab-orbit-wrap {
            position: absolute;
            inset: 0;
            pointer-events: none;
            display: grid;
            place-items: center;
            opacity: 0;
            transition: opacity 320ms cubic-bezier(0.2,0,0,1);
        }
        #${widgetId}.mm-open .mm-fab-orbit-wrap { opacity: 1; }
        #${widgetId} .mm-fab-dot {
            position: absolute;
            width: 6px; height: 6px;
            border-radius: 50%;
        }
        #${widgetId} .mm-fab-dot:nth-child(1) {
            background: rgba(var(--mm-fab-primary-rgb),1);
            box-shadow: 0 0 8px 2px rgba(var(--mm-fab-primary-rgb),.70);
            animation: mm-orbit 2.6s linear infinite;
        }
        #${widgetId} .mm-fab-dot:nth-child(2) {
            background: rgba(var(--mm-fab-secondary-rgb),1);
            box-shadow: 0 0 8px 2px rgba(var(--mm-fab-secondary-rgb),.70);
            width: 5px; height: 5px;
            animation: mm-orbit-rev 3.4s linear infinite;
        }
        #${widgetId} .mm-fab-dot:nth-child(3) {
            background: rgba(255,255,255,.90);
            box-shadow: 0 0 6px 1px rgba(255,255,255,.55);
            width: 4px; height: 4px;
            animation: mm-orbit-mid 4.1s linear infinite;
        }

        /* Hover / open states */
        #${widgetId} .mm-fab:hover {
            transform: translateY(-3px) scale(1.045);
            box-shadow:
                0 0 0 6px rgba(var(--mm-fab-primary-rgb),.14),
                0 20px 44px rgba(var(--mm-fab-primary-rgb),.34),
                0 8px 22px rgba(0,0,0,.28),
                inset 0 1.5px 0 rgba(255,255,255,.26);
        }
        #${widgetId}.mm-open .mm-fab {
            border-radius: 50%;
            background:
                linear-gradient(145deg,
                    rgba(var(--mm-fab-primary-rgb),.44) 0%,
                    rgba(var(--mm-fab-secondary-rgb),.28) 60%,
                    rgba(12,10,22,.94) 100%);
            box-shadow:
                0 0 0 10px rgba(var(--mm-fab-primary-rgb),.10),
                0 18px 44px rgba(var(--mm-fab-primary-rgb),.32),
                0 6px 18px rgba(0,0,0,.28),
                inset 0 1.5px 0 rgba(255,255,255,.22);
        }
        #${widgetId}.mm-open .mm-fab-pulse  { border-radius: 50%; inset: -8px; }
        #${widgetId}.mm-open .mm-fab-pulse2 { border-radius: 50%; inset: -16px; }

        /* ── GRID → X MORPHING ICON ─────────────────────── */
        #${widgetId} .mm-fab-icon {
            position: relative;
            z-index: 2;
            width: 28px;
            height: 28px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: 1fr 1fr;
            gap: 4.5px;
            padding: 0;
        }
        #${widgetId} .mm-fab-icon .mm-gd {
            border-radius: 4px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 1px 6px rgba(255,255,255,.20);
            transition:
                transform        380ms cubic-bezier(0.34,1.56,0.64,1),
                border-radius    380ms cubic-bezier(0.34,1.56,0.64,1),
                opacity          260ms cubic-bezier(0.2,0,0,1),
                background       260ms cubic-bezier(0.2,0,0,1),
                width            380ms cubic-bezier(0.34,1.56,0.64,1),
                height           380ms cubic-bezier(0.34,1.56,0.64,1);
        }
        /* dot 1 top-left */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(1) { background: rgba(var(--mm-fab-primary-rgb),1); box-shadow: 0 0 6px 1px rgba(var(--mm-fab-primary-rgb),.55); }
        /* dot 2 top-right */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(2) { background: rgba(255,255,255,.90); }
        /* dot 3 bottom-left */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(3) { background: rgba(255,255,255,.90); }
        /* dot 4 bottom-right */
        #${widgetId} .mm-fab-icon .mm-gd:nth-child(4) { background: rgba(var(--mm-fab-secondary-rgb),1); box-shadow: 0 0 6px 1px rgba(var(--mm-fab-secondary-rgb),.55); }

        /* OPEN: morph to X using two bars via pseudo — hide dots, show X */
        #${widgetId}.mm-open .mm-fab-icon .mm-gd {
            opacity: 0;
            transform: scale(.35) rotate(90deg);
        }
        /* X bar 1 */
        #${widgetId} .mm-fab-icon::before,
        #${widgetId} .mm-fab-icon::after {
            content: "";
            position: absolute;
            left: 50%; top: 50%;
            width: 22px; height: 3px;
            border-radius: 999px;
            background: rgba(255,255,255,.96);
            box-shadow: 0 1px 8px rgba(255,255,255,.24);
            opacity: 0;
            pointer-events: none;
            transition:
                opacity  260ms cubic-bezier(0.2,0,0,1),
                transform 400ms cubic-bezier(0.34,1.56,0.64,1);
        }
        #${widgetId} .mm-fab-icon::before {
            transform: translate(-50%,-50%) rotate(0deg) scale(.5);
        }
        #${widgetId} .mm-fab-icon::after {
            transform: translate(-50%,-50%) rotate(0deg) scale(.5);
        }
        #${widgetId}.mm-open .mm-fab-icon::before {
            opacity: 1;
            animation: mm-x-in  380ms cubic-bezier(0.34,1.56,0.64,1) forwards;
            transform: translate(-50%,-50%) rotate(45deg) scale(1);
        }
        #${widgetId}.mm-open .mm-fab-icon::after {
            opacity: 1;
            animation: mm-x-in2 380ms cubic-bezier(0.34,1.56,0.64,1) forwards;
            transform: translate(-50%,-50%) rotate(-45deg) scale(1);
        }

        @media (max-width: 620px) {
            #${widgetId} .mm-fab {
                width: 58px; height: 58px;
                border-radius: 22px;
            }
            #${widgetId} .mm-fab-pulse  { inset: -5px; }
            #${widgetId} .mm-fab-pulse2 { inset: -10px; }
        }

        #${widgetId} .mm-toast {
            position: absolute;
            right: 0;
            bottom: -36px;
            min-width: 190px;
            padding: 8px 11px;
            border-radius: 999px;
            background: rgba(15,10,28,.92);
            color: rgba(240,238,255,.82);
            border: 1px solid rgba(255,255,255,.12);
            box-shadow: 0 8px 24px rgba(0,0,0,.25);
            font-size: 11px;
            font-weight: 800;
            text-align: center;
            opacity: 0;
            pointer-events: none;
            transform: translateY(-4px);
            transition: opacity 160ms ease, transform 160ms ease;
        }
        #${widgetId} .mm-toast.mm-show { opacity: 1; transform: translateY(0); }
        @media (max-width: 620px) {
            #${widgetId} { right: 14px; bottom: 18px; }
            #${widgetId} .mm-widget-panel { width: min(310px, calc(100vw - 24px)); }
            #${widgetId} .mm-widget-grid { grid-template-columns: 1fr; }
            #${widgetId} .mm-tool { min-height: 58px; }
        }
    `;
    parentDoc.head.appendChild(style);

    const root = parentDoc.createElement("div");
    root.id = widgetId;
    const tools = [
        ['Assessment', 'AS', 'Assessment', 'Risk form', 'tab'],
        ['Insights', 'IN', 'Insights', 'Charts and drivers', 'tab'],
        ['Tools', 'TL', 'Tools', 'BMI and what-if', 'tab'],
        ['Wellness', 'WL', 'Wellness', 'Goals and quiz', 'tab'],
        ['History', 'HS', 'History', 'Saved assessments', 'tab'],
        ['About', 'AB', 'About', 'Project info', 'tab'],
        ['Scan Studio', 'SC', 'Scan Studio', 'Enhance cardiac scans', 'tab'],
        ['input', 'FS', 'Focus Form', 'Jump to inputs', 'input'],
        ['top', 'UP', 'Back to top', 'Scroll upward', 'top']
    ];
    root.innerHTML = `
        <div class="mm-widget-panel" role="menu" aria-label="Heart Risk quick actions">
            <div class="mm-widget-head">
                <div class="mm-widget-head-top">
                    <div style="display:flex;align-items:center;gap:9px;">
                        <div class="mm-widget-head-icon">❤️</div>
                        <div style="display:flex;flex-direction:column;gap:1px;">
                            <div class="mm-widget-title">Heart Risk tools</div>
                            <div class="mm-widget-sub">Page actions and navigation</div>
                        </div>
                    </div>
                    <div class="mm-widget-draghint">
                        <div class="mm-widget-draghint-dot"></div>
                        DRAG
                    </div>
                </div>
                <div class="mm-widget-head-divider"></div>
            </div>
            <div class="mm-widget-grid">
                ${tools.map(([target, code, label, note, kind]) => `
                    <button class="mm-tool" data-target="${target}" data-kind="${kind}" type="button">
                        <span class="mm-tool-code">${code}</span>
                        <span class="mm-tool-label">${label}</span>
                        <span class="mm-tool-note">${note}</span>
                    </button>
                `).join("")}
            </div>
        </div>
        <div class="mm-fab" role="button" tabindex="0" aria-label="Open quick tools">
            <div class="mm-fab-pulse"></div>
            <div class="mm-fab-pulse2"></div>
            <div class="mm-fab-shimmer"></div>
            <div class="mm-fab-ripple"></div>
            <div class="mm-fab-orbit-wrap">
                <div class="mm-fab-dot"></div>
                <div class="mm-fab-dot"></div>
                <div class="mm-fab-dot"></div>
            </div>
            <div class="mm-fab-icon">
                <div class="mm-gd"></div>
                <div class="mm-gd"></div>
                <div class="mm-gd"></div>
                <div class="mm-gd"></div>
            </div>
        </div>

        <div class="mm-toast">Drag me anywhere. Click for tools.</div>
    `;
    parentDoc.body.appendChild(root);

    const panel = root.querySelector(".mm-widget-panel");
    const fab = root.querySelector(".mm-fab");
    const icon = root.querySelector(".mm-fab-icon");
    const toast = root.querySelector(".mm-toast");
    let startX = 0, startY = 0, originX = 0, originY = 0, dragging = false, moved = false;

    const saved = (() => {
        try { return JSON.parse(parentWin.localStorage.getItem(widgetId + "-pos") || "null"); }
        catch (_) { return null; }
    })();

    function clampPosition(x, y) {
        const rect = root.getBoundingClientRect();
        return {
            x: Math.min(Math.max(x, 8), parentWin.innerWidth - rect.width - 8),
            y: Math.min(Math.max(y, 8), parentWin.innerHeight - 72),
        };
    }
    function setPosition(x, y) {
        const p = clampPosition(x, y);
        root.style.left = p.x + "px";
        root.style.top = p.y + "px";
        root.style.right = "auto";
        root.style.bottom = "auto";
        root.classList.toggle("mm-left", p.x < parentWin.innerWidth / 2);
        try { parentWin.localStorage.setItem(widgetId + "-pos", JSON.stringify(p)); } catch (_) {}
        if (root.classList.contains("mm-open")) fitPanelToViewport();
    }
    function fitPanelToViewport() {
        const rect = root.getBoundingClientRect();
        const spaceAbove = Math.max(0, rect.top - 12);
        const spaceBelow = Math.max(0, parentWin.innerHeight - rect.bottom - 12);
        const openDown = spaceBelow >= spaceAbove || spaceAbove < 220;
        root.classList.toggle("mm-down", openDown);
        root.classList.toggle("mm-up", !openDown);
        root.classList.toggle("mm-left", rect.left < parentWin.innerWidth / 2);
        const available = (openDown ? spaceBelow : spaceAbove) - 16;
        panel.style.maxHeight = Math.max(180, Math.min(520, available)) + "px";
        panel.style.overflowY = "auto";
    }
    function togglePanel(force) {
        const open = typeof force === "boolean" ? force : !root.classList.contains("mm-open");
        if (open) fitPanelToViewport();
        root.classList.toggle("mm-open", open);
        fab.setAttribute("aria-label", open ? "Close quick tools" : "Open quick tools");
        if (open) setTimeout(fitPanelToViewport, 40);
    }
    function showToast(message) {
        toast.textContent = message;
        toast.classList.add("mm-show");
        setTimeout(() => toast.classList.remove("mm-show"), 1500);
    }
    function normalizeText(value) {
        return (value || "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
    }
    function getElementText(el) {
        return el ? (el.innerText || el.textContent || el.getAttribute("aria-label") || "") : "";
    }
    function getMatchScore(el, text) {
        const wanted = normalizeText(text);
        const actual = normalizeText(getElementText(el));
        if (!wanted || !actual) return 999;
        if (actual === wanted) return 0;
        if (actual.endsWith(" " + wanted) || actual.startsWith(wanted + " ")) return 1;
        if (actual.includes(wanted)) return 2;
        return 999;
    }
    function textMatches(value, text) {
        const wanted = normalizeText(text);
        return wanted && normalizeText(value).includes(wanted);
    }
    function activateElement(el) {
        const target = el.querySelector("input") || el;
        ["pointerdown", "mousedown", "mouseup", "click"].forEach(type => {
            target.dispatchEvent(new MouseEvent(type, {bubbles: true, cancelable: true, view: parentWin}));
        });
        target.click();
    }
    function clickByText(selector, text) {
        const candidates = Array.from(parentDoc.querySelectorAll(selector))
            .filter(el => !root.contains(el))
            .map(el => ({el, score: getMatchScore(el, text), length: normalizeText(getElementText(el)).length}))
            .filter(item => item.score < 999)
            .sort((a, b) => a.score - b.score || a.length - b.length);
        if (candidates.length) {
            activateElement(candidates[0].el);
            return true;
        }
        return false;
    }
    function focusInput() {
        const target = parentDoc.querySelector('[data-testid="stChatInput"] textarea, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] input');
        if (target) {
            target.scrollIntoView({behavior: "smooth", block: "center"});
            setTimeout(() => { target.focus(); target.click(); }, 350);
            return true;
        }
        return false;
    }
    function scrollToText(text) {
        const selectors = [
            "main h1, main h2, main h3, main .md-section-title, main .section-title, main .sec-title, main .md-contact-title, main [class*='section-title'], main [class*='form-title']",
            "h1,h2,h3,.md-section-title,.section-title,.sec-title,.md-contact-title,[class*='section-title'],[class*='form-title']",
            "main p, main span, main div"
        ];
        for (const selector of selectors) {
            const target = Array.from(parentDoc.querySelectorAll(selector)).find(el => {
                if (root.contains(el) || !textMatches(el.innerText, text)) return false;
                return selector.includes("div") ? (el.innerText || "").trim().length <= 220 : true;
            });
            if (target) {
                const scrollTarget = target.closest('[data-testid="stElementContainer"], [data-testid="stVerticalBlock"], section') || target;
                scrollTarget.scrollIntoView({behavior: "smooth", block: "start"});
                return true;
            }
        }
        return false;
    }
    function runAction(kind, target) {
        if (kind === "top") {
            (function() {
                function smoothScrollTo(el) {
                    if (!el || el.scrollTop === 0) return;
                    var start = el.scrollTop;
                    var startTime = null;
                    var duration = 520;
                    function ease(t) { return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t+2,3)/2; }
                    function step(timestamp) {
                        if (!startTime) startTime = timestamp;
                        var progress = Math.min((timestamp - startTime) / duration, 1);
                        el.scrollTop = start * (1 - ease(progress));
                        if (progress < 1) parentWin.requestAnimationFrame(step);
                    }
                    parentWin.requestAnimationFrame(step);
                }
                var selectors = [
                    '[data-testid="stMain"]',
                    '[data-testid="stAppViewContainer"]',
                    '[data-testid="block-container"]',
                    '.main > div',
                    'section.main',
                    '.stApp > section',
                ];
                selectors.forEach(function(sel) {
                    var el = parentDoc.querySelector(sel);
                    if (el) smoothScrollTo(el);
                });
                smoothScrollTo(parentDoc.documentElement);
                smoothScrollTo(parentDoc.body);
                parentWin.scrollTo({top: 0, behavior: "smooth"});
            })();
            return true;
        }
        if (kind === "input") return focusInput();
        if (kind === "tab") {
            const ok = clickByText('[role="tab"], [data-baseweb="tab"], button', target);
            if (ok) {
                // After tab switches, scroll the tab list into view so content is visible
                parentWin.setTimeout(function() {
                    var tabList = parentDoc.querySelector('[data-baseweb="tab-list"], .stTabs [role="tablist"]');
                    if (tabList) {
                        tabList.scrollIntoView({behavior: "smooth", block: "start"});
                    } else {
                        parentDoc.documentElement.scrollTop = 0;
                        parentDoc.body.scrollTop = 0;
                        parentWin.scrollTo({top: 0, behavior: "smooth"});
                    }
                }, 300);
            }
            return ok;
        }
        if (kind === "nav") return clickByText('[data-testid="stButton"] button, button', target) || clickByText('[data-testid="stRadio"] label, [role="radio"]', target);
        if (kind === "scroll") return scrollToText(target);
        return false;
    }

    if (saved && Number.isFinite(saved.x) && Number.isFinite(saved.y)) setPosition(saved.x, saved.y);

    panel.addEventListener("wheel", event => event.stopPropagation(), {passive: false});
    panel.addEventListener("touchmove", event => event.stopPropagation(), {passive: true});

    fab.addEventListener("pointerdown", event => {
        dragging = true;
        moved = false;
        startX = event.clientX;
        startY = event.clientY;
        const rect = root.getBoundingClientRect();
        originX = rect.left;
        originY = rect.top;
        root.classList.add("mm-dragging");
        fab.setPointerCapture(event.pointerId);
    });
    fab.addEventListener("pointermove", event => {
        if (!dragging) return;
        const dx = event.clientX - startX;
        const dy = event.clientY - startY;
        if (Math.abs(dx) + Math.abs(dy) > 5) moved = true;
        if (moved) {
            togglePanel(false);
            setPosition(originX + dx, originY + dy);
        }
    });
    fab.addEventListener("pointerup", event => {
        dragging = false;
        root.classList.remove("mm-dragging");
        try { fab.releasePointerCapture(event.pointerId); } catch (_) {}
        if (moved) showToast("Position saved");
        else togglePanel();
    });
    fab.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            togglePanel();
        }
        if (event.key === "Escape") togglePanel(false);
    });

    root.querySelectorAll(".mm-tool").forEach(button => {
        button.addEventListener("click", () => {
            const kind = button.getAttribute("data-kind");
            const target = button.getAttribute("data-target");
            togglePanel(false);
            const ok = runAction(kind, target);
            showToast(ok ? "Opening " + button.querySelector(".mm-tool-label").innerText : "Could not find target");
        });
    });

    parentDoc.addEventListener("pointerdown", event => {
        if (root.classList.contains("mm-open") && !root.contains(event.target)) {
            togglePanel(false);
        }
    });
    parentDoc.addEventListener("keydown", event => {
        if (event.key === "Escape") togglePanel(false);
    });
    parentWin.addEventListener("resize", () => {
        const rect = root.getBoundingClientRect();
        setPosition(rect.left, rect.top);
    });
    parentWin.addEventListener("scroll", () => {
        if (root.classList.contains("mm-open")) fitPanelToViewport();
    }, {passive: true});

    root.classList.toggle("mm-left", root.getBoundingClientRect().left < parentWin.innerWidth / 2);
    root.classList.add("mm-up");
})();
</script>
        """,
        height=0,
        width=0,
    )

# =================== HEART SCAN ANALYSIS TAB ===================
with tab_scan:

    st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════════════
   SCAN STUDIO — MD3 EXPRESSIVE FULL REDESIGN
   ═══════════════════════════════════════════════════════════════════════ */

/* ── Hero ─────────────────────────────────────────────────────────────── */
.studio-hero {
    border-radius: 28px;
    padding: 52px 48px 48px;
    margin-bottom: 32px;
    position: relative; overflow: hidden;
    background:
        radial-gradient(ellipse 800px 300px at 95% -10%,  rgba(var(--md-tertiary-rgb),0.28),  transparent 60%),
        radial-gradient(ellipse 600px 400px at -5% 110%,  rgba(var(--md-secondary-rgb),0.20), transparent 60%),
        radial-gradient(ellipse 500px 250px at 50%  70%,  rgba(var(--md-primary-rgb),0.12),   transparent 65%),
        linear-gradient(145deg, rgba(var(--md-primary-rgb),0.16) 0%, rgba(var(--md-tertiary-rgb),0.06) 55%, transparent),
        var(--md-surface);
    border: 1.5px solid rgba(var(--md-primary-rgb),0.20);
    box-shadow:
        0 1px 2px rgba(0,0,0,0.10),
        0 4px 16px rgba(var(--md-primary-rgb),0.10),
        0 16px 48px rgba(var(--md-primary-rgb),0.07),
        inset 0 1px 0 rgba(255,255,255,0.08);
    animation: studio-hero-in 600ms cubic-bezier(0.05,0.7,0.1,1.0) both;
}
@keyframes studio-hero-in {
    from { opacity:0; transform: translateY(24px) scale(0.98); }
    to   { opacity:1; transform: translateY(0)   scale(1);    }
}
.studio-hero::before {
    content: "🫀"; font-size: 220px; opacity: 0.035;
    position: absolute; right: -24px; top: -32px;
    line-height: 1; pointer-events: none; user-select: none;
    transform: rotate(-12deg); filter: blur(1px);
}
.studio-hero::after {
    content: ""; position: absolute; inset: 0; border-radius: inherit;
    background: linear-gradient(180deg, rgba(255,255,255,0.04) 0%, transparent 50%);
    pointer-events: none;
}
.studio-title {
    font-size: clamp(1.8rem, 3.5vw, 2.6rem);
    font-weight: 900; letter-spacing: -0.03em; line-height: 1.1;
    background: linear-gradient(135deg, var(--md-on-surface) 30%, rgba(var(--md-primary-rgb),0.90) 70%, rgba(var(--md-tertiary-rgb),0.85) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; margin-bottom: 12px;
}
.studio-subtitle {
    font-size: 1.05rem; color: var(--md-on-surface-variant);
    line-height: 1.75; max-width: 660px; font-weight: 400;
}
.studio-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 7px 18px; border-radius: 999px; margin-bottom: 18px;
    background: linear-gradient(90deg, rgba(var(--md-primary-rgb),0.20), rgba(var(--md-tertiary-rgb),0.16));
    border: 1.5px solid rgba(var(--md-primary-rgb),0.40);
    color: #14b8a6; font-size: 10.5px; font-weight: 900; letter-spacing: 0.10em;
    text-transform: uppercase; backdrop-filter: blur(12px);
    box-shadow: 0 2px 8px rgba(var(--md-primary-rgb),0.12);
}
.studio-feature-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }
.studio-feature-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 8px 16px; border-radius: 16px; font-size: 12.5px; font-weight: 700;
    background: var(--md-surface-container-high);
    border: 1.5px solid var(--md-outline-variant);
    color: var(--md-on-surface-variant);
    transition: all 180ms cubic-bezier(0.05,0.7,0.1,1.0); cursor: default;
}
.studio-feature-chip:hover {
    background: rgba(var(--md-primary-rgb),0.12);
    border-color: rgba(var(--md-primary-rgb),0.35);
    color: var(--md-primary); transform: translateY(-1px);
}

/* ── Format pill strip ───────────────────────────────────────────────── */
.format-strip { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 28px; }
.format-pill {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 600;
    background: var(--md-surface-container); border: 1px solid var(--md-outline-variant);
    color: var(--md-on-surface-variant); transition: all 150ms ease;
}
.format-pill:hover { transform: translateY(-1px); }
.format-pill.accent {
    background: rgba(var(--md-primary-rgb),0.12); border-color: rgba(var(--md-primary-rgb),0.35);
    color: var(--md-primary); font-weight: 700;
}

/* ═══════════════════════════════════════════════════════════════════════
   UPLOAD ZONE — EXPRESSIVE DRAG-AND-DROP CARD
   ═══════════════════════════════════════════════════════════════════════ */
.upload-zone-wrapper { position: relative; margin: 4px 0 32px 0; }
.upload-zone-card {
    border-radius: 24px; padding: 0;
    background:
        radial-gradient(ellipse 600px 200px at 50% 0%, rgba(var(--md-primary-rgb),0.10), transparent 70%),
        var(--md-surface-container);
    border: 2px dashed rgba(var(--md-primary-rgb),0.38);
    overflow: hidden;
    transition: border-color 220ms ease, box-shadow 220ms ease, transform 220ms ease;
    animation: upload-card-in 500ms cubic-bezier(0.05,0.7,0.1,1.0) both;
    animation-delay: 100ms;
}
@keyframes upload-card-in {
    from { opacity:0; transform: translateY(16px); }
    to   { opacity:1; transform: translateY(0);    }
}
.upload-zone-card:hover, .upload-zone-card:focus-within {
    border-color: rgba(var(--md-primary-rgb),0.70);
    box-shadow: 0 0 0 4px rgba(var(--md-primary-rgb),0.10), 0 8px 32px rgba(var(--md-primary-rgb),0.14);
    transform: translateY(-2px);
}
.upload-zone-top {
    display: flex; flex-direction: column; align-items: center;
    padding: 44px 32px 18px; text-align: center; gap: 0;
}
.upload-icon-ring {
    width: 80px; height: 80px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.22), rgba(var(--md-tertiary-rgb),0.18));
    border: 2px solid rgba(var(--md-primary-rgb),0.30);
    display: flex; align-items: center; justify-content: center;
    font-size: 36px; margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(var(--md-primary-rgb),0.18);
    animation: icon-pulse 2.8s ease-in-out infinite;
}
@keyframes icon-pulse {
    0%, 100% { box-shadow: 0 4px 20px rgba(var(--md-primary-rgb),0.18); transform: scale(1); }
    50%       { box-shadow: 0 4px 28px rgba(var(--md-primary-rgb),0.34); transform: scale(1.05); }
}
.upload-zone-title {
    font-size: 1.35rem; font-weight: 900; color: var(--md-on-surface);
    letter-spacing: -0.01em; margin-bottom: 8px;
}
.upload-zone-sub {
    font-size: 13.5px; color: var(--md-on-surface-variant);
    line-height: 1.65; max-width: 440px; margin-bottom: 20px;
}
.upload-zone-formats {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 7px; margin-bottom: 12px;
}
.upload-fmt-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 5px 13px; border-radius: 10px; font-size: 12px; font-weight: 800;
    background: rgba(var(--md-primary-rgb),0.12);
    border: 1.5px solid rgba(var(--md-primary-rgb),0.30);
    color: var(--md-primary); letter-spacing: 0.06em; text-transform: uppercase;
}
.upload-zone-hint { font-size: 11.5px; color: var(--md-on-surface-variant); opacity: 0.70; margin-bottom: 8px; }
.upload-zone-divider {
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, var(--md-outline-variant), transparent);
    margin: 2px 0 0;
}
.upload-zone-bottom {
    display: flex; align-items: center; gap: 20px;
    padding: 14px 28px; background: var(--md-surface-container-high);
    flex-wrap: wrap;
}
.upload-feat-item {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 12px; font-weight: 700; color: var(--md-on-surface-variant);
}
.upload-feat-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.7), rgba(var(--md-tertiary-rgb),0.6));
    flex-shrink: 0;
}
.upload-streamlit-embed { padding: 0 28px 24px; }
.upload-streamlit-embed [data-testid="stFileUploader"] {
    border: none !important; background: transparent !important; padding: 0 !important;
}
.upload-streamlit-embed [data-testid="stFileUploaderDropzone"] {
    border-radius: 16px !important;
    border: 1.5px dashed rgba(var(--md-primary-rgb),0.30) !important;
    background: rgba(var(--md-primary-rgb),0.04) !important;
    transition: all 180ms ease !important; min-height: 80px !important;
}
.upload-streamlit-embed [data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(var(--md-primary-rgb),0.55) !important;
    background: rgba(var(--md-primary-rgb),0.09) !important;
}

/* ── Stats grid ──────────────────────────────────────────────────────── */
.stats-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 16px; margin: 28px 0 12px 0;
}
@media (max-width: 768px) { .stats-grid { grid-template-columns: repeat(2, 1fr); } }
.stat-card {
    padding: 26px 16px 20px; border-radius: 20px;
    background: var(--md-surface-container); border: 1.5px solid var(--md-outline-variant);
    text-align: center; position: relative; overflow: hidden;
    transition: transform 220ms cubic-bezier(0.05,0.7,0.1,1.0), box-shadow 220ms;
    animation: stat-card-in 450ms cubic-bezier(0.05,0.7,0.1,1.0) both;
}
@keyframes stat-card-in {
    from { opacity:0; transform: translateY(12px) scale(0.97); }
    to   { opacity:1; transform: translateY(0)    scale(1);    }
}
.stat-card:nth-child(1){animation-delay:0ms}.stat-card:nth-child(2){animation-delay:60ms}
.stat-card:nth-child(3){animation-delay:120ms}.stat-card:nth-child(4){animation-delay:180ms}
.stat-card:hover { transform: translateY(-5px) scale(1.02); box-shadow: 0 10px 32px rgba(var(--md-primary-rgb),0.18); }
.stat-card::before {
    content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, rgba(var(--md-primary-rgb),0.6), rgba(var(--md-tertiary-rgb),0.5));
    border-radius: 20px 20px 0 0;
}
.stat-card::after {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 50% -10%, rgba(var(--md-primary-rgb),0.10), transparent 65%);
    pointer-events: none;
}
.stat-icon { font-size: 26px; margin-bottom: 8px; }
.stat-value { font-size: 1.35rem; font-weight: 900; color: var(--md-primary); line-height: 1.1; letter-spacing: -0.01em; }
.stat-label { font-size: 10.5px; color: var(--md-on-surface-variant); margin-top: 5px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.quality-bar-bg { background: var(--md-surface-container-high); border-radius: 999px; height: 5px; width: 80%; margin: 8px auto 0; overflow: hidden; }
.quality-bar-fill { height: 5px; border-radius: 999px; background: linear-gradient(90deg, #14b8a6, #6366f1, var(--md-primary)); transition: width 1.0s cubic-bezier(0.05,0.7,0.1,1.0); }

/* ── Section header ──────────────────────────────────────────────────── */
.studio-section-header { display: flex; align-items: center; gap: 14px; margin: 36px 0 8px 0; }
.studio-section-icon {
    width: 44px; height: 44px; border-radius: 14px;
    background: linear-gradient(135deg, rgba(var(--md-primary-rgb),0.25), rgba(var(--md-tertiary-rgb),0.20));
    border: 1.5px solid rgba(var(--md-primary-rgb),0.28);
    display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0;
    box-shadow: 0 2px 10px rgba(var(--md-primary-rgb),0.14);
}
.studio-section-title { font-size: 1.2rem; font-weight: 900; color: var(--md-on-surface); letter-spacing: -0.01em; }
.studio-section-sub { font-size: 12.5px; color: var(--md-on-surface-variant); margin-top: 2px; line-height: 1.5; }

/* ── Panel cards ─────────────────────────────────────────────────────── */
.panel-card {
    border-radius: 20px; border: 1.5px solid var(--md-outline-variant);
    background: var(--md-surface-container); overflow: hidden; margin-bottom: 6px;
    transition: transform 220ms cubic-bezier(0.05,0.7,0.1,1.0), box-shadow 220ms, border-color 220ms;
    animation: md-fade-up 420ms cubic-bezier(0.05,0.7,0.1,1.0) both;
}
.panel-card:hover {
    transform: translateY(-5px) scale(1.01);
    box-shadow: 0 12px 36px rgba(var(--md-primary-rgb),0.18), 0 4px 12px rgba(0,0,0,0.08);
    border-color: rgba(var(--md-primary-rgb),0.50);
}
.panel-card-header {
    padding: 12px 16px 10px; display: flex; align-items: center; gap: 9px;
    border-bottom: 1px solid var(--md-outline-variant); background: var(--md-surface-container-high);
}
.panel-card-emoji { font-size: 18px; }
.panel-card-name { font-size: 10.5px; font-weight: 900; letter-spacing: 0.09em; text-transform: uppercase; color: var(--md-primary); flex: 1; }
.panel-card-body { padding: 0; }
.panel-card-footer { padding: 8px 10px 10px; background: var(--md-surface-container-high); }
.panel-desc { font-size: 10.5px; color: var(--md-on-surface-variant); line-height: 1.55; padding: 10px 16px 4px; }

/* ── Normalized section ──────────────────────────────────────────────── */
.norm-section {
    border-radius: 24px; padding: 32px 32px 24px; margin-top: 10px;
    background:
        radial-gradient(ellipse 600px 220px at 100% 0%, rgba(var(--md-tertiary-rgb),0.14), transparent),
        var(--md-surface-container);
    border: 1.5px solid var(--md-outline-variant); box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
.norm-compare-label {
    font-size: 11px; font-weight: 800; letter-spacing: 0.08em;
    text-transform: uppercase; color: var(--md-on-surface-variant); text-align: center; padding: 6px 0 4px;
}

/* ── Disclaimer ──────────────────────────────────────────────────────── */
.studio-disclaimer {
    margin-top: 32px; padding: 20px 24px; border-radius: 20px;
    background:
        radial-gradient(ellipse 400px 120px at 0% 50%, rgba(99,102,241,0.09), transparent),
        linear-gradient(135deg, rgba(99,102,241,0.08), rgba(var(--md-tertiary-rgb),0.06));
    border: 1.5px solid rgba(99,102,241,0.25);
    font-size: 12.5px; color: var(--md-on-surface-variant); line-height: 1.65;
    display: flex; gap: 12px; align-items: flex-start; box-shadow: 0 2px 10px rgba(99,102,241,0.07);
}
.disclaimer-icon { font-size: 20px; flex-shrink: 0; margin-top: 2px; }

/* ── Divider ─────────────────────────────────────────────────────────── */
.studio-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--md-outline-variant) 20%, var(--md-outline-variant) 80%, transparent 100%);
    margin: 32px 0; border: none;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="studio-hero">
  <div class="studio-badge">✦ AI-Powered · Clinically Inspired · Instant Results</div>
  <div class="studio-title">Scan Enhancement Studio</div>
  <div class="studio-subtitle">
    Upload any cardiac scan and instantly view it transformed across <strong>6 professional enhancement panels</strong> —
    contrast boost, edge detection, false-color mapping, sharpening, inversion, and percentile normalization.
  </div>
  <div class="studio-feature-row">
    <span class="studio-feature-chip">⚡ Instant Processing</span>
    <span class="studio-feature-chip">🔒 100% Local</span>
    <span class="studio-feature-chip">💾 Downloadable Results</span>
    <span class="studio-feature-chip">📊 Image Analytics</span>
    <span class="studio-feature-chip">🌈 6 Enhancement Modes</span>
  </div>
</div>
<div class="format-strip">
  <span class="format-pill accent">📁 JPG</span>
  <span class="format-pill accent">📁 JPEG</span>
  <span class="format-pill accent">📁 PNG</span>
  <span class="format-pill accent">📁 WEBP</span>
  <span class="format-pill">🫀 Echocardiogram</span>
  <span class="format-pill">🩻 Chest X-ray</span>
  <span class="format-pill">🧲 Cardiac MRI</span>
  <span class="format-pill">💻 CT Scan</span>
  <span class="format-pill">⬛ Grayscale Scans</span>
</div>
""", unsafe_allow_html=True)

    # ── Helper functions for image enhancement ────────────────────────────────
    def pil_to_b64(img: Image.Image, fmt="PNG") -> str:
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode()

    def enhance_contrast(img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance, ImageOps
        gray = ImageOps.autocontrast(img.convert("L"), cutoff=2)
        enhanced = ImageEnhance.Contrast(gray.convert("RGB")).enhance(2.2)
        return enhanced

    def enhance_edges(img: Image.Image) -> Image.Image:
        from PIL import ImageFilter, ImageOps
        gray = img.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        return ImageOps.autocontrast(edges).convert("RGB")

    def enhance_heatmap(img: Image.Image) -> Image.Image:
        import numpy as np_s
        gray = np_s.array(img.convert("L"), dtype=np_s.float32)
        norm = ((gray - gray.min()) / (gray.max() - gray.min() + 1e-8) * 255).astype(np_s.uint8)
        r = np_s.clip(norm * 2, 0, 255).astype(np_s.uint8)
        g = np_s.clip(255 - np_s.abs(norm.astype(np_s.int32) - 128) * 2, 0, 255).astype(np_s.uint8)
        b = np_s.clip((255 - norm) * 2, 0, 255).astype(np_s.uint8)
        return Image.fromarray(np_s.stack([r, g, b], axis=-1))

    def enhance_sharpen(img: Image.Image) -> Image.Image:
        from PIL import ImageEnhance, ImageFilter
        sharpened = img.convert("RGB").filter(ImageFilter.UnsharpMask(radius=2, percent=180, threshold=3))
        return ImageEnhance.Sharpness(sharpened).enhance(2.5)

    def enhance_invert(img: Image.Image) -> Image.Image:
        from PIL import ImageOps
        return ImageOps.invert(img.convert("RGB"))

    def enhance_normalize(img: Image.Image) -> Image.Image:
        import numpy as np_n
        from PIL import ImageOps
        arr = np_n.array(img.convert("L"), dtype=np_n.float32)
        p2, p98 = np_n.percentile(arr, 2), np_n.percentile(arr, 98)
        clipped = np_n.clip(arr, p2, p98)
        norm = ((clipped - p2) / (p98 - p2 + 1e-8) * 255).astype(np_n.uint8)
        return Image.fromarray(norm).convert("RGB")

    # ── Helper: load DICOM ────────────────────────────────────────────────────
    def load_dicom(file_obj) -> tuple:
        """Returns (PIL Image, metadata dict). Raises if pydicom not installed."""
        import pydicom
        import numpy as np_dcm
        ds = pydicom.dcmread(file_obj)
        arr = ds.pixel_array.astype(np_dcm.float32)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (np_dcm.clip(arr, 0, 1) * 255).astype(np_dcm.uint8)
        if arr.ndim == 2:
            pil_img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            pil_img = Image.fromarray(arr)
        meta = {}
        for tag in ["PatientName", "PatientAge", "PatientSex", "Modality",
                    "StudyDate", "StudyDescription", "InstitutionName",
                    "Manufacturer", "Rows", "Columns", "SliceThickness"]:
            try:
                val = getattr(ds, tag, None)
                if val is not None:
                    meta[tag] = str(val)
            except Exception:
                pass
        return pil_img, meta

    # ── Expressive Upload Zone Card ───────────────────────────────────────────
    st.markdown("""
<div class="upload-zone-wrapper">
  <div class="upload-zone-card">
    <div class="upload-zone-top">
      <div class="upload-icon-ring">🫀</div>
      <div class="upload-zone-title">Drop your cardiac scan here</div>
      <div class="upload-zone-sub">
        Upload an echocardiogram, chest X-ray, cardiac MRI, or any cardiac image to
        instantly generate 6 professional enhancement views with downloadable results.
      </div>
      <div class="upload-zone-formats">
        <span class="upload-fmt-badge">JPG</span>
        <span class="upload-fmt-badge">JPEG</span>
        <span class="upload-fmt-badge">PNG</span>
        <span class="upload-fmt-badge">WEBP</span>
      </div>
      <div class="upload-zone-hint">✦ All processing happens locally — no scan data leaves your device</div>
    </div>
    <div class="upload-zone-divider"></div>
""", unsafe_allow_html=True)

    st.markdown('<div class="upload-streamlit-embed">', unsafe_allow_html=True)
    uploaded_scan = st.file_uploader(
        "Upload Heart Scan",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPG, PNG, and WEBP cardiac scan images.",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-zone-bottom">
      <span class="upload-feat-item"><span class="upload-feat-dot"></span>Instant 6-panel enhancement</span>
      <span class="upload-feat-item"><span class="upload-feat-dot"></span>Image quality analytics</span>
      <span class="upload-feat-item"><span class="upload-feat-dot"></span>One-click download per panel</span>
      <span class="upload-feat-item"><span class="upload-feat-dot"></span>Percentile normalization view</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if uploaded_scan is not None:
        with st.spinner("🎨 Processing image..."):
            try:
                uploaded_scan.seek(0)
                pil_img = Image.open(uploaded_scan).convert("RGB")
            except Exception as e:
                st.error(f"❌ Could not load image: {e}")
                pil_img = None

        if pil_img is not None:
            import numpy as np_studio

            w, h = pil_img.size
            arr_gray = np_studio.array(pil_img.convert("L"), dtype=np_studio.float32)
            brightness = float(arr_gray.mean())
            contrast_val = float(arr_gray.std())
            sharpness_score = min(100, int((contrast_val / 128) * 100))
            quality_label = "Excellent" if sharpness_score > 70 else ("Good" if sharpness_score > 45 else ("Fair" if sharpness_score > 25 else "Poor"))
            quality_color = {"Excellent": "#14b8a6", "Good": "#6366f1", "Fair": "#f59e0b", "Poor": "#ef4444"}[quality_label]
            file_kb = uploaded_scan.size // 1024

            # ── Image stats bar ───────────────────────────────────────────────
            st.markdown(f"""
<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-icon">🖼️</div>
    <div class="stat-value">{w}×{h}</div>
    <div class="stat-label">Resolution (px)</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">💾</div>
    <div class="stat-value">{file_kb} KB</div>
    <div class="stat-label">File Size</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">🔬</div>
    <div class="stat-value" style="color:{quality_color};">{quality_label}</div>
    <div class="stat-label">Image Quality</div>
    <div class="quality-bar-bg"><div class="quality-bar-fill" style="width:{sharpness_score}%;"></div></div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">☀️</div>
    <div class="stat-value">{int(brightness)}<span style="font-size:0.7rem;font-weight:600;color:var(--md-on-surface-variant);">/255</span></div>
    <div class="stat-label">Avg Brightness</div>
  </div>
</div>
""", unsafe_allow_html=True)

            st.markdown("""
<hr class="studio-divider"/>
<div class="studio-section-header">
  <div class="studio-section-icon">🌈</div>
  <div>
    <div class="studio-section-title">6-Panel Enhancement View</div>
    <div class="studio-section-sub">Each panel applies a distinct image processing technique — click any image to expand full size</div>
  </div>
</div>
""", unsafe_allow_html=True)

            # ── Generate panels only when image changes, cache in session_state ──
            cache_key = f"studio_panels_{uploaded_scan.name}_{uploaded_scan.size}"
            if cache_key not in st.session_state:
                with st.spinner("🎨 Generating all 6 enhancement panels..."):
                    st.session_state[cache_key] = [
                        ("📷 Original",        "Unmodified scan as uploaded",                          pil_img),
                        ("⚡ Contrast Boost",   "Auto-contrast + 2.2× enhancement — reveals shadows",  enhance_contrast(pil_img)),
                        ("🔲 Edge Detection",   "FIND_EDGES filter — highlights structural boundaries", enhance_edges(pil_img)),
                        ("🌡️ False Color Map", "Intensity → RGB heatmap — hot=bright, cool=dark",     enhance_heatmap(pil_img)),
                        ("🔍 Sharpened",        "Unsharp mask + 2.5× sharpness — clarifies details",   enhance_sharpen(pil_img)),
                        ("🔄 Inverted",         "Luminosity inversion — X-ray style negative view",    enhance_invert(pil_img)),
                    ]

            panels = st.session_state[cache_key]

            # ── Display 3 columns × 2 rows ────────────────────────────────────
            row1 = st.columns(3, gap="small")
            row2 = st.columns(3, gap="small")
            all_cols = row1 + row2

            panel_colors = [
                "rgba(var(--md-primary-rgb),0.15)",
                "rgba(99,102,241,0.15)",
                "rgba(var(--md-tertiary-rgb),0.15)",
                "rgba(239,68,68,0.15)",
                "rgba(var(--md-secondary-rgb),0.15)",
                "rgba(148,163,184,0.15)",
            ]

            for i, (col, (label, desc, img)) in enumerate(zip(all_cols, panels)):
                emoji = label.split()[0]
                name  = " ".join(label.split()[1:])
                with col:
                    st.markdown(f"""
<div class="panel-card" style="animation-delay:{i*60}ms;">
  <div class="panel-card-header" style="background:linear-gradient(90deg,{panel_colors[i]},transparent);">
    <span class="panel-card-emoji">{emoji}</span>
    <span class="panel-card-name">{name}</span>
  </div>
</div>
""", unsafe_allow_html=True)
                    st.image(img, use_container_width=True)
                    st.markdown(f'<div class="panel-desc">{desc}</div>', unsafe_allow_html=True)
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="💾 Download",
                        data=buf.getvalue(),
                        file_name=f"scan_{name.lower().replace(' ','_')}.png",
                        mime="image/png",
                        use_container_width=True,
                        key=f"dl_{label}",
                    )

            # ── Normalized view full width ─────────────────────────────────────
            norm_cache_key = f"studio_norm_{uploaded_scan.name}_{uploaded_scan.size}"
            if norm_cache_key not in st.session_state:
                st.session_state[norm_cache_key] = enhance_normalize(pil_img)
            norm_img = st.session_state[norm_cache_key]

            st.markdown("""
<hr class="studio-divider"/>
<div class="studio-section-header">
  <div class="studio-section-icon">📊</div>
  <div>
    <div class="studio-section-title">Percentile Normalized View</div>
    <div class="studio-section-sub">Clips extreme pixel values (2nd–98th percentile) and stretches the range — best for revealing faint structural detail</div>
  </div>
</div>
<div class="norm-section">
""", unsafe_allow_html=True)

            col_n1, col_n2 = st.columns([1, 1], gap="large")
            with col_n1:
                st.markdown('<div class="norm-compare-label">⬛ Original</div>', unsafe_allow_html=True)
                st.image(pil_img, use_container_width=True)
            with col_n2:
                st.markdown('<div class="norm-compare-label">✨ Percentile Normalized</div>', unsafe_allow_html=True)
                st.image(norm_img, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            buf_norm = io.BytesIO()
            norm_img.save(buf_norm, format="PNG")
            st.download_button(
                "💾 Download Normalized View",
                data=buf_norm.getvalue(),
                file_name="scan_normalized.png",
                mime="image/png",
                key="dl_norm",
            )

            st.markdown("""
<div class="studio-disclaimer">
  <span class="disclaimer-icon">⚕️</span>
  <span><strong>Clinical Disclaimer:</strong> All transformations are performed locally using standard image processing techniques.
  No scan data is transmitted externally. These enhanced views are intended for <strong>visual reference and educational purposes only</strong>
  and do not constitute radiological analysis or medical diagnosis. Always consult a qualified cardiologist or radiologist
  for clinical interpretation of cardiac imaging.</span>
</div>
""", unsafe_allow_html=True)

    else:
        st.markdown("""
<div style="
    border-radius: 20px; padding: 48px 32px; text-align: center; margin-top: 8px;
    background: var(--md-surface-container);
    border: 2px dashed rgba(var(--md-primary-rgb),0.25);
    animation: md-fade-up 500ms cubic-bezier(0.05,0.7,0.1,1.0) both;
">
  <div style="font-size: 52px; margin-bottom: 16px; opacity: 0.6;">🫀</div>
  <div style="font-size: 1.1rem; font-weight: 800; color: var(--md-on-surface); margin-bottom: 8px;">
    Upload a cardiac scan to get started
  </div>
  <div style="font-size: 13px; color: var(--md-on-surface-variant); max-width: 380px; margin: 0 auto; line-height: 1.6;">
    Use the upload zone above to drop a JPG, PNG, or WEBP scan image and instantly launch all 6 enhancement panels.
  </div>
</div>
""", unsafe_allow_html=True)

# =================== END SCAN ENHANCEMENT STUDIO TAB ===================

footer_logo_html = '<div class="md-sb-avatar-box md-footer-avatar" role="img" aria-label="Heart logo">🫀</div>'

_footer_year = datetime.datetime.now().strftime('%Y')
st.markdown(
    f'<div class="md-footer">'
    f'<div class="md-footer-top">'
    f'<div class="md-footer-brand">{footer_logo_html}'
    f'<div><div class="md-footer-brand-name">Heart Risk Assessment<br/>AI Diagnosis</div>'
    f'<div class="md-footer-brand-sub">by {CREATOR_NAME}</div></div></div>'
    f'</div>'
    f'<div class="md-footer-links">'
    f'<a class="md-footer-link" href="https://github.com/YatinSharma1303/" target="_blank">🐙 GitHub</a>'
    f'<a class="md-footer-link" href="https://www.linkedin.com/in/yatin-sharma-793042372/" target="_blank">💼 LinkedIn</a>'
    f'<a class="md-footer-link" href="https://www.who.int/health-topics/cardiovascular-diseases" target="_blank">🌐 WHO</a>'
    f'<a class="md-footer-link" href="https://www.heart.org" target="_blank">❤️ AHA</a>'
    f'<a class="md-footer-link" href="https://www.cdc.gov/heart-disease/" target="_blank">🏥 CDC</a>'
    f'</div>'
    f'<div class="md-footer-meta">'
    f'<span class="md-footer-version">{APP_VERSION}</span>'
    f' &nbsp;·&nbsp; Made with <span class="md-footer-heart">❤️</span> by <strong>{CREATOR_NAME}</strong>'
    f' &nbsp;·&nbsp; © {_footer_year}'
    f'</div>'
    f'<div class="md-footer-disclaimer">⚕️ <strong>Medical Disclaimer:</strong> '
    f'This application is for educational purposes only and does not constitute medical advice, diagnosis, or treatment. '
    f'Always consult a qualified healthcare professional regarding any health concerns.</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Fix sidebar collapse button showing text instead of icon
st.markdown(
    """
<script>
(function fixSidebarBtn() {
    const COLLAPSE_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="11 17 6 12 11 7"/><polyline points="18 17 13 12 18 7"/></svg>';
    const EXPAND_SVG  = '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="13 17 18 12 13 7"/><polyline points="6 17 11 12 6 7"/></svg>';
    function getSVG(label) {
        return (label.toLowerCase().includes('close') || label.toLowerCase().includes('collapse'))
            ? COLLAPSE_SVG : EXPAND_SVG;
    }
    function fix() {
        const ariaLabels = [
            'Close sidebar', 'Open sidebar',
            'collapse sidebar', 'expand sidebar',
            'Collapse sidebar', 'Expand sidebar',
        ];
        ariaLabels.forEach(label => {
            const btn = document.querySelector('button[aria-label="' + label + '"]');
            if (!btn) return;
            const hasSVG = btn.querySelector('svg');
            const hasText = [...btn.childNodes].some(n => n.nodeType === Node.TEXT_NODE && n.textContent.trim().length > 0);
            if (hasText) {
                [...btn.childNodes].forEach(node => {
                    if (node.nodeType === Node.TEXT_NODE) node.textContent = '';
                });
            }
            if (!hasSVG) {
                btn.insertAdjacentHTML('beforeend', getSVG(label));
            }
        });
    }
    fix();
    const observer = new MutationObserver(fix);
    observer.observe(document.body, { childList: true, subtree: true, characterData: false });
})();
</script>
""",
    unsafe_allow_html=True,
)

render_floating_widget()