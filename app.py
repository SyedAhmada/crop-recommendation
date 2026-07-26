import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Crop Recommendation System", layout="wide")
st.title("🌾 Crop Recommendation System")
st.write("Predict the best crop for your soil and climate conditions using Machine Learning pipelines!")


# --- STEP 1: LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("merged_crop_dataset.csv")

    # Rename columns to standard names
    df = df.rename(columns={
        "P": "Phosphorous",
        "K": "Potassium",
        "N": "Nitrogen",
        "temperature": "Temperature",
        "humidity": "Humidity",
        "ph": "pH",
        "rainfall": "Rainfall",
        "label": "Crop"
    })

    df = df.dropna()
    return df


df = load_and_preprocess_data()


# --- STEP 1.5: CROP GROWING GUIDES DATABASE ---
CROP_GROWING_GUIDE = {
    "papaya": {
        "water": "Daily light watering in summer; every 2–3 days in cooler months. Keep soil moist, not flooded.",
        "season": "Warm tropical / subtropical climate year-round.",
        "harvest": "9 to 11 months after planting.",
        "tips": "Papayas are sensitive to waterlogging. Ensure excellent soil drainage to prevent root rot."
    },
    "rice": {
        "water": "High / shallow standing water (2–5 cm) required during initial growth stages.",
        "season": "Monsoon / Summer (Kharif season).",
        "harvest": "3 to 5 months after sowing.",
        "tips": "Stop watering about 10–14 days before harvest to allow grains to mature and dry."
    },
    "wheat": {
        "water": "Moderate (4 to 6 light irrigations at key growth stages like tillering and flowering).",
        "season": "Winter / Cool season (Rabi season).",
        "harvest": "4 to 5 months after sowing.",
        "tips": "Avoid heavy irrigation during harvesting stage to prevent crops from falling over."
    },
    "watermelon": {
        "water": "Deep, regular watering during vine development; reduce water significantly near maturity.",
        "season": "Warm Summer / Spring.",
        "harvest": "80 to 100 days (approx. 3 months).",
        "tips": "Cutting back water 1–2 weeks prior to harvest concentrates sugars for sweeter fruit."
    },
    "banana": {
        "water": "High and frequent watering (keep soil consistently moist, especially during fruit formation).",
        "season": "Humid, tropical warm regions year-round.",
        "harvest": "11 to 14 months after planting.",
        "tips": "Provide windbreaks and heavy organic mulching around the root zone."
    },
    "apple": {
        "water": "Deep watering every 7–10 days during dry spells.",
        "season": "Cool / Temperate high-altitude climates.",
        "harvest": "Perennial tree (produces fruit after 3–5 years, harvested in late summer/autumn).",
        "tips": "Prune annually during winter dormancy to maintain air circulation and sunlight penetration."
    },
    "mango": {
        "water": "Regular watering for young trees; mature trees require watering only during fruit set.",
        "season": "Tropical and subtropical regions with warm summers.",
        "harvest": "Perennial tree (harvest fruits 4 to 5 months after flowering).",
        "tips": "Withhold water during early winter to encourage heavy flowering in spring."
    },
    "maize": {
        "water": "Moderate (requires reliable moisture during tasseling and silking stages).",
        "season": "Warm summer / Monsoon (Kharif season).",
        "harvest": "3 to 4 months after sowing.",
        "tips": "Ensure deep soil aeration and avoid water stagnation at the base."
    },
    "chickpea": {
        "water": "Low to minimal (1 to 2 light irrigations usually suffice).",
        "season": "Winter (Rabi season).",
        "harvest": "3 to 4 months after sowing.",
        "tips": "Avoid over-irrigation as it leads to excess leafy growth instead of pods."
    },
    "potato": {
        "water": "Regular, moderate irrigation to keep ridged soil evenly moist.",
        "season": "Cool season / Winter.",
        "harvest": "3 to 4 months (90 to 120 days).",
        "tips": "Hill up the soil around growing stems to keep developing tubers buried away from sunlight."
    },
    "tomato": {
        "water": "Regular watering at the root base; avoid spraying leaves.",
        "season": "Warm spring/summer or dry winter without frost.",
        "harvest": "2 to 3 months after transplanting.",
        "tips": "Stake plants vertically and apply mulch to conserve moisture and keep fruit off damp soil."
    },
    "cotton": {
        "water": "Moderate water required during vegetative growth; dry weather needed during boll opening.",
        "season": "Warm / Summer season.",
        "harvest": "5 to 6 months.",
        "tips": "High sunshine and dry weather during harvesting ensure clean, quality cotton bolls."
    },
    "coffee": {
        "water": "Moderate to high rainfall; needs a short dry period to trigger uniform flowering.",
        "season": "Humid tropical highlands with partial shade.",
        "harvest": "Perennial (harvest cherries 7 to 9 months after flowering).",
        "tips": "Grow under shade trees to prevent leaf scorching and improve bean flavor."
    }
}

DEFAULT_GUIDE = {
    "water": "Moderate watering — maintain moist, well-drained soil without flooding.",
    "season": "Follow standard regional planting windows.",
    "harvest": "3 to 4 months on average.",
    "tips": "Keep the root zone clear of weeds and incorporate organic compost before planting."
}


# --- STEP 2: TRAIN MODELS WITH PIPELINES ---
@st.cache_resource
def train_all_models(data):
    X = data.drop(columns=["Crop"])
    y = data["Crop"]

    # Encode target variable
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    # STRATIFY added to ensure all classes exist in both train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Preprocessor Pipeline
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    # Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42)
    }

    trained_pipelines = {}
    metrics = {}

    # Get all unique numeric class IDs present in the test set
    unique_test_labels = np.unique(y_test)
    test_target_names = label_enc.inverse_transform(unique_test_labels)

    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        acc = accuracy_score(y_test, preds)

        # Safely pass labels and corresponding target names
        report = classification_report(
            y_test,
            preds,
            labels=unique_test_labels,
            target_names=test_target_names
        )
        cm = confusion_matrix(y_test, preds, labels=unique_test_labels)

        trained_pipelines[name] = pipeline
        metrics[name] = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": cm,
            "predictions": preds,
            "display_labels": test_target_names
        }

    return trained_pipelines, metrics, label_enc, y_test


pipelines, metrics, label_enc, y_test = train_all_models(df)

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Model Comparison", "📊 Data Analysis & Visuals"])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    st.subheader("1. Choose Model Algorithm")
    selected_model_name = st.radio(
        "Select Model:",
        list(pipelines.keys()),
        index=2  # Defaults to Random Forest
    )

    st.subheader("2. Enter Soil & Climate Parameters")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=50.0)
        p = st.number_input("Phosphorous (P)", min_value=0.0, max_value=200.0, value=50.0)
        k = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0)

    if st.button("Predict Recommended Crop", type="primary"):
        input_df = pd.DataFrame([{
            "Nitrogen": n,
            "Phosphorous": p,
            "Potassium": k,
            "Temperature": temp,
            "Humidity": humidity,
            "pH": ph,
            "Rainfall": rainfall
        }])

        chosen_pipeline = pipelines[selected_model_name]
        pred_encoded = chosen_pipeline.predict(input_df)[0]
        crop_raw = label_enc.inverse_transform([pred_encoded])[0]
        crop_key = crop_raw.lower().replace(" ", "_")

        # Display Prediction Banner
        st.success(f"🌱 Recommended Crop using **{selected_model_name}**: **{crop_raw.title()}**")

        # Fetch Growing Information
        guide = CROP_GROWING_GUIDE.get(crop_key, DEFAULT_GUIDE)

        st.markdown("---")
        st.subheader(f"📖 Cultivation & Growing Guide for **{crop_raw.title()}**")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.info(f"💧 **Water Requirement**\n\n{guide['water']}")
        with c2:
            st.warning(f"🗓️ **Ideal Season & Climate**\n\n{guide['season']}")
        with c3:
            st.success(f"⏱️ **Harvest Duration**\n\n{guide['harvest']}")

        st.markdown(f"💡 **Expert Growing Tip:** {guide['tips']}")

# ==================== TAB 2: MODEL COMPARISON ====================
with tab2:
    st.subheader("Model Evaluation & Accuracy Metrics")

    cols = st.columns(3)
    for idx, (m_name, m_data) in enumerate(metrics.items()):
        with cols[idx]:
            st.metric(label=f"{m_name} Accuracy", value=f"{m_data['accuracy'] * 100:.2f}%")
            with st.expander(f"Classification Report"):
                st.code(m_data["report"])

    st.markdown("---")
    st.subheader("Confusion Matrix")
    cm_model_choice = st.selectbox("Select Model for Confusion Matrix:", list(pipelines.keys()))

    fig_cm, ax_cm = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        metrics[cm_model_choice]["confusion_matrix"],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=metrics[cm_model_choice]["display_labels"],
        yticklabels=metrics[cm_model_choice]["display_labels"],
        ax=ax_cm
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{cm_model_choice} Confusion Matrix")
    st.pyplot(fig_cm)

# ==================== TAB 3: VISUALISATIONS ====================
with tab3:
    st.subheader("Dataset Exploratory Data Analysis")

    with st.expander("📄 Dataset Preview"):
        st.dataframe(df.head(10))
        st.write("Summary Statistics:")
        st.dataframe(df.describe())

    st.markdown("---")
    st.subheader("1. Distribution of N, P, K Values")
    fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
    num_cols = ['Nitrogen', 'Phosphorous', 'Potassium']
    for idx, col in enumerate(num_cols):
        axes[idx].hist(df[col], bins=20, color='skyblue', edgecolor='black')
        axes[idx].set_title(f'Distribution of {col}')
    st.pyplot(fig_hist)

    st.markdown("---")
    st.subheader("2. Correlation Heatmap")
    fig_heatmap, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig_heatmap)

    st.markdown("---")
    st.subheader("3. Rainfall Requirements Across Different Crops")
    fig_box, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df, x='Rainfall', y='Crop', ax=ax, palette='Set3')
    ax.set_title('Rainfall Requirements Across Different Crops', fontsize=14, fontweight='bold')
    st.pyplot(fig_box)
