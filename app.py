import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
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

# --- DATASET VALIDATION LIMITS ---
PARAM_LIMITS = {
    "Nitrogen": {"min": float(df["Nitrogen"].min()), "max": float(df["Nitrogen"].max())},
    "Phosphorous": {"min": float(df["Phosphorous"].min()), "max": float(df["Phosphorous"].max())},
    "Potassium": {"min": float(df["Potassium"].min()), "max": float(df["Potassium"].max())},
    "Temperature": {"min": float(df["Temperature"].min()), "max": float(df["Temperature"].max())},
    "Humidity": {"min": float(df["Humidity"].min()), "max": float(df["Humidity"].max())},
    "pH": {"min": float(df["pH"].min()), "max": float(df["pH"].max())},
    "Rainfall": {"min": float(df["Rainfall"].min()), "max": float(df["Rainfall"].max())}
}

# --- STEP 1.5: CROP GROWING GUIDES & PESTICIDE DATABASE ---
CROP_GROWING_GUIDE = {
    "papaya": {
        "water": "Daily light watering in summer; every 2–3 days in cooler months. Keep soil moist, not flooded.",
        "season": "Warm tropical / subtropical climate year-round.",
        "harvest": "9 to 11 months after planting.",
        "pesticide": "Spray **Imidacloprid** or **Neem Oil (10,000 ppm)** to control aphids, mealybugs, and whiteflies transmitting ringspot virus. Apply **Mancozeb** or **Copper Oxychloride** for anthracnose and leaf spot.",
        "tips": "Papayas are sensitive to waterlogging. Ensure excellent soil drainage to prevent root rot."
    },
    "rice": {
        "water": "High / shallow standing water (2–5 cm) required during initial growth stages.",
        "season": "Monsoon / Summer (Kharif season).",
        "harvest": "3 to 5 months after sowing.",
        "pesticide": "Use **Cartap Hydrochloride** or **Chlorantraniliprole** for stem borer and leaf folder. Spray **Tebuconazole** or **Validamycin** for sheath blight.",
        "tips": "Stop watering about 10–14 days before harvest to allow grains to mature and dry."
    },
    "wheat": {
        "water": "Moderate (4 to 6 light irrigations at key growth stages like tillering and flowering).",
        "season": "Winter / Cool season (Rabi season).",
        "harvest": "4 to 5 months after sowing.",
        "pesticide": "Spray **Propiconazole** for yellow/brown rust and leaf blight. Use **Thiamethoxam** for aphid control during grain filling stage.",
        "tips": "Avoid heavy irrigation during harvesting stage to prevent crops from falling over."
    },
    "watermelon": {
        "water": "Deep, regular watering during vine development; reduce water significantly near maturity.",
        "season": "Warm Summer / Spring.",
        "harvest": "80 to 100 days (approx. 3 months).",
        "pesticide": "Apply **Carbaryl** or **Spinosad** for red pumpkin beetle. Spray **Dinocap** or **Hexaconazole** for powdery mildew.",
        "tips": "Cutting back water 1–2 weeks prior to harvest concentrates sugars for sweeter fruit."
    },
    "banana": {
        "water": "High and frequent watering (keep soil consistently moist, especially during fruit formation).",
        "season": "Humid, tropical warm regions year-round.",
        "harvest": "11 to 14 months after planting.",
        "pesticide": "Inject or drench **Carbofuran** for banana pseudostem weevil and root nematodes. Spray **Propiconazole** for Sigatoka leaf spot disease.",
        "tips": "Provide windbreaks and heavy organic mulching around the root zone."
    },
    "apple": {
        "water": "Deep watering every 7–10 days during dry spells.",
        "season": "Cool / Temperate high-altitude climates.",
        "harvest": "Perennial tree (produces fruit after 3–5 years, harvested in late summer/autumn).",
        "pesticide": "Apply **Captan** or **Difenoconazole** for scab and powdery mildew. Use **Chlorpyrifos** or **Spirotermát** for San Jose scale and woolly aphids.",
        "tips": "Prune annually during winter dormancy to maintain air circulation and sunlight penetration."
    },
    "mango": {
        "water": "Regular watering for young trees; mature trees require watering only during fruit set.",
        "season": "Tropical and subtropical regions with warm summers.",
        "harvest": "Perennial tree (harvest fruits 4 to 5 months after flowering).",
        "pesticide": "Spray **Imidacloprid** or **Lambda-cyhalothrin** during flowering against mango hoppers. Use **Carbendazim** or **Copper Oxychloride** for anthracnose.",
        "tips": "Withhold water during early winter to encourage heavy flowering in spring."
    },
    "maize": {
        "water": "Moderate (requires reliable moisture during tasseling and silking stages).",
        "season": "Warm summer / Monsoon (Kharif season).",
        "harvest": "3 to 4 months after sowing.",
        "pesticide": "Apply **Emamectin Benzoate** or **Spinetoram** to combat Fall Armyworm (FAW). Drench **Phorate** granules in leaf whorls if infestation is severe.",
        "tips": "Ensure deep soil aeration and avoid water stagnation at the base."
    },
    "chickpea": {
        "water": "Low to minimal (1 to 2 light irrigations usually suffice).",
        "season": "Winter (Rabi season).",
        "harvest": "3 to 4 months after sowing.",
        "pesticide": "Spray **Indoxacarb** or **Flubendiamide** for pod borer (*Helicoverpa armigera*). Drench soil with **Trichoderma viride** or **Carbendazim** for wilt disease.",
        "tips": "Avoid over-irrigation as it leads to excess leafy growth instead of pods."
    },
    "potato": {
        "water": "Regular, moderate irrigation to keep ridged soil evenly moist.",
        "season": "Cool season / Winter.",
        "harvest": "3 to 4 months (90 to 120 days).",
        "pesticide": "Spray **Mancozeb** or **Cymoxanil + Mancozeb** preventatively for Late Blight (*Phytophthora*). Use **Imidacloprid** for aphids transmitting leaf roll virus.",
        "tips": "Hill up the soil around growing stems to keep developing tubers buried away from sunlight."
    },
    "tomato": {
        "water": "Regular watering at the root base; avoid spraying leaves.",
        "season": "Warm spring/summer or dry winter without frost.",
        "harvest": "2 to 3 months after transplanting.",
        "pesticide": "Spray **Cyantraniliprole** or **Emamectin Benzoate** for fruit borer (*Tuta absoluta*). Apply **Copper hydroxide** for bacterial leaf spot.",
        "tips": "Stake plants vertically and apply mulch to conserve moisture and keep fruit off damp soil."
    },
    "cotton": {
        "water": "Moderate water required during vegetative growth; dry weather needed during boll opening.",
        "season": "Warm / Summer season.",
        "harvest": "5 to 6 months.",
        "pesticide": "Use **Profex Super (Profenofos + Cypermethrin)** or **Spinetoram** against bollworms. Use **Diafenthiuron** for whiteflies and aphids.",
        "tips": "High sunshine and dry weather during harvesting ensure clean, quality cotton bolls."
    },
    "coffee": {
        "water": "Moderate to high rainfall; needs a short dry period to trigger uniform flowering.",
        "season": "Humid tropical highlands with partial shade.",
        "harvest": "Perennial (harvest cherries 7 to 9 months after flowering).",
        "pesticide": "Apply **Triadimefon** or **Hexaconazole** for coffee leaf rust. Use **Chlorpyrifos** or **Beauveria bassiana** for coffee berry borer.",
        "tips": "Grow under shade trees to prevent leaf scorching and improve bean flavor."
    }
}

DEFAULT_GUIDE = {
    "water": "Moderate watering — maintain moist, well-drained soil without flooding.",
    "season": "Follow standard regional planting windows.",
    "harvest": "3 to 4 months on average.",
    "pesticide": "Apply **Neem Oil spray** (5ml/L) bi-weekly for general sucking pests. Use **Mancozeb** or **Carbendazim** if fungal spots appear.",
    "tips": "Keep the root zone clear of weeds and incorporate organic compost before planting."
}


# --- STEP 2: TRAIN MODELS WITH PIPELINES & IDENTIFY BEST MODEL ---
@st.cache_resource
def train_all_models(data):
    X = data.drop(columns=["Crop"])
    y = data["Crop"]

    # Encode target variable
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Multi-Layer Perceptron (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }

    trained_pipelines = {}
    metrics = {}
    best_model_name = None
    best_accuracy = -1.0

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

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name

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

    return trained_pipelines, metrics, label_enc, y_test, best_model_name, best_accuracy


pipelines, metrics, label_enc, y_test, best_model_name, best_accuracy = train_all_models(df)

# --- NAVIGATION TABS ---
tab1, tab2, tab3 = st.tabs(["🔮 Make Prediction", "📈 Model Comparison", "📊 Data Analysis & Visuals"])

# ==================== TAB 1: PREDICTION ====================
with tab1:
    # Display Active Model Efficiency
    st.success(
        f"🎯 **Automated Model Selection Active:** Predictions will be performed using **{best_model_name}** "
        f"(Highest Accuracy: **{best_accuracy * 100:.2f}%**)."
    )

    st.subheader("Enter Soil & Climate Parameters")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=-50.0, max_value=500.0, value=50.0)
        p = st.number_input("Phosphorous (P)", min_value=-50.0, max_value=500.0, value=50.0)
        k = st.number_input("Potassium (K)", min_value=-50.0, max_value=500.0, value=50.0)
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=100.0, value=25.0)

    with col2:
        humidity = st.number_input("Humidity (%)", min_value=-20.0, max_value=150.0, value=70.0)
        ph = st.number_input("pH Level", min_value=-5.0, max_value=20.0, value=6.5)
        rainfall = st.number_input("Rainfall (mm)", min_value=-100.0, max_value=5000.0, value=100.0)

    if st.button("Predict Recommended Crop", type="primary"):
        input_values = {
            "Nitrogen": n,
            "Phosphorous": p,
            "Potassium": k,
            "Temperature": temp,
            "Humidity": humidity,
            "pH": ph,
            "Rainfall": rainfall
        }

        # Validate inputs against original dataset boundaries
        invalid_params = []
        for param, val in input_values.items():
            min_val = PARAM_LIMITS[param]["min"]
            max_val = PARAM_LIMITS[param]["max"]
            if val < min_val or val > max_val:
                invalid_params.append(f"**{param}** (Allowed range: {min_val:.2f} to {max_val:.2f})")

        if invalid_params:
            for error_msg in invalid_params:
                st.error(f"❌ Please enter valid inputs for {error_msg}")
        else:
            input_df = pd.DataFrame([input_values])

            # Automatically select best performing pipeline
            chosen_pipeline = pipelines[best_model_name]
            pred_encoded = chosen_pipeline.predict(input_df)[0]
            crop_raw = label_enc.inverse_transform([pred_encoded])[0]
            crop_key = crop_raw.lower().replace(" ", "_")

            st.success(
                f"🌱 Recommended Crop using **{best_model_name}** ({best_accuracy * 100:.2f}% accuracy): **{crop_raw.title()}**")

            guide = CROP_GROWING_GUIDE.get(crop_key, DEFAULT_GUIDE)

            st.markdown("---")
            st.subheader(f"📖 Cultivation & Growing Guide for **{crop_raw.title()}**")

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.info(f"💧 **Water Requirement**\n\n{guide['water']}")
            with c2:
                st.warning(f"🗓️ **Ideal Season & Climate**\n\n{guide['season']}")
            with c3:
                st.success(f"⏱️ **Harvest Duration**\n\n{guide['harvest']}")
            with c4:
                st.error(f"🛡️ **Pesticide & Pest Control**\n\n{guide['pesticide']}")

            st.markdown(f"💡 **Expert Growing Tip:** {guide['tips']}")

# ==================== TAB 2: MODEL COMPARISON ====================
with tab2:
    st.subheader("Model Evaluation & Accuracy Metrics")

    cols = st.columns(len(metrics))
    for idx, (m_name, m_data) in enumerate(metrics.items()):
        with cols[idx]:
            if m_name == best_model_name:
                st.metric(label=f"⭐ {m_name} (Best)", value=f"{m_data['accuracy'] * 100:.2f}%")
            else:
                st.metric(label=f"{m_name}", value=f"{m_data['accuracy'] * 100:.2f}%")

            with st.expander("Classification Report"):
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
