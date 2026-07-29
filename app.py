import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import requests

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- STREAMLIT PAGE SETUP ---
st.set_page_config(page_title="Crop Recommendation System", layout="wide")
st.title("🌾 Crop Recommendation System")
st.write("Predict the best crop for your soil and climate conditions using Machine Learning & Live Weather Data!")


# --- HELPER FUNCTIONS FOR LOCATION & WEATHER API ---
def get_accurate_location_weather(country, state, city):
    """Fetches exact weather parameters using full location search query."""
    try:
        # Build precise location query
        full_location_query = f"{city.strip()}, {state.strip()}, {country.strip()}"

        # 1. Precise Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(full_location_query)}&count=1&language=en&format=json"
        geo_res = requests.get(geo_url, timeout=5).json()

        # Fallback search if full query didn't match directly
        if "results" not in geo_res or not geo_res["results"]:
            fallback_query = f"{city.strip()}, {country.strip()}"
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(fallback_query)}&count=1&language=en&format=json"
            geo_res = requests.get(geo_url, timeout=5).json()

        if "results" not in geo_res or not geo_res["results"]:
            return None, f"Could not find coordinates for '{city}, {state}, {country}'. Please check spelling."

        loc_data = geo_res["results"][0]
        lat = loc_data["latitude"]
        lon = loc_data["longitude"]
        resolved_city = loc_data.get("name", city)
        resolved_admin = loc_data.get("admin1", state)
        resolved_country = loc_data.get("country", country)

        # 2. Weather Fetching
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m&daily=rain_sum&timezone=auto"
        w_res = requests.get(weather_url, timeout=5).json()

        temp = round(float(w_res["current"]["temperature_2m"]), 2)
        humidity = round(float(w_res["current"]["relative_humidity_2m"]), 2)

        # Daily rain sum scaled for seasonal baseline estimation
        rain_sum = w_res["daily"]["rain_sum"][0] if "daily" in w_res and "rain_sum" in w_res["daily"] else 10.0
        estimated_rainfall = round(max(float(rain_sum) * 15.0, 50.0), 2)

        return {
            "display_name": f"{resolved_city}, {resolved_admin}, {resolved_country}",
            "lat": lat,
            "lon": lon,
            "temperature": temp,
            "humidity": humidity,
            "rainfall": estimated_rainfall
        }, None

    except Exception as e:
        return None, f"Error connecting to weather service: {str(e)}"


# --- STEP 1: LOAD & PREPROCESS DATA ---
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("merged_crop_dataset.csv")

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

# --- CROP GROWING GUIDES & PESTICIDE DATABASE ---
CROP_GROWING_GUIDE = {
    "papaya": {
        "water": "Daily light watering in summer; every 2–3 days in cooler months.",
        "season": "Warm tropical / subtropical climate year-round.",
        "harvest": "9 to 11 months after planting.",
        "pesticide": "Spray **Imidacloprid** or **Neem Oil** for aphids and mealybugs. Use **Mancozeb** for anthracnose.",
        "tips": "Papayas are sensitive to waterlogging. Ensure excellent soil drainage to prevent root rot."
    },
    "rice": {
        "water": "High / shallow standing water (2–5 cm) required during initial growth stages.",
        "season": "Monsoon / Summer (Kharif season).",
        "harvest": "3 to 5 months after sowing.",
        "pesticide": "Use **Cartap Hydrochloride** for stem borer. Spray **Validamycin** for sheath blight.",
        "tips": "Stop watering about 10–14 days before harvest to allow grains to mature and dry."
    },
    "jute": {
        "water": "High water requirement; regular heavy watering or rainfall during early stages.",
        "season": "Warm, humid monsoon season.",
        "harvest": "4 to 5 months after sowing.",
        "pesticide": "Spray **Endosulfan** or **Neem Oil** for jute semilooper and red spider mite.",
        "tips": "Harvest when 50% of plants are in pod formation for best fiber quality."
    }
}

DEFAULT_GUIDE = {
    "water": "Moderate watering — maintain moist, well-drained soil without flooding.",
    "season": "Follow standard regional planting windows.",
    "harvest": "3 to 4 months on average.",
    "pesticide": "Apply **Neem Oil spray** bi-weekly. Use **Mancozeb** if fungal spots appear.",
    "tips": "Keep the root zone clear of weeds and incorporate organic compost before planting."
}


# --- STEP 2: TRAIN MODELS WITH PIPELINES & IDENTIFY BEST MODEL ---
@st.cache_resource
def train_all_models(data):
    X = data.drop(columns=["Crop"])
    y = data["Crop"]

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
        "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )}

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

        report = classification_report(y_test, preds, labels=unique_test_labels, target_names=test_target_names)
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
    # --- MODEL SELECTION UI ---
    st.subheader("🤖 Model Selection")
    
    # Selection menu allowing automatic selection or user choice
    model_options = [f"Best Model (Auto: {best_model_name})"] + list(pipelines.keys())
    selected_option = st.selectbox(
        "Choose the ML model to use for prediction:",
        options=model_options,
        index=0
    )

    # Determine which model pipeline to run
    if selected_option.startswith("Best Model"):
        active_model_name = best_model_name
    else:
        active_model_name = selected_option

    active_accuracy = metrics[active_model_name]["accuracy"]

    st.info(
        f"🎯 Active Model: **{active_model_name}** "
        f"(Test Accuracy: **{active_accuracy * 100:.2f}%**)"
    )

    # Initialize session state for climate values if not present
    if "temp_val" not in st.session_state:
        st.session_state["temp_val"] = 25.0
    if "hum_val" not in st.session_state:
        st.session_state["hum_val"] = 70.0
    if "rain_val" not in st.session_state:
        st.session_state["rain_val"] = 100.0

    st.markdown("---")

    # --- STEP 1: DETAILED LOCATION FORM ---
    st.subheader("📍 Step 1: Select Your Location (Country, State, City)")

    loc_col1, loc_col2, loc_col3 = st.columns(3)
    with loc_col1:
        country_in = st.text_input("Country:", value="India")
    with loc_col2:
        state_in = st.text_input("State / Province:", value="Uttar Pradesh")
    with loc_col3:
        city_in = st.text_input("City / District:", value="Bareilly")

    if st.button("🌐 Fetch Weather For This Location", type="secondary"):
        if not country_in or not state_in or not city_in:
            st.warning("⚠️ Please fill in Country, State, and City.")
        else:
            w_data, err = get_accurate_location_weather(country_in, state_in, city_in)
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state["temp_val"] = w_data["temperature"]
                st.session_state["hum_val"] = w_data["humidity"]
                st.session_state["rain_val"] = w_data["rainfall"]
                st.info(
                    f"✅ Exact Location Confirmed: **{w_data['display_name']}** (Lat: {w_data['lat']}, Lon: {w_data['lon']})")

    st.markdown("---")

    # --- STEP 2: SOIL & CLIMATE VALUES ---
    st.subheader("🧪 Step 2: Soil & Climate Parameters")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Soil Parameters**")
        n = st.number_input("Nitrogen (N)", min_value=-50.0, max_value=500.0, value=50.0)
        p = st.number_input("Phosphorous (P)", min_value=-50.0, max_value=500.0, value=50.0)
        k = st.number_input("Potassium (K)", min_value=-50.0, max_value=500.0, value=50.0)
        ph = st.number_input("pH Level", min_value=-5.0, max_value=20.0, value=6.5)

    with col2:
        st.markdown("**Climate Values (Fetched / Editable)**")
        temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=100.0, value=st.session_state["temp_val"])
        humidity = st.number_input("Humidity (%)", min_value=-20.0, max_value=150.0, value=st.session_state["hum_val"])
        rainfall = st.number_input("Rainfall (mm)", min_value=-100.0, max_value=5000.0,
                                   value=st.session_state["rain_val"])

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

        # Validate bounds
        invalid_params = []
        for param, val in input_values.items():
            min_val = PARAM_LIMITS[param]["min"]
            max_val = PARAM_LIMITS[param]["max"]
            if val < min_val or val > max_val:
                invalid_params.append(f"**{param}** (Allowed: {min_val:.2f} to {max_val:.2f})")

        if invalid_params:
            for error_msg in invalid_params:
                st.error(f"❌ Please enter valid inputs for {error_msg}")
        else:
            input_df = pd.DataFrame([input_values])

            # Use the pipeline selected from the dropdown
            chosen_pipeline = pipelines[active_model_name]
            pred_encoded = chosen_pipeline.predict(input_df)[0]
            crop_raw = label_enc.inverse_transform([pred_encoded])[0]
            crop_key = crop_raw.lower().replace(" ", "_")

            st.success(
                f"🌱 Recommended Crop using **{active_model_name}** ({active_accuracy * 100:.2f}% accuracy): **{crop_raw.title()}**")

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
