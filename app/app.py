import streamlit as st
import pandas as pd
import joblib
import os

# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

# ==================================================
# PROJECT PATHS
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_pipeline_shap.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "notebooks",
    "crop_yield_data.csv"
)

HISTORICAL_PATH = os.path.join(
    BASE_DIR,
    "notebooks",
    "historical_yield.csv"
)

# ==================================================
# LOAD MODEL AND DATA
# ==================================================

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

historical_yield = pd.read_csv(
    HISTORICAL_PATH
)

# ==================================================
# TITLE
# ==================================================

st.title(
    "🌾 Crop Yield Prediction "
)

st.write(
    "This system uses an XGBoost machine learning model "
    "to predict crop yield, compare crops, provide crop "
    "recommendations, and perform What-If analysis."
)

st.success(
    "Model and datasets loaded successfully!"
)

# ==================================================
# FARMING CONDITIONS
# ==================================================

st.header("🌱 Enter Farming Conditions")

col1, col2 = st.columns(2)

# LEFT COLUMN

with col1:

    crop = st.selectbox(
        "Crop",
        sorted(df["Crop"].dropna().unique())
    )

    year = st.number_input(
        "Year",
        min_value=1997,
        max_value=2030,
        value=2020,
        step=1
    )

    state = st.selectbox(
        "State",
        sorted(df["State"].dropna().unique())
    )

    available_seasons = sorted(
        df.loc[
            df["State"] == state,
            "Season"
        ].dropna().unique()
    )

    season = st.selectbox(
        "Season",
        available_seasons
    )

    area = st.number_input(
        "Area (hectares)",
        min_value=0.0,
        value=2.0
    )

    fertilizer = st.number_input(
        "Fertilizer (kg)",
        min_value=0.0,
        value=100.0
    )

    pesticide = st.number_input(
        "Pesticide (kg)",
        min_value=0.0,
        value=20.0
    )

# RIGHT COLUMN

with col2:

    temperature = st.number_input(
        "Average Temperature (°C)",
        value=28.0
    )

    rainfall = st.number_input(
        "Total Rainfall (mm)",
        min_value=0.0,
        value=700.0
    )

    humidity = st.number_input(
        "Average Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=70.0
    )

    nitrogen = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=80.0
    )

    phosphorus = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=40.0
    )

    potassium = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=40.0
    )

    soil_ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5
    )

# ==================================================
# CREATE INPUT DATA
# ==================================================

input_data = pd.DataFrame([{
    "Crop": crop,
    "Year": year,
    "Season": season,
    "State": state,
    "Area (hectares)": area,
    "Fertilizer (kg)": fertilizer,
    "Pesticide (kg)": pesticide,
    "Avg Temperature (C)": temperature,
    "Total Rainfall (mm)": rainfall,
    "Avg Humidity (%)": humidity,
    "Nitrogen (N)": nitrogen,
    "Phosphorus (P)": phosphorus,
    "Potassium (K)": potassium,
    "Soil pH": soil_ph
}])

# ==================================================
# CROP YIELD PREDICTION
# ==================================================

st.header("🔮 Crop Yield Prediction")

if st.button(
    "Predict Yield",
    type="primary"
):

    predicted_yield = model.predict(
        input_data
    )[0]

    st.success(
        f"Predicted Yield: "
        f"{predicted_yield:.2f} tonnes/hectare"
    )

# ==================================================
# CROP RECOMMENDATION AND RANKING
# ==================================================

st.header("🌾 Crop Recommendation and Ranking")

st.write(
    "The system predicts the yield of crops available "
    "in the selected state and season, then ranks them "
    "according to their predicted yield under the "
    "entered farming conditions."
)

if st.button("Recommend Crops"):

    available_crops = df.loc[
        (df["State"] == state) &
        (df["Season"] == season),
        "Crop"
    ].dropna().unique()

    recommendations = []

    for selected_crop in available_crops:

        crop_input = pd.DataFrame([{
            "Crop": selected_crop,
            "Year": year,
            "Season": season,
            "State": state,
            "Area (hectares)": area,
            "Fertilizer (kg)": fertilizer,
            "Pesticide (kg)": pesticide,
            "Avg Temperature (C)": temperature,
            "Total Rainfall (mm)": rainfall,
            "Avg Humidity (%)": humidity,
            "Nitrogen (N)": nitrogen,
            "Phosphorus (P)": phosphorus,
            "Potassium (K)": potassium,
            "Soil pH": soil_ph
        }])

        predicted = model.predict(
            crop_input
        )[0]

        recommendations.append({
            "Crop": selected_crop,
            "Predicted Yield": float(predicted)
        })

    recommendation_df = pd.DataFrame(
        recommendations
    )

    if not recommendation_df.empty:

        recommendation_df = (
            recommendation_df
            .sort_values(
                by="Predicted Yield",
                ascending=False
            )
            .reset_index(drop=True)
        )

        recommendation_df.insert(
            0,
            "Rank",
            range(1, len(recommendation_df) + 1)
        )

        recommendation_df[
            "Predicted Yield"
        ] = recommendation_df[
            "Predicted Yield"
        ].round(2)

        st.subheader("📋 Crop Ranking")

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )

        top_crop = recommendation_df.iloc[0]

        st.success(
            f"Highest model-predicted yield under "
            f"the selected conditions: "
            f"{top_crop['Crop']} "
            f"({top_crop['Predicted Yield']:.2f} "
            f"tonnes/hectare)"
        )

    else:

        st.warning(
            "No crops are available for the "
            "selected state and season."
        )

# ==================================================
# HISTORICAL AVERAGE YIELD
# ==================================================

st.header("📈 Historical Average Yield")

historical_crop = historical_yield[
    historical_yield["Crop"] == crop
]

if not historical_crop.empty:

    historical_value = historical_crop[
        "Historical Average Yield (tonnes/hectare)"
    ].iloc[0]

    st.metric(
        "Historical Average Yield",
        f"{historical_value:.2f} tonnes/hectare"
    )

else:

    st.info(
        "Historical yield data is not available "
        "for the selected crop."
    )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.write(
    "Developed as an AI-powered crop yield "
    "prediction and optimization system."
)
