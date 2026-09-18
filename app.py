"""
Sydney Housing Price Predictor - Streamlit App
SIT307 8.1 Distinction Task - Part 5 Deployment

Run with: streamlit run app.py
(requires model.pkl in the same folder — exported from the notebook's Part 5 cell)
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠")

st.title("🏠 Sydney Housing Price Predictor")
st.write("Predicts sale price for Newtown, Castle Hill, and Cronulla properties based on a Gradient Boosting model trained on 97 manually collected sold listings.")

try:
    model = joblib.load("model.pkl")
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.warning("model.pkl not found — run the notebook's Part 5 export cell first.")

# Suburb-level distance constants (matches the notebook's feature engineering)
SUBURB_DISTANCES = {
    "Newtown":     {"cbd": 4,  "train": 0.3, "beach": 8.0},
    "Castle Hill": {"cbd": 32, "train": 1.5, "beach": 35.0},
    "Cronulla":    {"cbd": 28, "train": 0.8, "beach": 0.5},
}

with st.form("property_form"):
    col1, col2 = st.columns(2)

    with col1:
        suburb = st.selectbox("Suburb", ["Newtown", "Castle Hill", "Cronulla"])
        property_type = st.selectbox(
            "Property Type",
            ["House", "Unit", "Semi-detached", "Studio", "Townhouse", "Villa", "Duplex"]
        )
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=10, value=3)
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

    with col2:
        car_spaces = st.number_input("Car Spaces", min_value=0, max_value=10, value=1)
        land_size_sqm = st.number_input(
            "Land Size (sqm) — leave at 0 if unknown/apartment", min_value=0, value=0
        )

    submitted = st.form_submit_button("Predict Price")

if submitted:
    if model_loaded:
        dist = SUBURB_DISTANCES[suburb]
        # land_size_sqm_filled: if user left it at 0, fall back to a rough median so
        # the imputation logic mirrors what the notebook did for missing values
        land_filled = land_size_sqm if land_size_sqm > 0 else 100

        input_df = pd.DataFrame([{
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "car_spaces": car_spaces,
            "land_size_sqm_filled": land_filled,
            "distance_to_cbd_km": dist["cbd"],
            "distance_to_beach_km": dist["beach"],
            "distance_to_train_km": dist["train"],
            "suburb": suburb,
            "property_type": property_type,
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"### Predicted Sale Price: ${prediction:,.0f}")

        rare_types = {"Townhouse", "Villa", "Duplex"}
        if property_type in rare_types:
            st.caption(
                f"⚠️ Note: only 1 {property_type} exists in the training data, "
                "so this prediction has very low confidence."
            )
    else:
        st.error("No model loaded — can't predict yet.")

st.caption("Built for SIT307 8.1 Distinction Task — for educational purposes only, not real valuation advice.")
