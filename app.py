import streamlit as st
import pandas as pd
from apputil import GroupEstimate

st.write(
    """
    # Week X: Group Estimate Demo

    Upload a dataset with categorical columns (X) and one continuous column (y).
    We'll fit a simple model that predicts group-level mean/median values.
    """
)

# --- File upload ---
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of uploaded data")
    st.dataframe(df.head())

    # --- Select columns ---
    cat_cols = st.multiselect("Select categorical columns (X)", df.columns)
    target_col = st.selectbox("Select target column (y)", df.columns)

    # --- Choose estimate type ---
    estimate_type = st.radio("Estimate type", ["mean", "median"])

    if st.button("Fit Model"):
        if not cat_cols or not target_col:
            st.error("Please select categorical columns and a target column.")
        else:
            # Fit the model
            ge = GroupEstimate(estimate=estimate_type)
            ge.fit(df[cat_cols], df[target_col])
            st.success("Model fitted successfully!")

            # --- Prediction input ---
            st.write("### Try Predictions")
            st.write("Enter values for the categorical columns:")

            input_data = {}
            for col in cat_cols:
                options = df[col].unique().tolist()
                input_data[col] = st.selectbox(f"{col}", options)

            if st.button("Predict"):
                X_new = pd.DataFrame([input_data])
                pred = ge.predict(X_new)
                st.write(f"Predicted {estimate_type}: **{pred[0]}**")