import streamlit as st
import pandas as pd
from forecast_utils import (
    preprocess_data, append_and_train, load_trained_model,
    forecast_sales, calculate_target_analysis,
    generate_recommendations, plot_forecast,
    plot_daily_bar_chart, generate_daily_table
)

st.set_page_config(page_title="📊 Smart Forecast App", layout="wide")
st.title("📊 Forecast App with Historical Learning")

uploaded_file = st.file_uploader("📤 Upload Sales Data", type=["csv", "xlsx"])
if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    st.success("✅ Data uploaded successfully")

    date_col = st.selectbox("Select Date Column", df.select_dtypes(include=["object", "datetime"]).columns)
    target_col = st.selectbox("Select Target Column", df.select_dtypes("number").columns)

    df_clean = preprocess_data(df, date_col, target_col)

    choice = st.radio("How should the model use this data?", ["Use trained model only", "Append & retrain with this data"])

    if st.button("🚀 Proceed with Forecast"):
        if choice == "Append & retrain with this data":
            model, full_data = append_and_train(df_clean, append=True)
        else:
            model, full_data = load_trained_model()
            if model is None:
                st.error("❌ No trained model found. Please retrain first.")
                st.stop()

        st.success("Model ready. Generating forecast...")

        last_date = pd.to_datetime(full_data['date'].max())
        target_mode = st.radio("🎯 Forecast Period", ["Monthly", "Yearly"])
        target_value = st.number_input("📌 Enter Sales Target", step=1000)

        forecast_df, forecast_days = forecast_sales(model, last_date, target_mode)

        if forecast_df.empty:
            st.warning("⚠️ Not enough future days to forecast.")
        else:
            st.subheader("📌 Target Analysis")
            metrics = calculate_target_analysis(full_data, forecast_df, last_date, target_value, target_mode)

            for k, v in metrics.items():
                st.metric(label=k, value=v)

            st.success(generate_recommendations(metrics))
            st.plotly_chart(plot_forecast(forecast_df), use_container_width=True)
            st.plotly_chart(plot_daily_bar_chart(full_data), use_container_width=True)
            st.subheader("📋 Daily Forecast Table")
            st.dataframe(generate_daily_table(forecast_df))
else:
    st.info("Upload a file to get started.")
