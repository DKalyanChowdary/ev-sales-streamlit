# %%
# ev_sales_app.py
# 🚗 Electric Vehicle (EV) Sales Forecasting App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import os, requests

# %%
# 🔹 Load models (auto-download from Google Drive if missing)

def download_from_drive(file_id, save_path):
    """Download file from Google Drive by file ID using gdown"""
    import gdown
    if not os.path.exists(save_path):
        with st.spinner(f"⬇️ Downloading {save_path} from Google Drive..."):
            try:
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, save_path, quiet=False)
                st.success(f"✅ {save_path} downloaded successfully.")
            except Exception as e:
                st.error(f"❌ Failed to download {save_path}: {e}")

# 🔹 Google Drive file IDs
EV_MODEL_ID = "1WCwemjoUHDOhUgijAbatakQ9Ezwmjvjl"       # ev_sales_model.pkl
FORECAST_MODEL_ID = "1-tYWk9NXPgDPKtfdjn9Qb74cxQkr4r04"  # forecast_model.pkl

# 🔹 Download models if not present
download_from_drive(EV_MODEL_ID, "ev_sales_model.pkl")
download_from_drive(FORECAST_MODEL_ID, "forecast_model.pkl")

# 🔹 Load models after download
try:
    model_rf = joblib.load("ev_sales_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading Random Forest model: {e}")
    model_rf = None

try:
    model_prophet = joblib.load("forecast_model.pkl")
except Exception as e:
    st.error(f"❌ Error loading Prophet model: {e}")
    model_prophet = None


# %%
# 🔹 Load dataset
df = pd.read_csv("C:/Users/kalay/OneDrive/Desktop/unified p/Electric Vehicle Sales by State in India.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month_Name'] = df['Date'].dt.month_name()

# %%
# --- Streamlit UI ---
st.title("🚗 Electric Vehicle Sales Forecasting Dashboard")
st.markdown("### Analyze and Predict EV Sales Across India")

# %%
# 🔹 Sidebar filters
st.sidebar.header("Filter Options")
state = st.sidebar.selectbox("Select State", sorted(df['State'].unique()))
category = st.sidebar.selectbox("Select Vehicle Category", sorted(df['Vehicle_Category'].unique()))

# %%
# 🔹 Filter dataset
filtered_df = df[(df['State'] == state) & (df['Vehicle_Category'] == category)]

# 📈 EV Sales Trend
st.subheader(f"📊 EV Sales Trend for {state} ({category})")
fig, ax = plt.subplots(figsize=(10,4))
sns.lineplot(data=filtered_df, x='Date', y='EV_Sales_Quantity', ax=ax, color='teal')
ax.set_title(f"EV Sales Trend: {state} ({category})")
st.pyplot(fig)

# %%
# 🔹 Time Series Forecast
ts = filtered_df.groupby('Date')['EV_Sales_Quantity'].sum().reset_index()
ts = ts.rename(columns={'Date': 'ds', 'EV_Sales_Quantity': 'y'})

if len(ts) > 10:
    model = Prophet(yearly_seasonality=True)
    model.fit(ts)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    plt.title(f"🔮 Forecasted EV Sales for {state} ({category}) - Next 12 Months")
    st.pyplot(fig1)

    # 🔹 Display future predictions table
    st.write("### Forecasted Sales (Next 12 Months):")
    forecast_df = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12)
    forecast_df.columns = ['Date','Predicted Sales','Lower Range','Upper Range']
    st.dataframe(forecast_df)
else:
    st.warning("⚠️ Not enough data for forecasting this selection.")

# %%
# 🔹 Feature Importance (using saved Random Forest model)
st.subheader("🌟 Top Features Influencing EV Sales")

try:
    # Load feature names from training
    feature_names = joblib.load("model_features.pkl")

    # Ensure length alignment
    if len(model_rf.feature_importances_) == len(feature_names):
        feat_importance = pd.Series(model_rf.feature_importances_, index=feature_names).sort_values(ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8,4))
        feat_importance.head(15).plot(kind='barh', ax=ax2, color='orange')
        ax2.set_title("🌟 Top 15 Features Influencing EV Sales")
        st.pyplot(fig2)
    else:
        st.warning("Feature count mismatch between model and dataset.")

except Exception as e:
    st.info(f"Feature importance visualization unavailable for this data selection. ({e})")

