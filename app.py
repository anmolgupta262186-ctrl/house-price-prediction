import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

model = pickle.load(open("house_price_model.pkl","rb"))

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.title("🏠 Smart House Price Prediction Dashboard")
st.write("Adjust house features to estimate the property price.")

st.divider()

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    GrLivArea = st.number_input("Living Area (sq ft)", min_value=0.0, value=1500.0)
    GarageCars = st.number_input("Garage Capacity", min_value=0.0, value=2.0)
    FullBath = st.number_input("Bathrooms", min_value=0.0, value=2.0)

with col2:
    OverallQual = st.slider("Overall Quality",1,10,5)
    TotalBsmtSF = st.number_input("Basement Area", min_value=0.0, value=800.0)
    YearBuilt = st.number_input("Year Built", min_value=1800, value=2000)

with col3:
    FirstFlrSF = st.number_input("1st Floor Area", min_value=0.0, value=1000.0)
    TotRmsAbvGrd = st.number_input("Rooms Above Ground", min_value=0.0, value=6.0)
    GarageArea = st.number_input("Garage Area", min_value=0.0, value=500.0)
    LotArea = st.number_input("Lot Area", min_value=0.0, value=5000.0)

st.divider()

if st.button("🔮 Predict House Price"):

    features = np.array([[GrLivArea,OverallQual,GarageCars,TotalBsmtSF,
                          FullBath,YearBuilt,FirstFlrSF,TotRmsAbvGrd,
                          GarageArea,LotArea]])

    prediction = model.predict(features)[0]

    st.success(f"💰 Estimated House Price: ${prediction:,.2f}")

    st.subheader("📊 Price Indicator")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={'prefix':"$"},
        gauge={
            'axis': {'range': [0, 600000]},
            'bar': {'color': "green"},
            'steps': [
                {'range': [0, 150000], 'color': "#ffcccc"},
                {'range': [150000, 300000], 'color': "#ffe6cc"},
                {'range': [300000, 450000], 'color': "#ccffcc"},
                {'range': [450000, 600000], 'color': "#b3e6ff"}
            ],
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Feature Overview")

    feature_data = {
        "Feature": ["Living Area","Garage Capacity","Bathrooms","Quality","Basement Area","Rooms"],
        "Value": [GrLivArea,GarageCars,FullBath,OverallQual,TotalBsmtSF,TotRmsAbvGrd]
    }

    bar_chart = px.bar(feature_data, x="Feature", y="Value", color="Feature")
    st.plotly_chart(bar_chart, use_container_width=True)

st.caption("Machine Learning Model: Random Forest | Dataset: Ames Housing")