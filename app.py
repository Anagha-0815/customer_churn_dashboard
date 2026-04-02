import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="Telco Churn Dashboard", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    if os.path.exists('data/customer_churn_data.csv'):
        df = pd.read_csv('data/customer_churn_data.csv')
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        return df
    return None

@st.cache_resource
def load_model():
    if os.path.exists('models/churn_model_pipeline.joblib'):
        return joblib.load('models/churn_model_pipeline.joblib')
    return None

df = load_data()
model = load_model()

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-title {
        font-size: 1.1rem;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)


st.title("📊 Customer Churn Prediction & Analytics")

tab1, tab2 = st.tabs(["📈 Analytics Dashboard", "🔮 Prediction Tool"])

with tab1:
    st.header("Overall Customer Insights")
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
             st.markdown(f'<div class="metric-card"><div class="metric-title">Total Customers</div><div class="metric-value">{len(df)}</div></div>', unsafe_allow_html=True)
        with col2:
             churn_rate = (df['Churn'] == 'Yes').mean() * 100
             st.markdown(f'<div class="metric-card"><div class="metric-title">Overall Churn Rate</div><div class="metric-value">{churn_rate:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
             avg_clv = df['TotalCharges'].mean()
             st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Lifetime Value</div><div class="metric-value">${avg_clv:.0f}</div></div>', unsafe_allow_html=True)
        with col4:
            lost_revenue = df[df['Churn'] == 'Yes']['TotalCharges'].sum()
            st.markdown(f'<div class="metric-card"><div class="metric-title">Lost Revenue</div><div class="metric-value">${lost_revenue:,.0f}</div></div>', unsafe_allow_html=True)
            
        st.write("---")
        
        c1, c2 = st.columns(2)
        with c1:
            fig_churn = px.pie(df, names='Churn', title='Customer Churn Distribution', hole=0.4, color='Churn', color_discrete_map={'Yes':'#EF553B', 'No':'#636EFA'})
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with c2:
            fig_contract = px.histogram(df, x='Contract', color='Churn', barmode='group', title='Churn by Contract Type')
            st.plotly_chart(fig_contract, use_container_width=True)
            
        c3, c4 = st.columns(2)
        with c3:
             fig_tenure = px.box(df, x='Churn', y='tenure', title='Tenure Distribution by Churn', color='Churn')
             st.plotly_chart(fig_tenure, use_container_width=True)
        with c4:
             fig_charges = px.histogram(df, x='MonthlyCharges', color='Churn', nbins=30, opacity=0.7, title='Monthly Charges Distribution')
             st.plotly_chart(fig_charges, use_container_width=True)
             
    else:
        st.warning("Data not found. Please run `python data_generator.py` first.")

with tab2:
    st.header("Predict Customer Churn")
    if model is not None:
        st.write("Enter the customer details below to predict their likelihood of churning.")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Demographics")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen", [0, 1])
                partner = st.selectbox("Partner", ["Yes", "No"])
                dependents = st.selectbox("Dependents", ["Yes", "No"])
            
            with col2:
                st.subheader("Account Info")
                tenure = st.slider("Tenure (Months)", 0, 72, 12)
                contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
            with col3:
                st.subheader("Services")
                internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                phone = st.selectbox("Phone Service", ["Yes", "No"])
                multiple = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                
                monthly_charges = st.number_input("Monthly Charges ($)", 10.0, 150.0, 50.0)
                total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, float(monthly_charges*tenure))

            submit = st.form_submit_button("Predict Churn Probability")
            
            if submit:
                # Prepare data
                input_data = pd.DataFrame({
                    'gender': [gender],
                    'SeniorCitizen': [senior_citizen],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone],
                    'MultipleLines': [multiple],
                    'InternetService': [internet],
                    'OnlineSecurity': ['No'], # default for simplicity
                    'OnlineBackup': ['No'],   # default for simplicity
                    'DeviceProtection': ['No'], # default for simplicity
                    'TechSupport': [tech_support],
                    'StreamingTV': ['No'],     # default for simplicity
                    'StreamingMovies': ['No'], # default for simplicity
                    'Contract': [contract],
                    'PaperlessBilling': [paperless],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges]
                })
                
                prob = model.predict_proba(input_data)[0][1]
                pred = "Churn" if prob > 0.5 else "Retain"
                
                st.write("---")
                st.subheader("Prediction Result")
                
                if pred == "Churn":
                    st.error(f"⚠️ High Risk of Churn! (Probability: {prob:.1%})")
                    st.write("Recommendation: Offer a discount on a yearly contract or free tech support.")
                else:
                    st.success(f"✅ Customer is likely to stay. (Churn Probability: {prob:.1%})")
                    
    else:
        st.warning("Model not found. Please run `python train_model.py` to train the model first.")
