import pandas as pd
import numpy as np
import os

def generate_data(num_records=5000):
    np.random.seed(42)
    
    # Generate independent features
    customer_ids = [f"CUST-{str(i).zfill(5)}" for i in range(1, num_records + 1)]
    genders = np.random.choice(['Male', 'Female'], num_records)
    senior_citizen = np.random.choice([0, 1], num_records, p=[0.85, 0.15])
    partner = np.random.choice(['Yes', 'No'], num_records)
    dependents = np.random.choice(['Yes', 'No'], num_records, p=[0.3, 0.7])
    
    tenure = np.random.randint(1, 73, num_records)
    phone_service = np.random.choice(['Yes', 'No'], num_records, p=[0.9, 0.1])
    multiple_lines = np.random.choice(['No phone service', 'No', 'Yes'], num_records, p=[0.1, 0.45, 0.45])
    
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], num_records, p=[0.35, 0.45, 0.2])
    online_security = np.random.choice(['No', 'Yes', 'No internet service'], num_records, p=[0.5, 0.3, 0.2])
    online_backup = np.random.choice(['No', 'Yes', 'No internet service'], num_records, p=[0.4, 0.4, 0.2])
    device_protection = np.random.choice(['No', 'Yes', 'No internet service'], num_records, p=[0.4, 0.4, 0.2])
    tech_support = np.random.choice(['No', 'Yes', 'No internet service'], num_records, p=[0.5, 0.3, 0.2])
    streaming_tv = np.random.choice(['No', 'Yes', 'No internet service'], num_records, p=[0.4, 0.4, 0.2])
    streaming_movies = np.random.choice(['No', 'Yes', 'No internet service'], num_records, p=[0.4, 0.4, 0.2])
    
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], num_records, p=[0.55, 0.2, 0.25])
    paperless_billing = np.random.choice(['Yes', 'No'], num_records, p=[0.6, 0.4])
    payment_method = np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], num_records, p=[0.35, 0.2, 0.2, 0.25])
    
    monthly_charges = np.random.uniform(18.0, 120.0, num_records)
    total_charges = monthly_charges * tenure + np.random.normal(0, 50, num_records)
    total_charges = np.where(total_charges < 0, monthly_charges, total_charges) # Fix negative values
    
    # Calculate a rough churn probability based on some features to make the dataset somewhat realistic
    churn_prob = np.zeros(num_records)
    
    # Base probability
    churn_prob += 0.2
    
    # Modifiers
    churn_prob += np.where(contract == 'Month-to-month', 0.2, -0.1)
    churn_prob += np.where(tenure < 12, 0.15, -0.05)
    churn_prob += np.where(internet_service == 'Fiber optic', 0.1, -0.05)
    churn_prob += np.where(tech_support == 'No', 0.1, -0.05)
    churn_prob += np.where(monthly_charges > 80, 0.05, -0.05)
    
    # Clip probabilities between 0.01 and 0.99
    churn_prob = np.clip(churn_prob, 0.01, 0.99)
    
    # Generate actual churn based on probabilities
    churn = [np.random.choice(['Yes', 'No'], p=[p, 1-p]) for p in churn_prob]
    
    data = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': np.round(monthly_charges, 2),
        'TotalCharges': np.round(total_charges, 2),
        'Churn': churn
    })
    
    os.makedirs('data', exist_ok=True)
    file_path = 'data/customer_churn_data.csv'
    data.to_csv(file_path, index=False)
    print(f"Data generated and saved to {file_path}")
    return data

if __name__ == "__main__":
    generate_data()
