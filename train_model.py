import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

def train():
    data_path = 'data/customer_churn_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run data_generator.py first.")
        return

    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Handle missing values if any (TotalCharges sometimes empty string in real datasets, but our generator outputs floats)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Separate features and target
    X = df.drop(['customerID', 'Churn'], axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Identify numerical and categorical columns
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    print("Building preprocessing pipeline...")
    # Preprocessing for numerical data
    numeric_transformer = StandardScaler()
    
    # Preprocessing for categorical data
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define the model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, class_weight='balanced')
    
    # Create the pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
                          
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the pipeline
    os.makedirs('models', exist_ok=True)
    model_path = 'models/churn_model_pipeline.joblib'
    joblib.dump(clf, model_path)
    print(f"\nModel pipeline saved to {model_path}")

if __name__ == "__main__":
    train()
