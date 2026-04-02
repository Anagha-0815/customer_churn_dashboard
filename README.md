# Customer Churn Prediction & Analytics Dashboard

This is an end-to-end Machine Learning web application designed to predict whether a customer will churn (cancel their subscription or service). It is built using Python, Scikit-Learn, and Streamlit.

## Features
- **Synthetic Data Generation**: `data_generator.py` creates a realistic telco customer dataset.
- **Machine Learning**: `train_model.py` trains a Random Forest Classifier with feature engineering and SMOTE (if necessary) to predict churn.
- **Interactive Dashboard**: `app.py` provides an analytics overview (using Plotly) and an interactive predictor to score individual customers.

## Quick Start

### 1. Setup Virtual Environment & Install Dependencies
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate the Data
Create the synthetic dataset. It will be saved as `data/customer_churn_data.csv`.
```bash
python data_generator.py
```

### 3. Train the Model
Train the Random Forest model on the generated data. The model and preprocessing pipeline will be saved in the `models/` directory.
```bash
python train_model.py
```

### 4. Run the Streamlit Dashboard
Launch the web interface.
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser to view the dashboard.

## Directory Structure
- `data/`: Contains the generated CSV files.
- `models/`: Stores the serialized ML models `.joblib` files.
- `data_generator.py`: Script to formulate dummy data.
- `train_model.py`: Training script for data processing and model creation.
- `app.py`: Streamlit front-end application.
