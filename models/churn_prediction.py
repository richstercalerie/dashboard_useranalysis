import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

def load_data():
    """Loads customer dataset and encodes categorical variables."""
    df = pd.read_csv("datasets/insurance_customers.csv")

    # Convert categorical values into numerical
    categorical_cols = ["Gender", "PolicyType", "Location"]
    label_encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le  # Store encoders

    return df, label_encoders

def train_churn_model(df):
    """Trains a churn prediction model with scaling and class balancing."""
    
    if "Churn" not in df.columns:
        raise ValueError("The dataset must contain a 'Churn' column!")

    X = df.drop(columns=["CustomerID", "Churn"], errors="ignore")
    y = df["Churn"]

    print(f"Training with {X.shape[1]} features.")  # Debugging feature count

    # 🔹 Apply StandardScaler to normalize numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔹 Adjust class weights dynamically if imbalance exists
    class_weights = {0: 1, 1: max(1, y.value_counts()[0] / y.value_counts()[1])}  

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 🔹 Use a tuned RandomForest to reduce overfitting
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,  # ✅ Prevents overfitting by limiting tree depth
        min_samples_split=5,  # ✅ Ensures better generalization
        class_weight=class_weights,
        random_state=42
    )
    
    # ✅ Fix: Ensure the model is trained with feature names
    model.fit(pd.DataFrame(X_train, columns=X.columns), y_train)

    return model, X.columns.tolist(), scaler

def predict_churn(model, customer_data, feature_names, scaler):
    """Predicts the churn probability for a given customer."""

    # ✅ Ensure input is a DataFrame with correct feature names
    customer_df = pd.DataFrame([customer_data], columns=feature_names)

    # ✅ Apply the same scaling used during training
    customer_df_scaled = scaler.transform(customer_df)

    # ✅ Ensure prediction uses DataFrame format
    churn_probability = model.predict_proba(pd.DataFrame(customer_df_scaled, columns=feature_names))[0][1]

    return round(churn_probability * 100, 2)  # Convert to percentage


# Run training
if __name__ == "__main__":
    df, label_encoders = load_data()
    model, feature_names, scaler = train_churn_model(df)

    # Test with a sample customer
    sample_customer = [30, 70000, 2, 4, 20000, 1, 15, 1, 0]
    churn_percentage = predict_churn(model, sample_customer, feature_names, scaler)
    loyalty_score = 100 - churn_percentage  # ✅ Fixed loyalty score calculation

    print(f"🔍 Loyalty Score for new customer: {loyalty_score:.2f}%")
