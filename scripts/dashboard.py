import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder

# ✅ Ensure Python finds the 'models' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ✅ Import models correctly
from models.recommendation_model import recommend_policy
from models.churn_prediction import train_churn_model, predict_churn, load_data

# Load dataset
df, label_encoders = load_data()

# Initialize database connection
conn = sqlite3.connect("users.db", check_same_thread=False)
cursor = conn.cursor()

# ✅ Ensure `score` column exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT,
        role TEXT,
        status TEXT DEFAULT 'Pending',
        score INTEGER DEFAULT 100  -- Default score for customers
    )
""")
conn.commit()

# Function to update user score
def update_score(user_id, points):
    cursor.execute("UPDATE users SET score = score + ? WHERE id = ?", (points, user_id))
    conn.commit()

# Sidebar Navigation
st.sidebar.header("🔍 Navigation")
menu = st.sidebar.radio("Choose an option:", ["Dashboard","Churn Insights","User Creation Form", "Manage Teams", "Leaderboard","Charts & Analytics", "FAQs", "Chatbot", "Contact Us"])

if menu == "Dashboard":
    # Sidebar - Customer Selection
    st.sidebar.header("🔍 Select Customer")
    customer_id = st.sidebar.selectbox("Choose Customer ID", df["CustomerID"].unique())

    # Filter dataset for the selected customer
    customer_df = df[df["CustomerID"] == customer_id].copy()
    customer_df["Churn"] = customer_df["Churn"].astype(str)

    # Main Dashboard Header
    st.title("📊 Insurance Customer Insights Dashboard")

    # Display Customer Details
    st.subheader(f"👤 Customer {customer_id} Details")
    st.dataframe(customer_df, use_container_width=True)

    # ✅ Award +10 points for viewing the dashboard
    user = cursor.execute("SELECT id FROM users WHERE email = ?", (st.session_state.get("user_email", ""),)).fetchone()
    if user:
        update_score(user[0], 10)

    # Display Policy Recommendations
    st.subheader("🔥 Recommended Policies")
    recommendations = recommend_policy(customer_id)
    if recommendations:
        selected_policy = st.radio("Choose a policy", recommendations, horizontal=True)
        if st.button("Select Policy"):
            st.success(f"📌 You selected {selected_policy} policy")
            update_score(user[0], 20)  # ✅ Award +20 points for selecting a policy
    else:
        st.warning("No policy recommendations available for this customer.")

    # Train Churn Model
    st.subheader("⚠️ Churn Prediction")
    churn_model, feature_names, scaler = train_churn_model(df)

    # Retrieve the encoded customer data for prediction
    customer_data = customer_df.drop(columns=["CustomerID", "Churn"], errors="ignore")
    customer_data = customer_data.reindex(columns=feature_names, fill_value=0).values[0]

    # Convert customer_data to DataFrame before passing to model
    customer_data_df = pd.DataFrame([customer_data], columns=feature_names)
    churn_probability = churn_model.predict_proba(customer_data_df)[0][1]
    loyalty_score = (1 - churn_probability) * 100

    # Show loyalty score as percentage with progress bar
    st.write(f"📊 **Loyalty Score:** {loyalty_score:.2f}%")
    st.progress(loyalty_score / 100)

    # ✅ Bonus Points for Loyalty Score
    if loyalty_score > 80:
        update_score(user[0], 30)  # ✅ Award +30 points for high loyalty
    elif loyalty_score > 50:
        update_score(user[0], 20)
elif menu == "Churn Insights":
    st.title("📉 Churn Prediction Insights")

    st.sidebar.header("Select Customer")
    customer_id = st.sidebar.selectbox("Choose Customer ID", df["CustomerID"].unique())

    # Select the customer data
    customer_df = df[df["CustomerID"] == customer_id].copy()
    churn_model, feature_names, scaler = train_churn_model(df)

    # Prepare customer data for prediction
    customer_data = customer_df.drop(columns=["CustomerID", "Churn"], errors="ignore")
    customer_data = customer_data.reindex(columns=feature_names, fill_value=0)

    # Convert to NumPy array
    customer_data_array = customer_data.to_numpy().reshape(1, -1)  # ✅ Ensure correct shape

    # Predict churn probability
    churn_probability = churn_model.predict_proba(customer_data_array)[0][1]
    loyalty_score = (1 - churn_probability) * 100

    # Display churn insights
    st.write(f"📊 **Loyalty Score:** {loyalty_score:.2f}%")
    st.progress(loyalty_score / 100)

    if churn_probability > 0.5:
        st.warning("⚠️ High Risk of Churn! Consider offering personalized discounts.")
    else:
        st.success("✅ Low Risk of Churn! Keep engaging the customer.")

    # ✅ SHAP Explanation Fix
    explainer = shap.TreeExplainer(churn_model)
    shap_values = explainer.shap_values(customer_data_array)

    st.subheader("🔍 Feature Importance Analysis")

    # ✅ Handling SHAP Output for Binary Classification
    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_values_fixed = shap_values[1]  # Select churn class SHAP values
    else:
        shap_values_fixed = shap_values  # Use directly if no list

    # ✅ Convert feature names to a NumPy array to prevent index issues
    feature_names_array = np.array(feature_names)

    # ✅ Fix Feature Names Length Mismatch
    if shap_values_fixed.shape[1] == len(feature_names):
    	
    	shap.summary_plot(shap_values_fixed, customer_data_array, feature_names=feature_names_array, show=False)
    
    	fig = plt.gcf()  # ✅ Get current figure to ensure rendering
    	st.pyplot(fig, clear_figure=True)  # ✅ Fix blank plot issue
    else:
    	st.error(f"🚨 SHAP mismatch: {shap_values_fixed.shape} vs {len(feature_names)} features!")


elif menu == "User Creation Form":
    st.title("👤 User Creation Form")
    name = st.text_input("Enter your name:")
    email = st.text_input("Enter your email:")
    role = st.selectbox("Select your role:", ["Customer", "Agent", "Manager"])

    if st.button("Submit"):
        initial_score = 100 if role == "Customer" else 200 if role == "Agent" else 300
        cursor.execute("INSERT INTO users (name, email, role, status, score) VALUES (?, ?, ?, ?, ?)", 
                       (name, email, role, "Pending", initial_score))
        conn.commit()
        st.success(f"User {name} created successfully! 🎉 You start with {initial_score} points.")

elif menu == "Manage Teams":
    st.title("💼 Manage Teams")
    st.write("### Registered Users & Scores:")

    # Fetch user data with scores
    users = pd.read_sql("SELECT id, name, email, role, COALESCE(status, 'Pending') AS status, COALESCE(score, 0) AS score FROM users", conn)

    if users.empty:
        st.info("No users found.")
    else:
        def get_badge(score):
            """Assigns a badge based on user score."""
            if score < 200:
                return "🟢 Beginner"
            elif 200 <= score < 500:
                return "🔵 Intermediate"
            elif 500 <= score < 800:
                return "🟣 Advanced"
            else:
                return "🏆 Elite"

        # Apply badge system
        users["Badge"] = users["score"].apply(get_badge)

        # Editable table with Score Badge
        edited_users = st.data_editor(
            users[["id", "name", "email", "role", "status", "Badge"]],
            column_config={
                "status": st.column_config.SelectboxColumn("Status", options=["Pending", "Approved", "Rejected"]),
            },
            disabled=["id", "Badge"],  # Prevent editing user ID and Badge
            hide_index=True,
        )

        # Save updates
        if st.button("Update Changes"):
            for index, row in edited_users.iterrows():
                cursor.execute("UPDATE users SET name = ?, email = ?, role = ?, status = ? WHERE id = ?",
                               (row["name"], row["email"], row["role"], row["status"], row["id"]))
                # ✅ Award +25 points for approving users
                if row["status"] == "Approved":
                    update_score(row["id"], 25)
            conn.commit()
            st.success("User details updated successfully!")
            st.rerun()

elif menu == "Leaderboard":
    st.title("🏆 Leaderboard - Top Users")
    top_users = pd.read_sql("SELECT name, role, score FROM users ORDER BY score DESC LIMIT 10", conn)

    if top_users.empty:
        st.info("No scores available yet.")
    else:
        st.write("### 🥇 Top Performers")
        st.dataframe(top_users, hide_index=True)
elif menu == "Charts & Analytics":
    st.title("📊 Charts & Analytics")
    
    # Churn Rate Distribution
    st.subheader("Customer Churn Rate")
    churn_counts = df["Churn"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(churn_counts, labels=["Not Churned", "Churned"], autopct='%1.1f%%', colors=['#6AB187', '#FF6F61'])
    st.pyplot(fig)
    
    # User Role Distribution
    st.subheader("User Role Distribution")
    users = pd.read_sql("SELECT role FROM users", conn)
    role_counts = users["role"].value_counts().reset_index()
    role_counts.columns = ["role", "count"]  # Rename for compatibility

    fig, ax = plt.subplots()
    sns.barplot(data=role_counts, x="role", y="count", hue="role", palette="viridis", legend=False, ax=ax)
    ax.set_xlabel("Role")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    
    # Score Distribution
    st.subheader("User Score Distribution")
    scores = pd.read_sql("SELECT score FROM users", conn)
    fig, ax = plt.subplots()
    sns.histplot(scores["score"], bins=10, kde=True, color='#4C72B0', ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)
    
    # Top Users by Score
    st.subheader("🏆 Top Users by Score")
    top_users = pd.read_sql("SELECT name, score FROM users ORDER BY score DESC LIMIT 5", conn)

    fig, ax = plt.subplots()
    sns.barplot(data=top_users, x="score", y="name", hue="name", palette="Blues_r", legend=False, ax=ax)
    ax.set_xlabel("Score")
    ax.set_ylabel("User")
    st.pyplot(fig)

elif menu == "FAQs":
    st.title("📃 Frequently Asked Questions")
    st.write("**Q1: How is churn probability calculated?**")
    st.write("A: Churn probability is predicted using a trained machine learning model.")
    st.write("**Q2: Can I update my policy recommendations?**")
    st.write("A: Yes, policy recommendations update based on customer interactions.")

elif menu == "Chatbot":
    st.title("🧐 AI Chatbot")
    st.markdown("[Click here to chat with our AI bot](https://your-chatbot-link.com)")

elif menu == "Contact Us":
    st.title("📞 Contact Us")
    st.write("**Email:** support@insurance.com")
    st.write("**Phone:** +91-6203684631")

# Close database connection
conn.close()

st.write("Developed by Anurag Mishra")
