import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    """Loads customer dataset."""
    df = pd.read_csv("datasets/insurance_customers.csv")
    df['PolicyType'] = df['PolicyType'].astype('category')
    return df

def recommend_policy(customer_id, num_recommendations=3):
    """Recommends diverse policies for a given customer."""
    df = load_data()

    # Convert categorical policy type into numerical values
    df['PolicyType_Code'] = df['PolicyType'].cat.codes

    # Create a customer-policy interaction matrix
    interaction_matrix = df.pivot_table(index='CustomerID', columns='PolicyType_Code', values='PremiumAmount', fill_value=0)

    # Compute similarity between customers
    similarity_matrix = cosine_similarity(interaction_matrix)
    customer_sim_df = pd.DataFrame(similarity_matrix, index=df['CustomerID'], columns=df['CustomerID'])

    # Get top 10 similar customers (excluding self)
    similar_customers = customer_sim_df[customer_id].sort_values(ascending=False).index[1:10]

    # Get policy recommendations from these similar customers
    recommended_policies = df[df['CustomerID'].isin(similar_customers)]['PolicyType'].tolist()

    # Ensure diversity: Get unique policy types
    unique_policies = list(set(recommended_policies))[:num_recommendations]

    return unique_policies if unique_policies else ["No recommendations found"]

# Example usage
if __name__ == "__main__":
    test_customer_id = 1003
    recommendations = recommend_policy(test_customer_id)
    print(f"✅ Recommended policies for customer {test_customer_id}: {recommendations}")
