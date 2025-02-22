import pandas as pd

def load_dataset():
    """Loads and displays the insurance customer dataset."""
    df = pd.read_csv("datasets/insurance_customers.csv")
    print("✅ Dataset Loaded Successfully!")
    print(df.head(10))  # Show first 10 rows
    print("\n📊 Summary Statistics:")
    print(df.describe())

if __name__ == "__main__":
    load_dataset()
