import pandas as pd
import numpy as np
import os

# Paths
RAW_PATH = os.path.join("..", "data", "raw_ecommerce_sales.csv")
CLEANED_PATH = os.path.join("..", "data", "cleaned_ecommerce_sales.csv")

# --------------------------------------------------------
# 1. Load Dataset
# --------------------------------------------------------
def load_data():
    print("📥 Loading dataset...")
    df = pd.read_csv(RAW_PATH)
    print(f"✔ Loaded dataset with shape: {df.shape}")
    return df


# --------------------------------------------------------
# 2. Handle Missing Values
# --------------------------------------------------------
def handle_missing_values(df):
    print("🧽 Handling missing values...")

    # Fill numeric missing values with median
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # Fill categorical missing values with mode
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    return df


# --------------------------------------------------------
# 3. Fix Data Types
# --------------------------------------------------------
def convert_data_types(df):
    print("🔧 Fixing data types...")

    # Convert date column
    if "OrderDate" in df.columns:
        df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors='coerce')

    # Convert numeric columns
    numeric_fields = ["UnitPrice", "Quantity", "TotalAmount"]
    for col in numeric_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# --------------------------------------------------------
# 4. Remove Duplicates
# --------------------------------------------------------
def remove_duplicates(df):
    print("🗑 Removing duplicates...")
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    print(f"✔ Removed {before - after} duplicate rows")
    return df


# --------------------------------------------------------
# 5. Outlier Removal
# --------------------------------------------------------
def remove_outliers(df):
    print("📉 Removing outliers using Z-score...")

    def z_score_filter(data, column):
        if column not in data.columns:
            return data
        threshold = 3
        mean = data[column].mean()
        std = data[column].std()
        if std == 0:
            return data
        return data[(data[column] - mean).abs() <= threshold * std]

    for col in ["UnitPrice", "Quantity", "TotalAmount"]:
        df = z_score_filter(df, col)

    return df


# --------------------------------------------------------
# 6. Clean Text Fields
# --------------------------------------------------------
def clean_text_fields(df):
    print("🔤 Cleaning text columns...")

    def clean(col):
        return col.str.lower().str.strip()

    text_cols = ["CustomerName", "ProductName", "Category"]
    for col in text_cols:
        if col in df.columns:
            df[col] = clean(df[col])

    return df


# --------------------------------------------------------
# 7. Feature Engineering
# --------------------------------------------------------
def feature_engineering(df):
    print("🧮 Creating new features...")

    # New computed total if missing
    if "Quantity" in df.columns and "UnitPrice" in df.columns:
        df["ComputedTotal"] = df["Quantity"] * df["UnitPrice"]

    # Extract year and month
    if "OrderDate" in df.columns:
        df["Year"] = df["OrderDate"].dt.year
        df["Month"] = df["OrderDate"].dt.month

    return df


# --------------------------------------------------------
# 8. Save Cleaned Dataset
# --------------------------------------------------------
def save_data(df):
    print(f"💾 Saving cleaned dataset to: {CLEANED_PATH}")
    df.to_csv(CLEANED_PATH, index=False)
    print("✔ Cleaned dataset saved successfully!")


# --------------------------------------------------------
# 9. Main Pipeline
# --------------------------------------------------------
def main():
    df = load_data()
    df = handle_missing_values(df)
    df = convert_data_types(df)
    df = remove_duplicates(df)
    df = remove_outliers(df)
    df = clean_text_fields(df)
    df = feature_engineering(df)
    save_data(df)


if __name__ == "__main__":
    main()

