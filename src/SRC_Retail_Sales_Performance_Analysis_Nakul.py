
import pandas as pd
import numpy as np
from pathlib import Path

# Function to calculate average revenue
def calculate_average_revenue(data):
    return data["Revenue"].mean()

# Load dataset
BASE_DIR = Path(__file__).resolve().parent.parent
df = pd.read_csv(BASE_DIR / "data" / "01_retail_sales.csv")

# Display first 5 rows
print(df.head())


# Dataset information
print(df.shape)

print(df.columns)

print(df.dtypes)


# Check missing values
print("Missing Values:")
print(df.isnull().sum())

# Check duplicate orders
print("Duplicate Orders:")
print(df.duplicated(subset="Order_ID").sum())


# Remove duplicate orders
df = df.drop_duplicates(subset="Order_ID")

print("Duplicate orders after cleaning:")
print(df.duplicated(subset="Order_ID").sum())

print("New dataset shape:")
print(df.shape)


# Fill missing City values
df["City"] = df["City"].fillna("Unknown")

# Check missing values again
print("Missing City values after cleaning:")
print(df["City"].isnull().sum())

print(df["City"].value_counts())


# Convert Order_Date into datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Extract month and weekday
df["Month"] = df["Order_Date"].dt.month
df["Weekday"] = df["Order_Date"].dt.day_name()

# Display result
print(df[["Order_Date", "Month", "Weekday"]].head())


# Calculate revenue
df["Calculated_Revenue"] = df["Quantity"] * df["Unit_Price"]

# Compare both revenue columns
print(df[["Quantity", "Unit_Price", "Revenue", "Calculated_Revenue"]].head())

# Check mismatched records
print("Mismatched Revenue:")
print((df["Revenue"] != df["Calculated_Revenue"]).sum())


# Total Revenue
total_revenue = df["Revenue"].sum()

# Total Quantity
total_quantity = df["Quantity"].sum()

average_order_value = calculate_average_revenue(df)

print("Total Revenue:", total_revenue)
print("Total Quantity:", total_quantity)
print("Average Order Value:", average_order_value)

# Top 10 Products by Revenue
top_products = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False).head(10)

print("Top 10 Products by Revenue:")
print(top_products)

# City-wise Revenue
city_revenue = df.groupby("City")["Revenue"].sum().sort_values(ascending=False)

print("City-wise Revenue:")
print(city_revenue)

# Monthly Revenue
monthly_revenue = df.groupby("Month")["Revenue"].sum().sort_index()

print("Monthly Revenue:")
print(monthly_revenue)

print("Best Month:", monthly_revenue.idxmax())
print("Highest Monthly Revenue:", monthly_revenue.max())

# Top 10 Customers by Spending
top_customers = df.groupby("Customer_ID")["Revenue"].sum().sort_values(ascending=False).head(10)

print("Top 10 Customers:")
print(top_customers)

# Payment Method Analysis
payment_analysis = df.groupby("Payment_Mode").agg(
    Total_Orders=("Order_ID", "count"),
    Total_Revenue=("Revenue", "sum")
)

print("Payment Method Analysis:")
print(payment_analysis)

# Detect Unusually Large Orders using IQR
Q1 = df["Revenue"].quantile(0.25)
Q3 = df["Revenue"].quantile(0.75)

IQR = Q3 - Q1

upper_limit = Q3 + 1.5 * IQR

large_orders = df[df["Revenue"] > upper_limit]

print("Unusually Large Orders:")
print(large_orders[["Order_ID", "Product", "Revenue"]])

print("Total Large Orders:", len(large_orders))

# Category-wise Summary
category_summary = df.groupby("Category").agg(
    Total_Orders=("Order_ID", "count"),
    Total_Quantity=("Quantity", "sum"),
    Total_Revenue=("Revenue", "sum")
)

print("Category-wise Summary:")
print(category_summary)

# NumPy Statistics
order_revenue = df["Revenue"].to_numpy()

print("Mean Revenue:", np.mean(order_revenue))
print("Median Revenue:", np.median(order_revenue))
print("Standard Deviation:", np.std(order_revenue))

# Save Cleaned Dataset
df.to_csv(BASE_DIR / "data" / "cleaned_retail_sales.csv", index=False)

print("Cleaned dataset saved successfully!")

# Business Recommendations

print("\nBUSINESS RECOMMENDATIONS")

print("1. Focus on Keyboard and Headphones as they generate the highest product revenue.")

print("2. Focus on Delhi and Bengaluru because they have the highest city-wise revenue.")

print("3. Increase marketing campaigns in February because it has the highest monthly revenue.")

print("4. Promote UPI payments because UPI generates the highest payment revenue.")

print("5. Focus on Accessories because this category generates higher total revenue.")

# NumPy Filtering and Aggregation

average_revenue = np.mean(order_revenue)

# Filter orders having revenue above average
high_revenue_orders = order_revenue[order_revenue > average_revenue]

print("\nOrders Above Average Revenue:")
print(high_revenue_orders)

print("Number of High Revenue Orders:", len(high_revenue_orders))
print("Total Revenue of High Revenue Orders:", np.sum(high_revenue_orders))

# Condition to identify high revenue orders
if len(high_revenue_orders) > 0:
    print("High revenue orders found:", len(high_revenue_orders))
else:
    print("No high revenue orders found")

# Loop to display business recommendations

recommendations = [
    "Focus on top revenue-generating products",
    "Focus on high-revenue cities",
    "Increase marketing in the best-performing month",
    "Promote the most popular payment method",
    "Focus on the highest-revenue category"
]

for i, recommendation in enumerate(recommendations, start=1):
    print(i, recommendation)

    # ==========================================
# END OF PROJECT 1
# ==========================================
# Name: Nakul
# Admission No.: 2026B15410287
# Branch: CSE DS 2
# GitHub Repository:
# https://github.com/aroranakul632-gif/Retail_Sales_Performance_Analysis.git
# ==========================================