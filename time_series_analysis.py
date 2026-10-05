import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ==============================
# 1. LOAD DATA
# ==============================

df = pd.read_csv("sales_data.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

print("\n===== DATASET INFORMATION =====")
print(df.head())
print("\nTotal records:", len(df))

# ==============================
# 2. CHECK MISSING VALUES
# ==============================

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

# ==============================
# 3. BASIC STATISTICS
# ==============================

print("\n===== SALES STATISTICS =====")
print(df["Sales"].describe())

# ==============================
# 4. TIME SERIES TREND
# ==============================

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Sales"])
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("sales_trend.png")
plt.show()

# ==============================
# 5. ROLLING AVERAGE
# ==============================

df["Rolling_Average"] = df["Sales"].rolling(window=7).mean()

plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["Sales"], label="Actual Sales")
plt.plot(
    df["Date"],
    df["Rolling_Average"],
    label="7-Day Rolling Average"
)

plt.title("Sales Trend with 7-Day Rolling Average")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("rolling_average.png")
plt.show()

# ==============================
# 6. DAY OF WEEK ANALYSIS
# ==============================

df["Day"] = df["Date"].dt.day_name()

day_sales = df.groupby("Day")["Sales"].mean()

print("\n===== AVERAGE SALES BY DAY =====")
print(day_sales)

plt.figure(figsize=(10, 5))
day_sales.plot(kind="bar")

plt.title("Average Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("seasonality.png")
plt.show()

# ==============================
# 7. ANOMALY DETECTION
# ==============================

mean_sales = df["Sales"].mean()
std_sales = df["Sales"].std()

upper_limit = mean_sales + 2 * std_sales
lower_limit = mean_sales - 2 * std_sales

df["Anomaly"] = (
    (df["Sales"] > upper_limit) |
    (df["Sales"] < lower_limit)
)

anomalies = df[df["Anomaly"]]

print("\n===== ANOMALIES =====")
print(anomalies[["Date", "Sales"]])

# ==============================
# 8. TRAIN-TEST SPLIT
# ==============================

train_size = int(len(df) * 0.8)

train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

print("\nTraining records:", len(train))
print("Testing records:", len(test))

# ==============================
# 9. NAIVE FORECAST
# ==============================

naive_predictions = np.repeat(
    train["Sales"].iloc[-1],
    len(test)
)

# ==============================
# 10. SEASONAL NAIVE FORECAST
# ==============================

seasonal_predictions = []

for i in range(len(test)):
    index = train_size + i - 7

    if index >= 0:
        seasonal_predictions.append(df["Sales"].iloc[index])
    else:
        seasonal_predictions.append(train["Sales"].iloc[-1])

# ==============================
# 11. MODEL EVALUATION
# ==============================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mape = np.mean(
        np.abs(
            (actual - predicted) / actual
        )
    ) * 100

    return mae, rmse, mape


naive_mae, naive_rmse, naive_mape = calculate_metrics(
    test["Sales"],
    naive_predictions
)

seasonal_mae, seasonal_rmse, seasonal_mape = calculate_metrics(
    test["Sales"],
    seasonal_predictions
)

# ==============================
# 12. PRINT RESULTS
# ==============================

print("\n===== MODEL PERFORMANCE =====")

print("\nNaive Forecast")
print("MAE :", round(naive_mae, 2))
print("RMSE:", round(naive_rmse, 2))
print("MAPE:", round(naive_mape, 2), "%")

print("\nSeasonal Naive Forecast")
print("MAE :", round(seasonal_mae, 2))
print("RMSE:", round(seasonal_rmse, 2))
print("MAPE:", round(seasonal_mape, 2), "%")

# ==============================
# 13. ACTUAL VS FORECAST
# ==============================

plt.figure(figsize=(12, 5))

plt.plot(
    train["Date"],
    train["Sales"],
    label="Training Data"
)

plt.plot(
    test["Date"],
    test["Sales"],
    label="Actual Test Data"
)

plt.plot(
    test["Date"],
    seasonal_predictions,
    label="Seasonal Forecast"
)

plt.title("Actual vs Forecasted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("actual_vs_forecast.png")
plt.show()

# ==============================
# 14. SAVE RESULTS
# ==============================

results = pd.DataFrame({
    "Model": [
        "Naive Forecast",
        "Seasonal Naive Forecast"
    ],
    "MAE": [
        naive_mae,
        seasonal_mae
    ],
    "RMSE": [
        naive_rmse,
        seasonal_rmse
    ],
    "MAPE": [
        naive_mape,
        seasonal_mape
    ]
})

results.to_csv(
    "model_performance.csv",
    index=False
)

print("\n===== PROJECT COMPLETED =====")
print("Charts saved successfully.")
print("Model performance saved as model_performance.csv")