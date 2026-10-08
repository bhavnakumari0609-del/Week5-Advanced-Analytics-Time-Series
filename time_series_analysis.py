import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# WEEK 5: ADVANCED ANALYTICS & TIME SERIES
# ==========================================

# 1. Load Dataset
df = pd.read_csv("sales_data.csv")

print("\n===== DATASET INFORMATION =====")
print(df.head())

print("\nNumber of records:", len(df))
print("Columns:", list(df.columns))


# 2. Data Cleaning
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())


# 3. Descriptive Statistics
print("\n===== DESCRIPTIVE STATISTICS =====")
print(df["Sales"].describe())


# 4. Add Time Features
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Weekday"] = df["Date"].dt.day_name()


# 5. Weekday Analysis
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

weekday_avg = (
    df.groupby("Weekday")["Sales"]
    .mean()
    .reindex(weekday_order)
)

print("\n===== WEEKDAY AVERAGE SALES =====")
print(weekday_avg)


# 6. Trend Analysis - 7 Day Rolling Average
df["Rolling_7_Day"] = df["Sales"].rolling(window=7).mean()

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["Sales"], label="Daily Sales")
plt.plot(df["Date"], df["Rolling_7_Day"], label="7-Day Rolling Average")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.title("Daily Sales Trend and 7-Day Rolling Average")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("daily_sales_trend.png")
plt.show()


# 7. Weekday Average Chart
plt.figure(figsize=(9, 5))
weekday_avg.plot(kind="bar")
plt.xlabel("Weekday")
plt.ylabel("Average Sales")
plt.title("Average Sales by Weekday")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("weekday_average_sales.png")
plt.show()


# 8. Anomaly Detection using IQR
Q1 = df["Sales"].quantile(0.25)
Q3 = df["Sales"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

anomalies = df[
    (df["Sales"] < lower_bound) |
    (df["Sales"] > upper_bound)
]

print("\n===== ANOMALY DETECTION =====")
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Number of anomalies:", len(anomalies))

if len(anomalies) > 0:
    print(anomalies[["Date", "Sales"]])
else:
    print("No anomalies detected.")


# 9. Chronological Train-Test Split
train_size = int(len(df) * 0.80)

train = df["Sales"].iloc[:train_size]
test = df["Sales"].iloc[train_size:]

print("\n===== TRAIN TEST SPLIT =====")
print("Training records:", len(train))
print("Testing records:", len(test))


# 10. Naive Forecast
naive_forecast = np.repeat(train.iloc[-1], len(test))


# 11. Seasonal Naive Forecast
# Weekly seasonality = 7 days
seasonal_forecast = train.iloc[-7:].values

seasonal_forecast = np.tile(
    seasonal_forecast,
    int(np.ceil(len(test) / 7))
)[:len(test)]


# 12. Evaluation Metrics
def calculate_metrics(actual, predicted):

    mae = np.mean(np.abs(actual - predicted))

    rmse = np.sqrt(
        np.mean((actual - predicted) ** 2)
    )

    mape = np.mean(
        np.abs((actual - predicted) / actual)
    ) * 100

    return mae, rmse, mape


naive_mae, naive_rmse, naive_mape = calculate_metrics(
    test.values,
    naive_forecast
)

seasonal_mae, seasonal_rmse, seasonal_mape = calculate_metrics(
    test.values,
    seasonal_forecast
)


# 13. Display Model Performance
print("\n===== MODEL PERFORMANCE =====")

print("\nNaive Forecast")
print("MAE :", naive_mae)
print("RMSE:", naive_rmse)
print("MAPE:", naive_mape)

print("\nSeasonal Naive Forecast")
print("MAE :", seasonal_mae)
print("RMSE:", seasonal_rmse)
print("MAPE:", seasonal_mape)


# 14. Save Model Performance
performance = pd.DataFrame({
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

performance.to_csv(
    "model_performance.csv",
    index=False
)

print("\nmodel_performance.csv created successfully.")


# 15. Model Comparison Chart
plt.figure(figsize=(9, 5))

plt.bar(
    performance["Model"],
    performance["RMSE"]
)

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Forecasting Model RMSE Comparison")

plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("model_performance.png")
plt.show()


# 16. Final Conclusion
best_model = performance.loc[
    performance["RMSE"].idxmin(),
    "Model"
]

print("\n===== FINAL RESULT =====")
print("Best performing model:", best_model)
print("Analysis completed successfully.")