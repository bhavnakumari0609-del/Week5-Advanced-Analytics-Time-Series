# Advanced Analytics and Time-Series Analysis

## Week 5 Internship Project

This project focuses on advanced analytics and time-series analysis using a daily sales dataset. The analysis identifies temporal trends, weekday patterns, possible anomalies, and evaluates simple forecasting models.

The project uses Python-based data analysis and visualization techniques to understand historical sales behavior and compare forecasting performance.

---

## 1. Project Objective

The main objectives of this project are:

- Analyze daily sales data over time.
- Identify overall sales trends.
- Analyze sales patterns by weekday.
- Detect unusual observations using statistical techniques.
- Divide the dataset into chronological training and testing sets.
- Build baseline forecasting models.
- Evaluate forecasting performance using MAE, RMSE, and MAPE.
- Compare forecasting models and identify the better-performing model.
- Generate visualizations to support the analysis.
- Translate analytical results into practical business insights.

---

## 2. Dataset Description

The project uses a daily sales dataset containing 90 observations.

### Dataset Columns

| Column | Description |
|--------|-------------|
| Date | Date of the sales observation |
| Sales | Daily sales value |

### Dataset Size

- Total records: 90
- Total columns: 2
- Time period: January 2026 to March 2026
- Missing Date values: 0
- Missing Sales values: 0

### Sample Data

| Date | Sales |
|------|------:|
| 2026-01-01 | 120 |
| 2026-01-02 | 125 |
| 2026-01-03 | 130 |
| 2026-01-04 | 128 |
| 2026-01-05 | 135 |

---

## 3. Tools and Technologies

The project was developed using Python.

### Python Libraries

- **Pandas** – Used for loading, cleaning, transforming, and analyzing the dataset.
- **NumPy** – Used for numerical calculations and forecasting operations.
- **Matplotlib** – Used to create charts and visualizations.

### Development Environment

- Visual Studio Code
- Python
- CSV dataset

Advanced time-series libraries such as Statsmodels and Prophet are relevant for more advanced forecasting approaches and can be applied in future extensions of this project.

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the CSV dataset using Pandas.
2. Converted the `Date` column into a proper datetime format.
3. Sorted the observations chronologically.
4. Checked for missing values.
5. Created time-related features such as weekday, day, and month.
6. Verified that the dataset contained 90 observations.

The missing-value analysis showed that both `Date` and `Sales` contained zero missing values.

---

## 5. Descriptive Statistical Analysis

The sales data was summarized using descriptive statistics.

| Statistic | Value |
|-----------|------:|
| Count | 90 |
| Mean | 279.06 |
| Standard Deviation | 96.28 |
| Minimum | 120 |
| 25% | 195.75 |
| Median | 277.50 |
| 75% | 363.75 |
| Maximum | 445 |

The average daily sales value was approximately 279.06. Sales ranged from 120 to 445, showing considerable variation across the 90-day period.

For example, a sales value of 120 represents a relatively low-sales observation compared with the overall average, while 445 represents the highest recorded daily sales value.

---

## 6. Trend Analysis

A time-series trend analysis was performed using daily sales values and a 7-day rolling average.

The rolling average helps reduce short-term fluctuations and makes the underlying movement of the sales series easier to observe.

### Generated Visualization

`daily_sales_trend.png`

The visualization compares the daily sales values with the 7-day rolling average.

A rolling average is useful in business analysis because individual daily sales can fluctuate, while the rolling average provides a smoother view of the recent sales direction.

---

## 7. Weekday Analysis

Average sales were calculated for each day of the week.

| Day | Average Sales |
|-----|--------------:|
| Monday | 284.69 |
| Tuesday | 289.31 |
| Wednesday | 278.58 |
| Thursday | 268.92 |
| Friday | 274.15 |
| Saturday | 276.69 |
| Sunday | 281.00 |

Tuesday had the highest average sales value at approximately 289.31, while Thursday had the lowest average at approximately 268.92.

This indicates that there is some variation in sales across weekdays. However, because the dataset contains only 90 days, these differences should be treated as an observed pattern rather than a permanent business seasonality.

### Generated Visualization

`weekday_average_sales.png`

---

## 8. Anomaly Detection

Anomaly detection was performed using the Interquartile Range (IQR) method.

The calculated bounds were:

- Lower Bound: -56.25
- Upper Bound: 615.75

No sales observations fell outside these limits.

### Result

**Number of anomalies detected: 0**

This means that, according to the selected IQR rule, there were no statistically unusual sales observations in the dataset.

For example, although 120 is much lower than the maximum value of 445, it is still within the calculated acceptable range and therefore is not classified as an anomaly.

---

## 9. Train-Test Split

For forecasting evaluation, the data was divided chronologically rather than randomly.

- Training records: 72
- Testing records: 18

The first 72 observations were used for model development, while the final 18 observations were kept for testing.

A chronological split is appropriate for time-series data because future observations should not be used to train a model before evaluating its forecasting ability.

---

## 10. Forecasting Models

Two baseline forecasting models were evaluated.

### 10.1 Naive Forecast

The Naive Forecast assumes that the next value will be equal to the most recently observed training value.

For example, if the last training-day sales value is 300, the model uses 300 as the forecast for the next period.

This method is simple but provides an important baseline for evaluating more advanced forecasting methods.

---

### 10.2 Seasonal Naive Forecast

The Seasonal Naive Forecast uses values from the previous seasonal cycle to predict future observations.

For this project, weekly behavior was considered when creating the seasonal forecast.

The purpose of this model is to determine whether repeating patterns can provide better predictions than simply using the last observed value.

---

## 11. Model Evaluation

The models were evaluated using three commonly used forecasting metrics:

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted sales values.

A lower MAE indicates better forecasting performance.

### Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error.

RMSE gives greater weight to larger forecasting errors.

### Mean Absolute Percentage Error (MAPE)

MAPE expresses forecasting error as a percentage of actual values.

A lower MAPE indicates that the forecast is closer to the actual observations.

---

## 12. Model Performance Results

The final model evaluation produced the following results:

| Model | MAE | RMSE | MAPE |
|-------|----:|-----:|-----:|
| Naive Forecast | 32.28 | 37.46 | 7.63% |
| Seasonal Naive Forecast | 25.28 | 25.36 | 6.14% |

The Seasonal Naive Forecast performed better across all three evaluation metrics.

Compared with the Naive Forecast, the Seasonal Naive Forecast produced:

- Lower MAE
- Lower RMSE
- Lower MAPE

Therefore, the Seasonal Naive Forecast was selected as the better-performing baseline model for this dataset.

### Generated Visualization

`model_performance.png`

The model performance chart provides a visual comparison of forecasting errors.

---

## 13. Business Interpretation

The analysis provides several useful business observations.

First, the sales dataset shows meaningful variation over the 90-day period. The average sales value was 279.06, while the maximum was 445.

Second, weekday analysis showed that Tuesday had the highest average sales. A business could use this observation as an initial signal when planning inventory, staffing, or promotional activities.

Third, no anomalies were detected using the IQR method. This suggests that the dataset does not contain extreme sales observations according to the selected statistical rule.

Finally, the Seasonal Naive Forecast performed better than the simple Naive Forecast. This suggests that considering repeated temporal behavior can improve forecasting accuracy compared with using only the most recent observation.

### Example

If a business wants to plan inventory for upcoming days, a forecasting approach that considers recurring weekly behavior may provide more useful estimates than simply repeating the previous day's sales.

However, these findings should be validated with a longer historical dataset before making major operational decisions.

---

## 14. Project Files

The project contains the following files:

| File | Purpose |
|------|---------|
| `sales_data.csv` | Source sales dataset |
| `time_series_analysis.py` | Main Python analysis and forecasting script |
| `model_performance.csv` | Forecasting model evaluation results |
| `daily_sales_trend.png` | Daily sales and rolling-average visualization |
| `weekday_average_sales.png` | Average sales by weekday |
| `model_performance.png` | Forecasting model performance comparison |
| `README.md` | Project documentation |

---

## 15. Key Findings

The major findings from the analysis are:

1. The dataset contains 90 daily sales observations.
2. There are no missing values.
3. Mean daily sales are approximately 279.06.
4. Sales range from 120 to 445.
5. Tuesday has the highest average sales at approximately 289.31.
6. Thursday has the lowest average sales at approximately 268.92.
7. No anomalies were detected using the IQR method.
8. The dataset was divided into 72 training and 18 testing observations.
9. Seasonal Naive Forecast achieved lower MAE, RMSE, and MAPE than the Naive Forecast.
10. Seasonal Naive Forecast is the better-performing baseline model for this dataset.

---

## 16. Limitations

This project has some limitations:

- The dataset contains only 90 observations.
- Only two baseline forecasting models were evaluated.
- External factors such as promotions, holidays, pricing, weather, and marketing activities were not included.
- The weekday pattern may not remain stable over a longer period.
- Only one chronological test period was used for evaluation.

These limitations mean that the results should be interpreted as an analytical project outcome rather than a production-level forecasting system.

---

## 17. Future Improvements

The project can be extended by:

- Using a larger historical dataset.
- Adding holiday and promotional information.
- Including pricing and marketing variables.
- Applying advanced statistical models using Statsmodels.
- Experimenting with Prophet for time-series forecasting.
- Testing machine-learning forecasting approaches.
- Performing cross-validation designed specifically for time-series data.
- Comparing additional forecasting models.
- Building an interactive dashboard for business users.

---

## 18. Conclusion

This project demonstrates the practical use of Python for advanced analytics and time-series analysis. The sales dataset was cleaned, explored, analyzed for temporal patterns, checked for anomalies, and used to evaluate forecasting models.

The analysis showed that the Seasonal Naive Forecast achieved better performance than the Naive Forecast across MAE, RMSE, and MAPE. This provides evidence that considering recurring temporal behavior can improve forecasting performance for the analyzed sales data.

Overall, the project demonstrates how historical sales data can be transformed into useful analytical insights and forecasting evidence that can support business planning and decision-making.