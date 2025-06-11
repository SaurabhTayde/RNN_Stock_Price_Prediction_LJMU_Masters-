# Stock Price Prediction Using Recurrent Neural Networks

## Project Overview

This project aims to predict stock prices for four major technology companies: Amazon (AMZN), Google (GOOGL), IBM (IBM), and Microsoft (MSFT), using historical stock data and Recurrent Neural Networks (RNNs). The primary objective is to leverage the sequential nature of stock market data to forecast future price movements. The project explores both Simple RNN and advanced GRU (Gated Recurrent Unit) architectures for single-target (AMZN) and multi-target (all four companies) price prediction.

A key component of this project is a robust data preprocessing pipeline designed to handle the non-stationarity and distribution shifts commonly found in financial time series data.

### Author
Saurabh Tayde

### Email
saurabhtayde2810@gmail.com

## Business Objective

The goal is to develop RNN models that can predict future stock prices given a historical window of stock data. Accurate predictions can provide valuable insights for investment strategies and risk management. Using data from multiple companies within the same sector (Technology) aims to capture broader market sentiment and potentially improve model performance.

## Dataset

The dataset consists of four CSV files, one for each company (AMZN, GOOGL, IBM, MSFT). Each file contains historical daily stock data from January 1, 2006, to January 1, 2018.

The columns in each dataset are:
*   **Date**: The date of the record.
*   **Open**: The opening price of the stock on that day.
*   **High**: The highest price of the stock on that day.
*   **Low**: The lowest price of the stock on that day.
*   **Close**: The closing price of the stock on that day.
*   **Volume**: The number of shares traded on that day.
*   **Name**: The stock ticker symbol (this column is dropped after initial processing).

## Project Structure

The project is implemented in a Jupyter Notebook (`RNN_Stock_Price_Prediction_SaurabhTayde_.ipynb`) and follows these main stages:

1.  **Data Loading and Preparation:**
    *   Importing necessary libraries (Pandas, NumPy, Matplotlib, Seaborn, TensorFlow/Keras, Scikit-learn).
    *   **Data Aggregation:** Combining the four individual stock CSV files into a single master DataFrame, aligning by date.
    *   **Missing Value Handling:** Identifying and imputing missing values (using forward-fill).
    *   **Exploratory Data Analysis (EDA) & Visualization:**
        *   Analyzing frequency distribution and time-based variation of stock volumes.
        *   Analyzing correlations between different stock prices and features.
    *   **Data Processing for RNNs:**
        *   **Robust Price-Based Transformation:** Transforming raw prices into percentage changes from a baseline price and log-transforming volume data to stabilize the series and mitigate distribution shift.
        *   **Windowing:** Creating sequential windows of data (input features X) and corresponding target values (y).
        *   **Train-Test Split:** Splitting the windowed data into training and testing sets chronologically.
        *   **Scaling:** Applying StandardScaler to the windowed features and targets (fitted on training data only).

2.  **RNN Model Development (Single Target - AMZN):**
    *   **Simple RNN Model:**
        *   Defining a function to build a Simple RNN model with configurable layers.
        *   Performing hyperparameter tuning to find an optimal network configuration.
        *   Training the optimal Simple RNN model and evaluating its performance on training and test data.
    *   **Advanced RNN (GRU) Model:**
        *   Defining a function to build a GRU model with more sophisticated architecture and regularization.
        *   Performing hyperparameter tuning for the GRU model.
        *   Training the optimal GRU model and comparing its performance against the Simple RNN.

3.  **Predicting Multiple Target Variables (Optional):**
    *   Preparing data for predicting the closing prices of all four companies simultaneously.
    *   Training and evaluating both Simple RNN and Advanced GRU models for this multi-target task.
    *   Analyzing per-company performance and overall multi-target prediction capabilities.

4.  **Conclusion:**
    *   Summarizing key findings, model performance, insights gained, and limitations.
    *   Discussing potential areas for future improvement.

## Key Methodologies and Techniques

*   **Recurrent Neural Networks (RNNs):** SimpleRNN and GRU layers from Keras.
*   **Data Preprocessing:**
    *   Percentage change from baseline for price stabilization.
    *   Log transformation for volume data.
    *   Windowing for sequence generation.
    *   StandardScaler for feature normalization.
*   **Time Series Splitting:** Chronological train-test split to prevent data leakage.
*   **Hyperparameter Tuning:** Iterative testing of different network configurations (number of units, layers, dropout rates, learning rates, batch sizes, optimizers).
*   **Regularization Techniques:** Dropout, L2 regularization, recurrent dropout, gradient clipping, and Batch Normalization (during tuning for GRU) to combat overfitting.
*   **Evaluation Metrics:**
    *   Mean Squared Error (MSE)
    *   Mean Absolute Error (MAE)
    *   R-squared (R²)
    *   Price Correlation
    *   Directional Accuracy
*   **Visualization:** Matplotlib and Seaborn for plotting price trends, correlations, error distributions, and training history.

## Key Results and Insights

### Single-Target Prediction (AMZN)

*   **Training Performance:** Both Simple RNN and GRU models demonstrated a strong ability to fit the training data, achieving high R² (Simple RNN: 0.9903, GRU: 0.9760) and Price Correlation (Simple RNN: 0.9956, GRU: 0.9940).
*   **Test Performance:**
    *   **GRU Outperformed Simple RNN:** The GRU model showed better generalization.
        *   **Price Correlation (Test):** GRU (0.9560) vs Simple RNN (0.7953).
        *   **R² (Test):** GRU (-1.3254) vs Simple RNN (-2.0042). While still negative, the GRU showed improvement.
        *   **Price MAE (Test):** GRU ($239.93) vs Simple RNN ($261.24).
    *   **Overfitting:** A significant challenge was overfitting, where models performed exceptionally well on training data but poorly on test data (especially in terms of R²). The GRU model demonstrated a reduced overfitting gap compared to the Simple RNN.
    *   **Directional Accuracy (Test):** Around 51-52% for both models, slightly better than random.

### Multi-Target Prediction (All Four Stocks - Optional)

*   **Training Performance:** Both models also fit the multi-target training data well, with overall R² scores above 0.95.
*   **Test Performance:**
    *   The Advanced GRU model generally provided better average correlation (0.9200 vs 0.8252 for Simple RNN) and directional accuracy across the four stocks.
    *   R² scores remained negative on the test set for both models, indicating difficulty in predicting the magnitude of price changes accurately for multiple stocks simultaneously.
    *   MSFT and AMZN generally showed higher individual correlations in multi-target predictions.

### General Insights

*   **Robust Data Transformation:** The "robust price-based" transformation (percentage change from baseline) was crucial for stabilizing the time series data and improving model trainability.
*   **Challenge of Price Magnitude Prediction:** While models could capture price trends (evidenced by high correlation scores on test data with GRU), accurately predicting the absolute magnitude of future prices (reflected in negative R² on test data) proved extremely difficult.
*   **Overfitting Mitigation:** Advanced architectures like GRU, combined with regularization techniques, helped reduce overfitting compared to Simple RNNs, but it remained a significant challenge.

## Limitations and Future Work

*   **Improving Test R²:** The primary area for future work is to improve the R² scores on the test set to achieve positive values, indicating true predictive power for price variance.
*   **Enhanced Overfitting Control:** Explore more aggressive regularization, different model complexities, or larger datasets if available.
*   **Feature Engineering:** Incorporate external factors like market sentiment, macroeconomic indicators, or advanced technical indicators.
*   **Advanced Architectures:** Experiment with Transformer models, Attention mechanisms, or hybrid CNN-RNN architectures.
*   **More Sophisticated Hyperparameter Tuning:** Employ techniques like Bayesian optimization or more extensive grid/random searches.
*   **Alternative Loss Functions:** Investigate loss functions more aligned with financial objectives (e.g., penalizing incorrect directional predictions more heavily).

## Conclusion

This project successfully implemented RNN-based models for stock price prediction, highlighting the effectiveness of GRUs over Simple RNNs in capturing trends and improving generalization for this task. The robust data preprocessing was a key enabler. While the models demonstrated strong fitting capabilities on training data and good trend-following on test data (especially GRU), accurately predicting price magnitudes (indicated by test R²) remains a significant challenge requiring further research and model refinement.
