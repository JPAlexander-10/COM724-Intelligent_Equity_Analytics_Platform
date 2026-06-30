# IEAP — Intelligent Equity Analytics Platform

An end-to-end machine learning decision-support system built for SOLiGence, a multinational financial organisation managing over £150 billion in client investments. Developed as part of COM724 — Applied AI in Business (Assessment AE2), MSc Applied AI and Data Science, Southampton Solent University.

## Overview

The IEAP transforms raw equity market data into actionable Buy, Hold, or Sell trading recommendations through a seven-stage pipeline combining dimensionality reduction, unsupervised clustering, statistical analysis, deep learning forecasting, and ensemble signal classification.

Using ten years of daily price data (2016–2026) for the top 30 NASDAQ-100 constituents by index weight, the platform objectively selects four representative stocks, forecasts their prices, and generates explainable trading signals with real-time scenario analysis.

## Selected Portfolio

| Stock | Sector | Cluster |
|---|---|---|
| ISRG | Medical Robotics | Large Cap Tech & Diversified Growth |
| AMAT | Semiconductor Equipment | High-Performance Semiconductor & AI Hardware |
| TMUS | Telecommunications | Defensive & Steady Growth |
| TSLA | Electric Vehicles / Tech | High Volatility / Speculative Growth |

Stocks were selected using PCA dimensionality reduction followed by K-Means clustering (k=4) on weekly return data across the full 30-stock universe — with one representative stock chosen per cluster based on minimum distance to centroid.

## Pipeline

1. **Data Acquisition** — Historical price data via yfinance, cleaned and validated
2. **Stock Grouping & Selection** — PCA + K-Means clustering to select 4 representative stocks
3. **Correlation Analysis** — Pearson, Spearman, and rolling correlation across the portfolio
4. **Exploratory Data Analysis** — Price trends, return distributions, volatility, drawdown
5. **Forecasting** — ARIMA (baseline) vs LSTM (advanced), compared on a held-out test set
6. **Trading Signals** — Random Forest classifier generating Buy/Hold/Sell signals with feature importance and what-if scenario analysis
7. **System Interface** — Interactive notebook dashboard + standalone Streamlit web application

## Key Results

- **LSTM achieved a 12/12 sweep over ARIMA** across all four stocks and all evaluation metrics (RMSE, MAE, MAPE)
- **Average MAPE improved from 23.85% (ARIMA) to 4.76% (LSTM)** — a 79% reduction in forecast error
- **K-Means clustering produced four economically coherent groups** that mapped onto real-world sectors without any prior domain knowledge
- **Random Forest signal generation** achieved 35.88% average test accuracy, with results critically evaluated against market regime change between the 2016–2023 training window and the 2024–2025 bull market test period

## Repository Structure

```
IEAP-Intelligent-Equity-Analytics-Platform/
├── README.md
├── COM724_AE2_Colab_IPYNB.ipynb      # Full analysis — Tasks 1 to 7
├── ieap_app.py                        # Streamlit web application
├── requirements.txt                   # Python dependencies
└── ieap_data/                         # Exported model outputs and trained models
    ├── ieap_prices.csv
    ├── ieap_returns.csv
    ├── ieap_arima_results.csv
    ├── ieap_lstm_results.csv
    ├── ieap_scenario_results.csv
    ├── ieap_signal_summary.csv
    └── ieap_models/                   # Trained Random Forest classifiers (.pkl)
```

## Running the Streamlit App Locally

```bash
pip install -r requirements.txt
streamlit run ieap_app.py
```

The app opens at `http://localhost:8501` and includes:
- A stock-level analysis view (Overview, Forecasting, Trading Signals, Scenario Analysis)
- A portfolio-level overview (correlation, volatility, model comparison, summary table)
- A live scenario simulator with interactive sliders for volatility, price level, RSI, and MACD

## Tech Stack

Python · pandas · numpy · scikit-learn · statsmodels · TensorFlow/Keras · Streamlit · Plotly · matplotlib · seaborn

## Author

Jean-Pierre Alexander
MSc Applied AI and Data Science — Southampton Solent University
