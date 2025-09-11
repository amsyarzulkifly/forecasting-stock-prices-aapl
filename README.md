# Forecasting Apple Inc. (AAPL) Stock Prices using LSTM Neural Network Model

## 📌 Project Overview  
This project applies **Long Short-Term Memory (LSTM)** neural networks to forecast **Apple Inc. (AAPL) daily closing prices**. Two sliding-window setups are evaluated: a **60-day lookback window** and a **365-day lookback window**.  

Both **one-step forecasting** and **multi-step forecasting** are implemented to capture both short-term and long-term stock price dynamics.  

## 📂 Dataset  
- **Source**: Yahoo Finance (`yfinance` API)  
- **Period**: 2015 – 2024  
- **Variable**: Daily closing price of AAPL  
- **Windows**: 60-day and 365-day sliding windows for feature engineering  

## ⚙️ Methodology  
1. **Data Preprocessing**  
   - Downloaded daily AAPL closing prices from Yahoo Finance API.  
   - Forward-filled missing values to maintain continuity.  
   - Scaled data for neural network training.  

2. **Feature Engineering (Sliding Windows)**  
   - Constructed sequences of **60-day** and **365-day** lookback windows.  

3. **Model Development**  
   - Built LSTM models using **TensorFlow/Keras**.  
   - Architecture: LSTM layers → Dense output layer.  
   - Loss function: Mean Squared Error (MSE).  

4. **Forecasting Strategies**  
   - **One-Step Forecasting**: Predict one day ahead, append prediction, and repeat recursively across the test set.  
   - **Multi-Step Forecasting**: Predict the entire test horizon in a single forward pass.  

5. **Evaluation**  
   - Compared one-step vs multi-step forecasts for both 60-day and 365-day windows.
   - Evaluation metrics: **MSE** and **MAE**.  

### 🛠️ Model Architecture  

The forecasting model is implemented using **TensorFlow/Keras** with the following architecture:  

- **Input Layer**: Accepts sequences of length `Tx` (lookback window, e.g., 60 or 365) with `n_features = 1` (log-transformed closing price).  
- **LSTM Layer**: A single **LSTM layer with 32 units**, responsible for learning temporal dependencies and sequential patterns in stock price data.  
- **Dense Output Layer**: A fully connected layer with **1 neuron**, producing the forecasted stock price for the next day.  

**Training configuration:**  
- **Loss function**: Mean Squared Error (MSE) — penalizes larger errors more strongly.  
- **Optimizer**: RMSprop — efficient for recurrent neural networks.  
- **Metrics**: Mean Absolute Error (MAE) for interpretability.  
- **Epochs**: 100 (with early stopping/checkpointing).  
- **Callbacks**: `ModelCheckpoint` saves the best-performing model.  

---

## 🖼️ Visualizations  

### 📈 60-day Window  
<div align="left">
  <img width="800" height="450" alt="Image" src="https://github.com/user-attachments/assets/f41981fe-42bc-4074-8ee1-c55f5219a15f" />
  <p><em>AAPL Log Price: One-Step vs Multi-Step Forecast (60-day sliding window).</em></p>
</div>  

### 📈 365-day Window  
- **One-Step (365d)**  
<div align="left">
  <img width="800" height="450" alt="Image" src="https://github.com/user-attachments/assets/f29c7bbd-830e-46a0-be93-a7eab7329aec" />
  <p><em>AAPL Log Price: One-Step vs Multi-Step Forecast (365-day sliding window).</em></p>
</div>  

---

## 📊 Model Performance  

### 60-day Window  
| Metric | One-Step | Multi-Step |  
|--------|----------|------------|  
| MSE    |  0.0012  |  0.0201    |  
| MAE    |  0.0280  |  0.1161    |  

### 365-day Window  
| Metric | One-Step | Multi-Step |  
|--------|----------|------------|  
| MSE    |  0.0017  |  0.0215    |  
| MAE    |  0.0320  |  0.1135    |  

## 🚀 Conclusion  
- LSTM models can effectively forecast AAPL stock prices using sliding windows of historical data.
- One-step forecasts remain highly accurate for short-term predictions, while multi-step forecasts capture longer horizons but introduce greater uncertainty.
- A 60-day window often outperforms a 365-day window by focusing on recent trends and ignoring outdated data.
- This project demonstrates the use of deep learning for financial time series forecasting.  

## 📌 Tools Used  
- **Python**  
- **TensorFlow / Keras** — LSTM implementation  
- **scikit-learn** — evaluation metrics & preprocessing  
- **Pandas & NumPy** — data manipulation  
- **Matplotlib** — visualization  
- **yfinance** — stock data retrieval  

## 🔄 How to Run  
1. Clone the repository:  
   ```bash
   git clone <repo-url>
   cd <repo-folder>

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib yfinance scikit-learn tensorflow jupyterlab

3. Run the notebook:
   ```bash
   jupyter notebook "forecasting stocks AAPL 60d.ipynb"
