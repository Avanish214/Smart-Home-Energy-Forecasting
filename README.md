# Wattwise: Smart Home Energy Forecasting

A Streamlit dashboard for exploring household active power and forecasting the next four hours with a multi-step LSTM. It uses the UCI Individual Household Electric Power Consumption dataset, sampled at one-minute intervals from December 2006 through November 2010.

## Run locally

Python 3.11 or 3.12 is recommended for TensorFlow compatibility.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens with synthetic sample readings so it can be explored immediately. Choose **Load UCI dataset** in the sidebar to download and process the original archive; the compressed file is stored locally in `data/`. Choose **Train LSTM** to fit the model on whichever series is loaded. Training runs in the current Streamlit session and is not persisted after restarting the app.

## Forecast design

- Global active power is parsed as kilowatts, resampled to 15-minute means, and missing readings are time-interpolated.
- The model input is the previous 24 hours (96 intervals); the model predicts 16 future intervals (four hours).
- Data is split chronologically: the first 85% is used for training, and future forecast windows are reserved for validation. Training is capped at 10,000 windows to keep CPU training practical.
- The dashboard reports held-out mean absolute error in kW. Daily energy is estimated by summing 15-minute mean power multiplied by 0.25 hours.
- Initial sample forecasts are illustrative, not LSTM output. The interface labels synthetic readings and forecasts explicitly.

## Dataset

UCI Machine Learning Repository, [Individual Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption). The model uses the `Global_active_power` feature only. Energy-saving prompts are general suggestions, not tariff-specific advice.
