# Crop Price Prediction

A Flask-based crop price forecasting dashboard for the two supported crops in the project: Groundnut and Paddy.

## Overview

This application provides a simple web form for entering a year, selecting either a month or season, and choosing a crop. It then predicts a crop price using the bundled ML model and historical crop-price patterns when needed.

## Project structure

- `app.py` — Flask application and prediction endpoint
- `Maincode.py` — ML training pipeline and data processing utilities
- `templates/index.html` — dashboard UI
- `groundnut4.csv` — historical Groundnut price dataset
- `paddy4.csv` — historical Paddy price dataset
- `requirements.txt` — Python package dependencies
- `*.pkl` — trained model artifacts

## Requirements

- Python 3.11+
- pip
- Windows PowerShell or any shell with Python access

## Setup

1. Open a terminal in the project folder.
2. Create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

3. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

## Run the app

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## API usage

Example request:

```powershell
curl -X POST http://127.0.0.1:5000/predict `
  -H "Content-Type: application/json" `
  -d "{\"year\":2023,\"month\":\"January\",\"crop_name\":\"Groundnut\"}"
```

## Notes

- The application supports two crop types: `Groundnut` and `Paddy`.
- If the saved model files are not available or cannot be loaded, the app uses a historical fallback estimate so the service continues to work.
- The project does not require a database or external API credentials.
- Model artifact compatibility can vary across Python/scikit-learn versions; if you retrain the models, use the same environment consistently.

## Known caveats

- This app is designed for a local project environment and not a production deployment by default.
- If the host machine blocks native Python libraries, the environment may need to be recreated in a clean virtual environment or a machine without those restrictions.
