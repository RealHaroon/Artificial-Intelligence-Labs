# 🚗 Used Car Price Predictor

An end-to-end **Machine Learning project** that predicts used car prices in Pakistan using real-world vehicle listings collected from PakWheels.

The project covers the complete machine learning workflow, from **web scraping and data preprocessing to exploratory data analysis, feature engineering, model training, hyperparameter tuning, and deployment with FastAPI**.

The application features a responsive web interface where users can enter vehicle details and receive an estimated market price in Pakistani Rupees (PKR).

## 📊 Model Performance

The best-performing model is a **tuned Random Forest Regressor**, evaluated on a held-out test set.

| Metric | Result |
|---|---:|
| R² Score | **0.8754 (87.54%)** |
| Mean Absolute Error (MAE) | PKR 509,442 |
| Root Mean Squared Error (RMSE) | PKR 1,991,780 |
| Dataset Size | 2,115 scraped listings |
| Listings with valid prices | 2,076 |

The model explains approximately **87.5% of the variance** in used car prices on the test set.

## ✨ Features

- Predicts estimated used car prices in PKR.
- Displays predictions in lakh or crore format.
- Supports multiple car manufacturers, models, and variants.
- Uses vehicle year, mileage, engine capacity, fuel type, transmission, and city.
- Provides category suggestions learned from the fitted preprocessor.
- Validates prediction inputs using Pydantic.
- Exposes prediction functionality through REST API endpoints.
- Includes interactive Swagger API documentation.
- Provides a responsive HTML frontend.
- Reuses the saved preprocessing pipeline for consistent predictions.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Web Scraping | Requests, BeautifulSoup |
| Data Manipulation | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Hyperparameter Tuning | RandomizedSearchCV |
| Backend | FastAPI, Uvicorn |
| Data Validation | Pydantic |
| Model Serialization | Joblib |
| Frontend | HTML |
| Development | Jupyter Notebook |

## 📁 Project Structure

```text
Used Car Price Predictor/
├── main.py
├── readme.md
├── data/
│   ├── pakwheels_raw.csv
│   └── pakwheels_featured.csv
├── model/
│   ├── car_price_model.pkl
│   └── preprocessor.pkl
├── notebook/
│   └── UsedCarPricePredictor.ipynb
├── plots/
└── static/
    └── index.html
```

## 📂 Dataset

### Data Collection

The dataset was collected from [PakWheels](https://www.pakwheels.com/used-cars/) using a custom Python web scraper built with **Requests and BeautifulSoup**.

The scraper was designed to:

- Navigate through used-car listing pages.
- Extract vehicle information from HTML.
- Convert listed prices from lakh/crore formats into numerical PKR values.
- Extract manufacturing year, mileage, engine capacity, fuel type, and transmission.
- Normalize listing URLs and avoid duplicate records.
- Handle temporary request failures with retries.
- Save collected records incrementally to CSV.

### Dataset Overview

The initial dataset contains **2,115 used-car listings** and 10 columns.

| Column | Description |
|---|---|
| `title` | Full vehicle listing title |
| `price_pkr` | Listed vehicle price in Pakistani Rupees |
| `city` | Vehicle listing city |
| `year` | Manufacturing year |
| `mileage_km` | Total distance driven in kilometers |
| `fuel` | Fuel type |
| `engine_cc` | Engine displacement in cubic centimeters |
| `transmission` | Manual or automatic transmission |
| `url` | Original PakWheels listing URL |
| `scraped_at` | Listing collection timestamp |

### Missing Values

The raw dataset contained the following missing values:

| Column | Missing Values |
|---|---:|
| `price_pkr` | 39 |
| `fuel` | 30 |
| `engine_cc` | 23 |
| Other columns | 0 |

### Data Cleaning

The following preprocessing steps were performed:

1. Removed records with missing target prices.
2. Removed records with non-positive prices.
3. Used the most frequent fuel category (**Petrol**) to fill missing fuel values during initial cleaning.
4. Used the median engine capacity to fill missing numerical values.
5. Inspected duplicate listings and suspicious records.
6. Investigated unusual prices using statistical and model-based comparisons.

After excluding listings with missing prices, **2,076 observations** remained before any additional filtering.

For a strictly leakage-safe training workflow, imputation statistics should be fitted on training data only.

### Feature Engineering

Additional features were extracted from vehicle listing titles using regular expressions and manufacturer/model dictionaries.

For example:

```text
Original title:
Toyota Corolla 2015 GLi Automatic 1.3 VVTi for Sale

Extracted features:
Make: Toyota
Model: Corolla
Variant: GLi Automatic 1.3 VVTi
Year: 2015
```

The feature-engineering process included:

- Extracting car manufacturer (`make`).
- Extracting vehicle model (`model`).
- Extracting trim or variant (`variant`).
- Removing unnecessary title information such as ratings and sale text.
- Supporting multiword manufacturers and models.
- Identifying and updating previously unrecognized manufacturers.

Manufacturer names included Toyota, Suzuki, Honda, Mercedes Benz, Daihatsu, Hyundai, Kia, and others.

Vehicle age was also explored as a derived feature, but manufacturing year was retained for the initial modeling experiments to avoid redundant predictors.

### Features Used for Training

| Feature | Type |
|---|---|
| `make` | Categorical |
| `model` | Categorical |
| `variant` | Categorical |
| `year` | Numerical |
| `mileage_km` | Numerical |
| `engine_cc` | Numerical |
| `fuel` | Categorical |
| `transmission` | Categorical |
| `city` | Categorical |
| `price_pkr` | Target Variable |

## 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand the distribution of prices, identify relationships between features, and investigate unusual records.

### Price Distribution

The valid-price dataset showed:

| Statistic | Price (PKR Lakh) |
|---|---:|
| Mean | 47.51 |
| Median | 34.00 |
| Minimum | 2.50 |
| Maximum | 785.00 |
| Standard Deviation | 59.52 |

**Key observation:** Vehicle prices are strongly right-skewed, with a relatively small number of expensive vehicles extending the upper tail.

A logarithmic transformation of the target was explored as a potential modeling improvement.

### Feature Correlations

The approximate Pearson correlations with vehicle price were:

| Feature | Correlation |
|---|---:|
| Engine Capacity | +0.57 |
| Manufacturing Year | +0.33 |
| Mileage | −0.24 |
| Vehicle Age | −0.33 |

These values describe linear associations, not causal relationships or definitive feature importance.

### Outlier Analysis

Outlier detection included:

- Price distribution boxplots.
- Interquartile Range (IQR) analysis.
- Comparing listing prices against median prices for the same manufacturer/model.
- Comparing prices within similar manufacturing-year groups.
- Inspecting suspicious mileage and engine capacity values.

Legitimate luxury vehicles and older inexpensive cars were not automatically removed simply because of their extreme prices.

### Manufacturer and Model Analysis

Prices were analyzed across manufacturers and vehicle models, revealing substantial variation between models, generations, manufacturing years, and variants.

## 🤖 Machine Learning Pipeline

### 1. Train-Test Split

The dataset was split into:

- **80% Training Data**
- **20% Testing Data**

Using `random_state=42` for reproducibility.

### 2. Preprocessing

A scikit-learn `ColumnTransformer` was used to handle numerical and categorical columns.

**Numerical features:**

- `SimpleImputer(strategy="median")`
- `StandardScaler()`

**Categorical features:**

- `SimpleImputer(strategy="most_frequent")`
- `OneHotEncoder(handle_unknown="ignore")`

The preprocessing transformer was fitted on training data and applied to test data.

### 3. Model Training

Two regression algorithms were evaluated:

1. **Ridge Regression**
2. **Random Forest Regressor**

#### Initial Model Comparison

| Model | MAE (PKR) | RMSE (PKR) | R² |
|---|---:|---:|---:|
| Ridge Regression | 985,294 | 2,126,236 | 0.858 |
| Random Forest | 534,353 | 2,012,591 | 0.873 |

Random Forest performed better across all three evaluation metrics.

### 4. Hyperparameter Tuning

`RandomizedSearchCV` was used to tune the Random Forest model with 5-fold cross-validation.

**Best hyperparameters:**

```python
{
    "n_estimators": 100,
    "max_depth": 20,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.7
}
```

#### Final Model Performance

| Metric | Before Tuning | After Tuning |
|---|---:|---:|
| MAE (PKR) | 534,353 | **509,442** |
| RMSE (PKR) | 2,012,591 | **1,991,780** |
| R² | 0.873 | **0.8754** |

The tuned Random Forest was selected as the final prediction model.

## 🚀 Running the Application

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd "Used Car Price Predictor"
```

### 2. Install Dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib pydantic
```

For notebook exploration and scraping, additional packages include:

```bash
pip install requests beautifulsoup4 lxml matplotlib seaborn jupyter
```

**Note:** Use a scikit-learn version compatible with the version used when the model and preprocessing artifacts were saved.

### 3. Start the FastAPI Server

```bash
uvicorn main:app --reload
```

Alternatively, if the application contains a direct execution entry point:

```bash
python main.py
```

### 4. Access the Application

**Web Interface:**

http://127.0.0.1:8000

**Interactive API Documentation:**

http://127.0.0.1:8000/docs

## 🔌 API Endpoints

### Web Interface

```http
GET /
```

Serves the browser-based frontend.

### Health Check

```http
GET /api/health
```

Example response:

```json
{
  "status": "ok",
  "model": "loaded"
}
```

### Model Options

```http
GET /api/options
```

Returns the makes, models, variants, fuel types, transmissions, and cities learned by the fitted preprocessing pipeline.

### Predict Car Price

```http
POST /api/predict
Content-Type: application/json
```

**Example Request:**

```json
{
  "make": "Toyota",
  "model": "Corolla",
  "variant": "GLi 1.3",
  "year": 2020,
  "mileage_km": 50000,
  "engine_cc": 1300,
  "fuel": "Petrol",
  "transmission": "Automatic",
  "city": "Lahore"
}
```

**Example Response:**

```json
{
  "predicted_price_pkr": 4500000,
  "predicted_price_lakh": 45.0,
  "formatted_price": "PKR 45.00 lakh"
}
```

*The response above is illustrative and not an actual model prediction.*

### Test with cURL

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "make": "Toyota",
    "model": "Corolla",
    "variant": "GLi 1.3",
    "year": 2020,
    "mileage_km": 50000,
    "engine_cc": 1300,
    "fuel": "Petrol",
    "transmission": "Automatic",
    "city": "Lahore"
  }'
```

## 💾 Model Artifacts

Two serialized artifacts are used for inference:

**`model/preprocessor.pkl`**

Contains the fitted scikit-learn preprocessing transformer, including imputation, scaling, and categorical encoding.

**`model/car_price_model.pkl`**

Contains the trained Random Forest regression model.

During prediction:

```text
User Input
    ↓
Pydantic Validation
    ↓
Saved Preprocessor
    ↓
Trained Random Forest
    ↓
Predicted Price (PKR)
    ↓
Formatted Price (Lakh/Crore)
    ↓
FastAPI Response
```

## ⚠️ Limitations

- The dataset contains approximately 2,000 labeled vehicle listings, which may not represent the entire Pakistani used-car market.
- Predictions depend on the quality and coverage of scraped listing data.
- The model estimates **listing prices**, not necessarily final negotiated selling prices.
- Important factors such as accident history, vehicle condition, documentation, maintenance, and ownership history are not included.
- Rare manufacturers and variants may have less reliable predictions because of limited training examples.
- Market prices change over time, so the model may require periodic retraining.
- The reported evaluation metrics are based on the collected dataset and should not be interpreted as guaranteed prediction accuracy for every vehicle.

## 🔮 Future Improvements

- Collect more listings to improve dataset coverage.
- Include vehicle condition, registration information, and ownership history.
- Improve manufacturer, model, and variant extraction.
- Compare additional algorithms such as Gradient Boosting, XGBoost, and CatBoost.
- Experiment with log-transformed price prediction.
- Add prediction intervals to communicate uncertainty.
- Retrain periodically using newer listing data.
- Improve automated testing and deployment workflows.

## 🎯 Project Objective

The primary objective of this project is to gain practical experience building an **end-to-end machine learning application** using real-world data.

The project demonstrates skills in:

- Web scraping and data collection
- Data cleaning and preprocessing
- Feature engineering
- Exploratory data analysis
- Regression modeling
- Hyperparameter optimization
- Model evaluation
- REST API development
- Machine learning deployment

## 📌 Disclaimer

This project is intended for educational and informational purposes.

Vehicle price estimates should not be used as the sole basis for buying or selling decisions. Actual market values may differ depending on vehicle condition, location, and market demand.

PakWheels is the source of the collected listing data. This project is independent and is not affiliated with or endorsed by PakWheels.