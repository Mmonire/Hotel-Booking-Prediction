# Hotel Booking Prediction Project

## 📖 Overview

This project develops a machine learning model to predict whether a user will book a hotel based on search behavior and related features from a hotel reservation dataset. The goal is to preprocess search queries (e.g., dates, locations, user profiles), train a predictive model, and generate booking probabilities for new searches. This enables real-time decisions like offering discounts to high-intent users. The implementation uses **pandas** for data handling, **scikit-learn** for preprocessing, and **XGBoost** for modeling, with feature importance analysis saved as JSON files.

The project is implemented in a Jupyter Notebook (`will_not_travel_again.ipynb`) and outputs predictions in `submission.csv` along with feature importance JSONs for evaluation.

## 🎯 Objectives

- **Predict Booking Intent**: Forecast if a user will book a hotel (binary classification: book/no-book).
- **Preprocess Complex Data**: Handle categorical (hotel names, cities), temporal (search dates, check-in/out), and numerical features (price, distance).
- **Analyze Feature Importance**: Extract and save top features (e.g., search hour, check-in month) as JSON for interpretability.
- **Generate Submission**: Produce a CSV with predicted probabilities or labels for the test set.

## ✨ Features

- **Data Preprocessing**:
  - Parse dates for features like search hour, check-in month/day, days between search and check-in, length of stay (LOS).
  - Encode categorical variables (hotel ID, city, user segments) using one-hot or label encoding.
  - Handle missing values and outliers in price/distance features.
- **Model Training**:
  - Uses **XGBoost** classifier for gradient boosting on imbalanced data.
  - Cross-validation with ROC-AUC scoring for hyperparameter tuning.
  - Early stopping to prevent overfitting.
- **Feature Importance**:
  - Extracts top features (e.g., `search_hour`, `checkIn_date_month`) and saves as JSON files.
- **Evaluation & Submission**:
  - ROC-AUC and precision-recall metrics.
  - Probability predictions thresholded for binary output if needed.
  - Filtered test data to valid users/hotels from training.

## 🛠 Prerequisites

- **Input Files** (in `data/` directory):
  - `train.csv`: Training data with search features and booking labels.
  - `test.csv`: Test data for predictions.
  - `hotel_info.csv`: Hotel metadata (names, cities, prices).

## 🚀 Usage

1. Open `will_not_travel_again.ipynb` in Jupyter Notebook.
2. Run cells sequentially:
   - Load and explore training/test data.
   - Preprocess: Extract temporal features (e.g., `search_hour`, `checkIn_date_month`), encode categoricals, scale numerics.
   - Train XGBoost model with cross-validation.
   - Extract feature importances and save as JSONs (e.g., `search_hour.json`).
   - Predict on test set and generate `submission.csv`.
3. Output:
   - `submission.csv`: Predicted booking probabilities/labels.
   - JSON files: Feature importances (e.g., `days_between.json`).
   - `result.zip`: Zipped submission package.

### Key Code Snippets

- **Data Loading & Preprocessing**:

  ```python
  train = pd.read_csv('data/train.csv')
  # Extract features like search_hour = train['search_date'].dt.hour
  # One-hot encode 'city', label encode 'hotel_id'
  ```
- **Model Training**:

  ```python
  from xgboost import XGBClassifier
  model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
  model.fit(X_train, y_train)
  ```
- **Feature Importance & Submission**:

  ```python
  importances = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
  top_features = importances.nlargest(5, 'importance')
  top_features.to_json('search_hour.json', orient='records')
  # Predict and save submission
  preds = model.predict_proba(X_test)[:, 1]
  submission = pd.DataFrame({'id': test_id, 'booking_prob': preds})
  submission.to_csv('submission.csv', index=False)
  ```

## 📊 Code Structure

- **Cells 1-3**: Load data, initial EDA (distributions, missing values).
- **Cells 4-6**: Feature engineering (temporal extraction, encoding, scaling).
- **Cells 7-9**: Train-validation split, model training with CV.
- **Cells 10-12**: Evaluation (ROC-AUC), feature importance extraction/saving.
- **Cell 13**: Test predictions, submission generation, zip packaging.

## 🔍 Evaluation

- **Metrics**: ROC-AUC (primary), precision-recall for imbalanced classes.
- **Cross-Validation**: 5-fold CV to tune hyperparameters and estimate performance.
- **Thresholding**: Optimal threshold from ROC curve for binary decisions.
- **Feature Insights**: JSONs highlight key predictors like search timing and stay length.

## 📝 Notes

- **Imbalanced Data**: Booking class is minority; use class weights or SMOTE if needed.
- **Temporal Features**: Critical for intent prediction (e.g., late-night searches may indicate urgency).
- **Scalability**: XGBoost handles large datasets efficiently; GPU acceleration possible.
- **Improvements**: Ensemble methods, SHAP for advanced interpretability, or deep learning for embeddings.