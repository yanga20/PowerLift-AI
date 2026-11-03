
# 🏋️ PowerLift AI
### Athlete Performance Prediction System Using Machine Learning

PowerLift AI is a Data Science and Machine Learning project that predicts an athlete's total lifting performance in their next recorded powerlifting competition using historical competition records.

The project uses historical OpenPowerlifting data, data preprocessing, feature engineering, and regression algorithms to estimate future competition totals.

---

## 📌 Project Overview

The objective of PowerLift AI is to analyse an athlete's previous competition performance and estimate their total in their next recorded competition.

The system considers historical lifting performance and other relevant features to generate a prediction.

This project demonstrates the practical application of Data Science, Machine Learning, and Python in sports performance analysis.

## 🎯 Objectives

- Analyse historical powerlifting competition records.
- Clean and preprocess the dataset.
- Integrate competition dates.
- Perform feature engineering using previous competition records.
- Train and compare multiple regression algorithms.
- Evaluate model performance using regression metrics.
- Develop an interactive web application using Streamlit.

## 📊 Dataset

**Source:** OpenPowerlifting historical competition dataset.

The dataset includes information such as:

- Athlete name
- Sex
- Age
- Bodyweight
- Equipment category
- Squat performance
- Bench press performance
- Deadlift performance
- Competition total
- Competition information

Competition dates were integrated using competition identifiers.

**Data limitation:** The historical dataset used in this project ends on 28 January 2018. Therefore, predictions reflect patterns in historical records and should not be interpreted as verified forecasts of current athlete performance.

## ⚙️ Methodology

1. Data Collection
2. Data Cleaning and Preprocessing
3. Competition Date Integration
4. Chronological Sorting
5. Feature Engineering
6. Model Training
7. Model Comparison
8. Model Evaluation
9. Streamlit Application Development

### Feature Engineering

The model uses the following input features:

- PreviousTotalKg
- PreviousSquatKg
- PreviousBenchKg
- PreviousDeadliftKg
- BodyweightKg
- Age
- DaysSincePrevious

**Target variable:** NextTotalKg

The target represents the athlete's total in their next recorded competition.

## 🤖 Machine Learning Algorithms

Three regression algorithms were trained and compared:

| Algorithm | Validation MAE |
|---|---:|
| Linear Regression | 56.78 kg |
| Random Forest Regressor | 54.42 kg |
| Gradient Boosting Regressor | 55.18 kg |

**Selected model:** Random Forest Regressor

Random Forest achieved the lowest validation MAE among the three models.

## 📈 Final Model Evaluation

The selected model was evaluated on the test dataset.

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 47.88 kg |
| Root Mean Squared Error (RMSE) | 88.59 kg |
| R² Score | 0.7422 |

The R² score indicates that the model explains approximately 74.22% of the variation in the test-set target values.

These metrics measure model performance; they do not represent prediction accuracy as a percentage.

## 🖥️ Application

An interactive web interface was developed using Streamlit.

The application allows users to enter athlete-related performance information and obtain a model-generated estimate.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib
- Joblib
- Git and GitHub

## 📁 Project Structure

```text
PowerLift-AI/
│
├── app.py/
│   └── streamlit_app.py
│
├── data/
│
├── models/
│   └── powerlift_model.pkl
│
├── outputs/
│   ├── model_comparison.csv
│   └── sample_predictions.csv
│
├── src/
│   ├── check_history.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── merge_competition_dates.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── verify_dates.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/PowerLift-AI.git
```

### 2. Navigate to the project folder

```bash
cd PowerLift-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py/streamlit_app.py
```

**Note:** The trained model file must be present in the `models` folder. The original dataset files are not included in the repository and must be obtained separately from the dataset source if you want to reproduce the data preparation and training workflow.

## ⚠️ Limitations

- The dataset contains historical records ending in January 2018.
- Athlete identities are grouped using available identifying attributes, which may not uniquely identify individuals.
- Missing values and inconsistencies in historical competition records may affect predictions.
- Model predictions are estimates and are not guaranteed future competition results.

## 🔮 Future Improvements

- Integrate more recent competition data.
- Improve athlete identification and historical record matching.
- Explore additional machine learning algorithms.
- Improve feature engineering.
- Enhance the Streamlit interface.
- Evaluate model performance on newer competition records.

## 👩‍💻 Project Information

**Project Name:** PowerLift AI

**Project Type:** Data Science and Machine Learning

**Developed Using:** Python and Streamlit

**Academic Context:** Summer Internship, 2025–2026

---

⭐ If you find this project interesting, feel free to explore the repository!
