# Student-Performance-Analysis-and-Prediction-System
Student Performance Analysis and Prediction System using Python and Machine Learning. Analyzes academic factors, visualizes student performance patterns, and predicts examination scores using a regression model.


## Project Overview

The **Student Performance Analysis and Prediction System** is a minor project developed as part of the AI/ML with Python Training Program at Interns Elite.

The project uses Python and Machine Learning to analyze student academic performance, identify patterns in educational data, and predict expected examination scores based on selected academic and lifestyle-related factors.

The system analyzes factors such as study hours, attendance, sleep hours, previous examination scores, tutoring sessions, and physical activity.

## Project Objectives

- Load and analyze student performance data using Pandas and NumPy.
- Handle missing values and duplicate records.
- Perform data cleaning and preprocessing.
- Calculate descriptive statistics.
- Explore relationships between academic factors and examination scores.
- Create visualizations using Matplotlib and Seaborn.
- Develop a regression model to predict examination scores.
- Evaluate model performance using MAE, RMSE, and R².
- Generate predictions for sample student data.

## Dataset Description

**Dataset name:** `Student Performance Factors.csv`

The dataset contains:
- 6,607 records
- 20 columns

It includes information about:
- Hours studied
- Attendance
- Previous examination scores
- Sleep hours
- Tutoring sessions
- Physical activity
- Parental involvement
- Access to resources
- School type
- Other student-related factors

**Target variable:** `Exam_Score`

**Dataset source:** Add the original dataset source or download URL here before publishing.

## Technologies and Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- VS Code

## Machine Learning Model

The project currently implements a **Ridge Regression** model using Scikit-learn.

Ridge Regression is a regularized linear regression algorithm used to predict numerical values.

### Input Features

1. Hours_Studied
2. Attendance
3. Sleep_Hours
4. Previous_Scores
5. Tutoring_Sessions
6. Physical_Activity

### Target Variable

`Exam_Score`

### Model Workflow

1. Load the dataset.
2. Remove duplicate records.
3. Select input features and target variable.
4. Split the dataset into training and testing sets (80:20).
5. Handle missing values using median imputation.
6. Apply feature scaling using StandardScaler.
7. Train the Ridge Regression model.
8. Perform five-fold cross-validation.
9. Apply hyperparameter tuning using GridSearchCV.
10. Generate predictions on unseen test data.
11. Evaluate the model and save the trained model.

**Note:** The project guidelines specify Linear Regression. The current implementation uses Ridge Regression, which is a regularized linear regression method.

## Model Evaluation

The model is evaluated using the following metrics:

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 1.266 |
| Root Mean Squared Error (RMSE) | 2.251 |
| R² Score | 0.642 |
| Best Alpha | 0.010 |
| Cross-Validation MAE | 1.327 |

### Result Interpretation

- **MAE (1.266):** The average absolute difference between actual and predicted examination scores is approximately 1.266 points.
- **RMSE (2.251):** Measures prediction error while giving greater weight to larger errors.
- **R² Score (0.642):** The model explains approximately 64.2% of the variation in examination scores on the test dataset.

These results are based on the held-out test dataset and should not be interpreted as a guarantee of future student performance.

## Sample Prediction

The trained model was tested using sample student information.

Example input:

- Hours Studied: 3
- Attendance: 90
- Sleep Hours: 8
- Previous Scores: 75
- Tutoring Sessions: 2
- Physical Activity: 4

**Predicted Examination Score: 64.69**

Additional sample predictions:

| Student | Predicted Score |
|---|---:|
| Student 1 | 58.72 |
| Student 2 | 67.41 |
| Student 3 | 63.63 |

These are illustrative model predictions.

## Project Folder Structure

```text
Student_Performance_Analysis/
│
├── data/
│   └── Student Performance Factors.csv
│
├── notebooks/
│   └── Student_Performance_Analysis.ipynb
│
├── src/
│   ├── model.py
│   └── student_performance_model.joblib
│
├── outputs/
│   ├── visualizations/
│   └── evaluation_results/
│
├── screenshots/
│
├── Project_Report.pdf
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the Project Folder

```bash
cd Student_Performance_Analysis
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Requirements

The following libraries are required:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

## How to Run the Project

### Run the Jupyter Notebook

Open VS Code, select the project's Python virtual environment as the notebook kernel, and open:

```text
notebooks/Student_Performance_Analysis.ipynb
```

Run the notebook cells sequentially to perform data analysis, generate visualizations, and test predictions.

### Run the Machine Learning Model

From the project root directory, execute:

```powershell
python src/model.py
```

The script trains the model, evaluates its performance, saves prediction results, and stores the trained model.

## Output Files

The project generates the following evaluation files:

- `descriptive_statistics.csv`
- `model_comparison.csv`
- `model_evaluation_metrics.csv`
- `actual_vs_predicted_results.csv`
- `standalone_model_metrics.csv`
- `standalone_predictions.csv`
- `sample_student_predictions.csv`

The trained model is saved as:

```text
src/student_performance_model.joblib
```

Visualizations are stored in:

```text
outputs/visualizations/
```

## Visualizations

The project includes the following exploratory data analysis visualizations:

- Examination score histogram
- Bar chart of average scores by parental involvement
- Study hours versus examination score scatter plot
- Attendance versus examination score scatter plot
- Previous scores versus examination score plot
- Examination score box plot
- Numerical correlation heatmap

Screenshots of the visualizations and model outputs can be found in the `screenshots/` folder.

## Limitations

- The model uses a selected set of numerical features.
- Predictions may differ from actual examination results.
- The model's performance depends on the quality and representativeness of the dataset.
- The optional pass/fail classification model has not been implemented.
- The model identifies statistical patterns and does not establish cause-and-effect relationships.

## Future Improvements

- Implement and compare a standard Linear Regression model.
- Include additional relevant student features.
- Compare multiple machine learning algorithms.
- Implement an optional pass/fail classification model.
- Develop a web-based interface for student score prediction.
- Improve model performance using additional datasets and feature engineering.

## Conclusion

The Student Performance Analysis and Prediction System demonstrates how Python, data analysis, and machine learning can be applied to educational data.

The project explores academic performance patterns, develops a regression-based prediction system, and evaluates its performance using standard regression metrics.

It provides a practical demonstration of data preprocessing, exploratory data analysis, machine learning model development, and prediction.

## Author

**Shubham Aher**

Sanjivani University

AI/ML with Python Training Program

**Internship Organization:** Interns Elite

**Academic Year:** 2025–2028

**Internship Duration:** 3 Months
