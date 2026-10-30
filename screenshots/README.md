# Student Performance Analysis and Prediction System

## 1. Project Overview

This project analyzes student performance data and develops a machine learning model to predict students' exam scores based on selected academic and lifestyle factors.

The project uses Python for data analysis, visualization, and predictive modeling.

## 2. Objectives

- Clean and prepare the student performance dataset.
- Perform exploratory data analysis (EDA).
- Calculate descriptive statistics.
- Explore relationships between student factors and exam scores.
- Train a regression model to predict exam scores.
- Evaluate model performance using MAE, RMSE, and R².
- Apply cross-validation and hyperparameter tuning.
- Save model predictions, evaluation results, and the trained model.

## 3. Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## 4. Dataset

The dataset contains 6,607 rows and 20 columns.

The target variable is `Exam_Score`. Selected input features include study hours, attendance, sleep hours, previous scores, tutoring sessions, and physical activity.

**Dataset source:** Add the original dataset download URL and its license or usage terms here after verifying them.

## 5. Project Structure

```text
project1/
├── data/
│   └── Student Performance Factors.csv
├── notebooks/
│   └── Student_Performance_Analysis.ipynb
├── src/
│   ├── model.py
│   └── student_performance_model.joblib
├── outputs/
│   ├── visualizations/
│   └── evaluation_results/
├── screenshots/
├── requirements.txt
└── README.md
```

## 6. Setup Instructions

1. Install Python.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment.
4. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

5. Place the dataset in the `data/` directory.
6. Open and run the notebook in VS Code.

## 7. Run the Model

Run the standalone model script from the project root:

```bash
python src/model.py
```

The script trains and tunes a Ridge Regression model and saves the evaluation metrics, predictions, and trained pipeline.

## 8. Evaluation

The project evaluates regression performance using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of determination

Record the final metrics from the completed model run here.

## 9. Limitations

- The model uses a selected subset of available features.
- Relationships in the dataset do not establish causation.
- Performance on this dataset may not generalize to other schools or student populations.
- Predictions are estimates and should not be treated as definitive judgments of student ability.

## 10. Future Improvements

- Compare additional regression algorithms.
- Include categorical features with appropriate preprocessing.
- Analyze feature importance and prediction errors.
- Evaluate the model on an independent dataset.
- Develop a simple user interface for entering student information.

## 11. Author

Name: Add your name

Internship/Institution: Add your internship or institution details