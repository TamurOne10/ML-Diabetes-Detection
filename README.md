# Diabetes Prediction Using Machine Learning

## Project Description
This project involves analyzing a diabetes dataset to uncover key patterns and trends, followed by building machine learning models to predict the likelihood of diabetes in individuals. The process includes:
- Data cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model evaluation using metrics like accuracy and F1-score

The goal is to create an accurate and interpretable model to aid healthcare providers in identifying at-risk patients.

---

## Project Workflow

### 1. **Importing Libraries**
Essential libraries used:
- `pandas` for data manipulation
- `numpy` for numerical computations
- `seaborn` and `matplotlib` for data visualization
- Suppressing warnings for cleaner outputs

### 2. **Dataset Description**
The dataset is loaded from a CSV file and contains the following columns:
- Pregnancies
- Glucose
- BloodPressure (renamed as BP)
- SkinThickness (renamed as ST)
- Insulin
- BMI
- DiabetesPedigreeFunction (renamed as DPF)
- Age
- Outcome (target variable: 1 indicates diabetes, 0 indicates no diabetes)

Key characteristics:
- 768 records with 9 features
- No missing values
- Some features (e.g., Glucose, BMI, Insulin) contain zeros, which may represent missing or invalid data.

### 3. **Data Cleaning and Preprocessing**
#### Renaming Columns
- Renamed `BloodPressure` to `BP`
- Renamed `SkinThickness` to `ST`
- Renamed `DiabetesPedigreeFunction` to `DPF`

#### Outlier Detection and Removal
Used the Interquartile Range (IQR) method to identify and remove outliers for columns like `BP`.

Example code:
```python
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# Removing outliers for BP
df = df[(df['BP'] >= lower_bound) & (df['BP'] <= upper_bound)]
```

### 4. **Exploratory Data Analysis (EDA)**
#### Summary Statistics
- Mean, standard deviation, and range for each feature were computed.
- Approximately 35% of individuals in the dataset have diabetes (Outcome = 1).

#### Visualizations
- Used pairplots to explore relationships between variables.
- Boxen plots to identify outliers visually.

### 5. **Modeling and Evaluation**
- Split data into training and testing sets.
- Trained multiple machine learning models (e.g., Logistic Regression, Decision Trees).
- Evaluated models using:
  - Accuracy
  - Precision, Recall
  - F1-score

---

## Dataset Insights
#### Before Cleaning
| Feature | Mean  | Min | Max |
|---------|-------|-----|-----|
| Glucose | 120.9 | 0   | 199 |
| BP      | 69.1  | 0   | 122 |
| BMI     | 32.0  | 0   | 67.1 |

#### After Cleaning
| Feature | Mean  | Min | Max |
|---------|-------|-----|-----|
| Glucose | 121.3 | 44  | 199 |
| BP      | 72.4  | 24  | 104 |
| BMI     | 32.1  | 18  | 50.1 |

---

## How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```
3. Load the dataset (`Diabetes.csv`).
4. Run the notebook or Python script to clean the data, perform EDA, and train models.

---

## Results
- Final model achieved an accuracy of XX% and an F1-score of XX.
- Identified key predictors of diabetes: Glucose, BMI, and Age.

---

## Future Work
- Incorporate additional data for better generalization.
- Test more advanced models (e.g., Random Forest, Gradient Boosting).
- Develop a web application for real-time diabetes risk prediction.

---

## Contact
For any questions or suggestions, feel free to contact:
- Name: Tamoor Abbas
- Email: [Tamur110@gmail.com](mailto:Tamur110@gmail.com)

