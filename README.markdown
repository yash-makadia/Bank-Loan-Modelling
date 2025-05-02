# Bank Loan Modelling
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Overview

This data science project predicts the likelihood of bank customers accepting personal loans, developed as a personal project to enhance marketing campaigns for banking products. Using Python, we conducted exploratory data analysis (EDA), applied feature engineering, and built multiple classification models to identify potential loan customers. The goal is to increase the success ratio of marketing campaigns while minimizing costs by targeting customers with a higher probability of purchasing loans.

## Features

- **Exploratory Data Analysis**: Visualizes distributions of customer demographics, financial behaviors, and loan acceptance rates.
- **Data Preprocessing**: Cleans and transforms skewed features to improve model performance.
- **Classification Models**: Implements Logistic Regression, Random Forest, Decision Tree, and Naive Bayes to predict loan acceptance.
- **Actionable Insights**: Identifies key customer characteristics for targeted marketing campaigns.

## Dataset

The dataset, sourced from [Kaggle](https://www.kaggle.com/itsmesunil/bank-loan-modelling), contains data on 5,000 bank customers with the following attributes:
- **Demographic**: Age, Income, Family size, Education level (1: Undergrad, 2: Graduate, 3: Advanced/Professional).
- **Financial**: Average credit card spending (CCAvg), Mortgage value, Securities Account, CD Account, Online banking usage, Credit Card usage.
- **Target**: Personal Loan acceptance (0: No, 1: Yes; 9.6% acceptance rate).
- **Other**: ZIP Code, Years of professional experience.

**Access**: Download the dataset from [https://www.kaggle.com/itsmesunil/bank-loan-modelling/download](https://www.kaggle.com/itsmesunil/bank-loan-modelling/download). Due to size, it is not hosted in the repository but can be processed using the provided code.

## Methodology

### Data Preparation
- **Data Collection**: Loaded the `Bank_Personal_Loan_Modelling.xlsx` dataset using Pandas.
- **Preprocessing**:
  - Dropped irrelevant columns: `ID` (no predictive value), `ZIP Code` (nominal, 467 unique values), and `Experience` (highly correlated with Age, contained negative values).
  - Checked for null values (none found) and verified data types.
- **Feature Engineering**:
  - Applied Yeo-Johnson transformation to normalize skewed features: `Income` and `CCAvg`.
  - Binned `Mortgage` into discrete intervals (0–600) to handle skewness.
  - Standardized features using `StandardScaler` for model compatibility.

### Exploratory Data Analysis (EDA)
- **Univariate Analysis**:
  - `Age`: Normally distributed (mean ~45 years).
  - `Income`, `CCAvg`, `Mortgage`: Right-skewed, addressed via transformations.
  - `Family`: Even distribution across sizes 1–4.
  - `Education`: Most customers have undergraduate (41.9%) or advanced (30%) education.
  - `Securities Account`: 10.4% have accounts.
  - `CD Account`: 6% have accounts.
  - `Online`: 59.7% use online banking.
  - `CreditCard`: 29.4% use bank-issued credit cards.
  - `Personal Loan`: 9.6% acceptance rate.
- **Bivariate Analysis**:
  - Higher income correlates with loan acceptance, but education level 1 (Undergrad) customers have higher incomes.
  - Customers with CD Accounts are highly likely to accept loans.
  - Family size 3 is slightly more likely to take loans.
  - `Income` and `CCAvg` are highly correlated.

### Modeling
- **Data Split**: 70:30 train-test split with stratification to maintain class balance.
- **Models**:
  - **Logistic Regression**: Baseline model with 95.47% test accuracy, F1-score 0.73, ROC AUC 0.82.
  - **Random Forest Classifier**: Ensemble model with 98.73% test accuracy, F1-score 0.93, ROC AUC 0.94 (best performer).
  - **Decision Tree Classifier**: Achieved 98% test accuracy, F1-score 0.89, ROC AUC 0.93.
  - **Naive Bayes**: Lower performance with 91.53% test accuracy, F1-score 0.55, ROC AUC 0.75.
- **Evaluation Metrics**:
  - Accuracy, precision, recall, F1-score, and ROC AUC score.
  - Confusion matrices visualized to assess true positives and false positives.
- **Key Findings**: Random Forest outperformed others due to its ability to handle complex feature interactions and imbalanced classes.

### Visualizations
- **Univariate**: Histograms for `Age`, `Income`, `CCAvg`, `Mortgage`; count plots for categorical variables (`Family`, `Education`, `CreditCard`, `Online`).
- **Bivariate**: Box plots (`Income` vs. `Education` by `Personal Loan`), count plots (`Securities Account`, `CD Account`, `Family` by `Personal Loan`), correlation heatmap, pair plots.
- **Model Performance**: Pie chart for loan acceptance rate, confusion matrix heatmaps.

### Libraries
- Python libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.

## Setup Instructions

### Prerequisites
- Python (version 3.8 or higher).
- Jupyter Notebook or any Python IDE (e.g., VS Code, PyCharm).
- Install required libraries:
  ```bash
  pip install pandas numpy matplotlib seaborn scikit-learn jupyter
  ```

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yash-makadia/Bank-Loan-Modelling.git
   cd Bank-Loan-Modelling
   ```
2. **Download the Dataset**:
   - Download `Bank_Personal_Loan_Modelling.xlsx` from [Kaggle](https://www.kaggle.com/itsmesunil/bank-loan-modelling/download).
   - Place the file in the repository root or update the file path in `Bank_Loan_Modeling_Final.ipynb`.
3. **Run the Code**:
   - Open `Bank_Loan_Modeling_Final.ipynb` in Jupyter Notebook.
   - Update the dataset path in the notebook as needed.
   - Execute the cells to preprocess data, perform EDA, generate visualizations, and train classification models.

## Key Findings

- **Loan Acceptance**: Only 9.6% of customers accepted personal loans, indicating a highly imbalanced dataset.
- **Feature Importance**: `Income`, `CCAvg`, `CD Account`, and `Education` are strong predictors of loan acceptance.
- **Customer Profiles**:
  - Customers with CD Accounts are highly likely to accept loans.
  - Higher income and credit card spending correlate with loan acceptance.
  - Family size 3 shows a slight tendency to accept loans.
- **Model Performance**: Random Forest Classifier achieved the highest accuracy (98.73%), F1-score (0.93), and ROC AUC (0.94), making it the best model for deployment.

## Impact and Recommendations

- **For Bank Marketing Teams**:
  - Target customers with CD Accounts, as they are highly likely to accept loans.
  - Focus on customers with higher incomes and credit card spending.
  - Prioritize family sizes of 3 for targeted campaigns.
- **Campaign Optimization**:
  - Use the Random Forest model to score customers by loan acceptance probability, reducing campaign costs by focusing on high-probability customers.
  - Monitor false positives to avoid wasting resources on unlikely candidates.
- **Business Impact**:
  - Increase loan conversion rates beyond the previous 9% success rate.
  - Reduce marketing costs by targeting fewer, more promising customers.
  - Enhance customer retention by offering tailored loan products.

## Project Files

- `Bank_Loan_Modeling_Final.ipynb`: Jupyter Notebook with code for data preprocessing, EDA, visualizations, and model training.
- `docs/Bank_Loan_Modeling_Final.pdf`: Exported Jupyter Notebook detailing the project methodology and findings.
- `LICENSE`: GNU General Public License v3.0.

## Challenges and Lessons Learned

- **Data Imbalance**: The 9.6% loan acceptance rate required careful handling (stratified splitting, evaluating F1-score and ROC AUC).
- **Feature Skewness**: Transforming `Income`, `CCAvg`, and `Mortgage` improved model performance.
- **Key Takeaways**:
  - Feature engineering is critical for handling skewed data in classification tasks.
  - Ensemble methods like Random Forest excel in imbalanced datasets.
  - Visualizations (e.g., heatmaps, box plots) enhance interpretability of customer behaviors.

## Future Scope

- Incorporate additional customer data (e.g., transaction history) for richer feature sets.
- Explore advanced models (e.g., XGBoost, Neural Networks) for improved accuracy.
- Develop a real-time prediction API for integration into marketing systems.
- Create an interactive dashboard to visualize customer segments and model predictions.

## References

- [Kaggle Dataset](https://www.kaggle.com/itsmesunil/bank-loan-modelling)
- Python Documentation: [pandas](https://pandas.pydata.org/), [numpy](https://numpy.org/), [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/), [scikit-learn](https://scikit-learn.org/)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes. Ensure compliance with the GPL-3.0 license.

## Contact

For questions or feedback, please open an issue on GitHub or contact the project maintainer at [yashmakadia1908@gmail.com](mailto:yashmakadia1908@gmail.com).

## Acknowledgments

Special thanks to the open-source community for providing tools like Pandas, Scikit-learn, and Jupyter, which made this project possible.
