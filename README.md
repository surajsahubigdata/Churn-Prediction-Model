# Churn-Prediction-Model

## Problem Statement: 
Financial institutions, such as banks and insurance companies, face the constant challenge of customer churn, where customers stop using their services. Predicting customer churn allows these institutions to proactively engage at-risk customers with retention strategies. The goal is to build a model that can predict whether a customer is likely to leave the financial institution within a specific time frame, based on historical data.

## Features
+ **Churn Prediction**: Classifies whether a customer is likely to churn (exit) or not based on various features.
+ **User Input**: Users can input customer features via the web interface, and the app provides the predicted result along with the churn probability.
+ **Model Evaluation**: The model's performance is evaluated using cross-validation, accuracy score, confusion matrix, and other metrics.
+ **Hyperparameter Tuning**: Performed using GridSearchCV to optimize the Random Forest model's performance.
+ **Interactive Web Interface**: Created using Streamlit for easy interaction.

## Technologies Used
+ **Python**: Programming language.
+ **Pandas and NumPy**: For data manipulation and analysis.
+ **Seaborn and Matplotlib**: For visualization of the dataset.
+ **Scikit-learn**: For model training, evaluation, and hyperparameter tuning.
+ **Random Forest Classifier**: The machine learning algorithm used for classification.
+ **Streamlit**: For building an interactive web application.

## Dataset
### The dataset used for this project is the Churn Modelling dataset. It contains features such as:

+ **CreditScore**: The customer's credit score.
+ **Geography**: The country of the customer.
+ **Gender**: The gender of the customer.
+ **Age**: The age of the customer.
+ **Tenure**: How long the customer has been with the bank.
+ **Balance**: The account balance of the customer.
+ **NumOfProducts**: The number of products the customer uses.
+ **HasCrCard**: Whether the customer has a credit card or not.
+ **IsActiveMember**: Whether the customer is an active member.
+ **EstimatedSalary**: The estimated salary of the customer.
+ **Exited**: Whether the customer churned (1) or not (0).

## Model

The project uses a Random Forest Classifier, a robust and efficient ensemble method for classification. The model is trained on various features to predict if a customer will exit (churn) or not.

+ **Cross-Validation**: The model's accuracy is validated using 5-fold cross-validation.
+ **Hyperparameter Tuning**: Parameters such as n_estimators, max_depth, min_samples_split, and others are tuned using GridSearchCV for improved accuracy.

## Installation
+ Clone the repository:

    Inline `code`

    git clone https://github.com/surajsahubigdata/Churn-Prediction-Model.git

+ Create virtual environment and install the required packages:

    Inline `code`

    conda create -p venv python==3.10

    pip install -r requirements.txt

## Usage

+ Run the streamlit application:

    Inline `code`

    streamlit run interface.py

+ Open your web browser and navigate to http://localhost:8501
+ Use the sidebar to enter customer information (e.g., credit score, age, balance, etc.).
+ The app will predict whether the customer is likely to churn and display the result.

## Model Performance
+ **Cross-Validation Accuracy**: The model's cross-validation accuracy is printed in the console during training.
+ **Confusion Matrix and Other Metrics**: After prediction, the model's performance is evaluated using precision, recall, F1-score, and AUC-ROC.

## Acknowledgements

[GithubAccount](https://github.com/)
[VSCodeIDE](https://code.visualstudio.com/download)
[Streamlit](https://streamlit.io/)

