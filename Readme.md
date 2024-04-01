# Wine Quality Prediction

This project is designed to predict the quality of red or white wine based on various features. It utilizes Streamlit for the user interface and incorporates machine learning models from scikit-learn.

## Data

The project uses two datasets:
- Red wine data (`red_wine.csv`)
- White wine data (`white_wine.csv`)

## Preprocessing

The data is preprocessed using the `preprocess_data` function, which separates features (`X`) from the target variable (`y`).

## Model Selection

Users can choose from the following classifiers:
- Decision Tree
- Random Forest
- Extreme Tree (ExtraTreesClassifier)

## Model Training and Evaluation

After selecting a classifier, the model is trained on the chosen wine type (red or white) and evaluated using accuracy score and confusion matrix. The confusion matrix is displayed as a plot to visualize the model's performance.

## Input Values for Prediction

Users can input specific values for wine features such as fixed acidity, volatile acidity, citric acid, etc., via the sidebar. Upon clicking the "Get Prediction" button, the model predicts the quality of the wine based on the input values.

## How to Use

1. Select the wine type (Red or White) from the sidebar.
2. Choose a classifier (Decision Tree, Random Forest, or Extreme Tree) from the sidebar.
3. Input values for wine features in the sidebar.
4. Click the "Get Prediction" button to see the predicted quality of the wine.

## Note

Ensure that the necessary data files (`red_wine.csv` and `white_wine.csv`) are available in the `data` directory before running the application.

