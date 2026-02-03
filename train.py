import pandas as pd
import joblib
import argparse
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path, model_path):
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Hyperparameter Tuning
    param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
    grid.fit(X_train, y_train)

    print(f"Best Params: {grid.best_params_}")
    print(f"Accuracy: {grid.best_score_}")

    # Save
    joblib.dump(grid.best_estimator_, model_path)
    print("Model saved.")

if __name__ == "__main__":
    print('Starting training process...')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/heart.csv")
    parser.add_argument("--model", default="model.pkl")
    args = parser.parse_args()
    main(args.data, args.model)