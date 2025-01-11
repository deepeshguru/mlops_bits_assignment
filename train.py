import mlflow
import mlflow.sklearn
import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Set experiment name
mlflow.set_experiment("Model_Comparison_Experiment")

# Load dataset
data = pd.read_csv('data/titanic.csv')
data = data.dropna()

# Features and target
X = data[['Pclass', 'Age', 'SibSp', 'Fare']]
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Models dictionary
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=50)
}

# Experiment runs
for model_name, model in models.items():
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_param("model_name", model_name)

        # Record training time
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        train_time = end_time - start_time
        mlflow.log_metric("training_time", train_time)

        # Make predictions
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # Log accuracy
        mlflow.log_metric("accuracy", acc)

        # Log classification report as artifact
        report = classification_report(y_test, predictions,
                                       target_names=data.target_names)
        report_filename = f"{model_name}_classification_report.txt"
        with open(report_filename, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_filename)

        # Optionally, log the model
        mlflow.sklearn.log_model(model, f"{model_name}_model")
