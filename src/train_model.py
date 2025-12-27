import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from data_loader import load_raw_data, add_target
from feature_engineering import add_academic_ratios
from preprocessing import build_preprocessor
from config import RANDOM_STATE, TEST_SIZE

DATA_PATH = "data/raw/students.csv"
MODEL_PATH = "models/dropout_model.joblib"

def main():
    df = load_raw_data(DATA_PATH)
    df = add_target(df)
    df = add_academic_ratios(df)

    X = df.drop(columns=["Target", "dropout"])
    y = df["dropout"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ]
    )

    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Log Loss:", log_loss(y_test, y_prob))

    joblib.dump(model, MODEL_PATH)

if __name__ == "__main__":
    main()

