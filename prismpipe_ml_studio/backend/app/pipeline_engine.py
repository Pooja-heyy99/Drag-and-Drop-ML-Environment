import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from app.schemas import AlgorithmType, PipelineResult


def _setup_time_metrics(algorithm_type: AlgorithmType) -> tuple[float, float]:
    manual_defaults = {
        "regression": 24.0,
        "classification": 22.0,
        "clustering": 20.0,
        "nlp": 28.0,
        "neural_network": 30.0,
    }
    manual = manual_defaults[algorithm_type]
    builder = manual * 0.30
    return manual, builder


def run_regression(sample_size: int) -> tuple[str, float, str]:
    x, y = make_regression(n_samples=sample_size, n_features=12, noise=9.0, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return "RMSE", rmse, "Lower RMSE indicates better regression fit."


def run_classification(sample_size: int) -> tuple[str, float, str]:
    x, y = make_classification(
        n_samples=sample_size,
        n_features=16,
        n_informative=8,
        n_classes=2,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    return "Accuracy", acc, "Higher accuracy indicates better classifier performance."


def run_clustering(sample_size: int) -> tuple[str, float, str]:
    x, _ = make_blobs(n_samples=sample_size, centers=4, n_features=8, random_state=42)
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = model.fit_predict(x)
    score = float(silhouette_score(x, labels))
    return "Silhouette Score", score, "Higher silhouette score indicates cleaner cluster separation."


def run_nlp(sample_size: int) -> tuple[str, float, str]:
    positives = ["great product", "excellent quality", "love this", "works perfectly"]
    negatives = ["bad experience", "poor quality", "hate this", "does not work"]

    texts: list[str] = []
    labels: list[int] = []
    rng = np.random.default_rng(seed=42)
    for _ in range(sample_size):
        if rng.random() > 0.5:
            texts.append(rng.choice(positives))
            labels.append(1)
        else:
            texts.append(rng.choice(negatives))
            labels.append(0)

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=800)
    model.fit(x_train_vec, y_train)
    preds = model.predict(x_test_vec)
    acc = float(accuracy_score(y_test, preds))
    return "Sentiment Accuracy", acc, "NLP pipeline score based on sentiment classification accuracy."


def run_neural_network(sample_size: int) -> tuple[str, float, str]:
    x, y = make_classification(
        n_samples=sample_size,
        n_features=20,
        n_informative=10,
        n_classes=2,
        random_state=42,
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = float(accuracy_score(y_test, preds))
    return "NN Accuracy", acc, "Neural net score based on holdout-set classification accuracy."


def execute_pipeline(algorithm_type: AlgorithmType, sample_size: int) -> PipelineResult:
    runners = {
        "regression": run_regression,
        "classification": run_classification,
        "clustering": run_clustering,
        "nlp": run_nlp,
        "neural_network": run_neural_network,
    }

    metric_name, metric_value, explanation = runners[algorithm_type](sample_size)
    manual_time, builder_time = _setup_time_metrics(algorithm_type)
    reduction_pct = ((manual_time - builder_time) / manual_time) * 100.0

    return PipelineResult(
        algorithm_type=algorithm_type,
        metric_name=metric_name,
        metric_value=metric_value,
        setup_time_manual_min=manual_time,
        setup_time_builder_min=builder_time,
        setup_time_reduction_pct=reduction_pct,
        explanation=explanation,
    )
