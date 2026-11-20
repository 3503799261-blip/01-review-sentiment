"""Tiny, offline Chinese review sentiment benchmark; no remote model or API."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

SEED = 42
ROOT = Path(__file__).resolve().parent


def load_reviews(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or any(not row["text"].strip() or row["label"] not in {"0", "1"} for row in rows):
        raise ValueError(f"Expected nonempty text and labels 0/1 in {path}")
    return [row["text"] for row in rows], np.array([int(row["label"]) for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="包装虽然简单，但是东西很好用，我很满意。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if not args.text.strip():
        parser.error("--text must not be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = load_reviews(ROOT / "data" / "train.csv")
    test_x, test_y = load_reviews(ROOT / "data" / "test.csv")
    if set(train_x) & set(test_x):
        raise ValueError("Exact duplicate reviews found across train and test")

    # Fit vocabulary and IDF on training text only.
    model = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), sublinear_tf=True),
        LogisticRegression(C=4.0, max_iter=1000, random_state=SEED),
    )
    model.fit(train_x, train_y)
    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train_y), 1)), train_y)
    prediction = model.predict(test_x)
    baseline_prediction = baseline.predict(np.zeros((len(test_y), 1)))

    def score(pred):
        return {
            "accuracy": float(accuracy_score(test_y, pred)),
            "positive_f1": float(f1_score(test_y, pred, zero_division=0)),
            "macro_f1": float(f1_score(test_y, pred, average="macro", zero_division=0)),
        }

    metrics = {
        "project": ROOT.name,
        "seed": SEED,
        "data_source": "Repository-authored teaching reviews; fixed independently worded test set, not a public benchmark.",
        "train_size": len(train_y),
        "test_size": len(test_y),
        "split": "Fixed authored train/test files; no tuning on test labels",
        "metrics": {"tfidf_logistic_regression": score(prediction), "majority_baseline": score(baseline_prediction)},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    probabilities = model.predict_proba(test_x)[:, 1]
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "actual", "predicted", "positive_probability"])
        writer.writerows(zip(test_x, test_y, prediction, probabilities))
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(test_y, prediction, display_labels=["negative", "positive"], ax=ax, colorbar=False)
    ax.set_title("Independent toy test set")
    fig.tight_layout()
    fig.savefig(args.output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    demo = {
        "text": args.text,
        "label": "正面" if model.predict([args.text])[0] else "负面",
        "positive_probability": float(model.predict_proba([args.text])[0, 1]),
        "note": "Probability is not calibrated; short or sarcastic reviews can fail.",
    }
    (args.output_dir / "demo.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": metrics["metrics"], "demo": demo}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
