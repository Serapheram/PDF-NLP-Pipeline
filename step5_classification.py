import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, adjusted_rand_score
)

INPUT_CSV = "dataset_cleaned.csv"


def load_data():
    texts, labels = [], []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["cleaned_text"])
            labels.append(row["category"])
    return texts, labels


def evaluate_model(name, ngram_label, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"[{name} | {ngram_label}] Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
    return {"model": name, "ngram": ngram_label, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}


if __name__ == "__main__":
    texts, labels = load_data()
    n_classes = len(set(labels))
    print(f"Loaded {len(texts)} documents across {n_classes} categories: {set(labels)}")

    # NOTE: with a very small dataset (like 12 docs), train/test split is tiny.
    # This is expected -- mention it as a limitation in your report.
    results = []
    ngram_ranges = {"unigram": (1, 1), "bigram": (2, 2), "trigram": (3, 3)}

    best_f1 = -1
    best_info = None  # (name, ngram, y_test, y_pred, model)

    for ngram_label, ngram_range in ngram_ranges.items():
        vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1)
        X = vectorizer.fit_transform(texts)

        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.3, random_state=42, stratify=labels
        )

        # --- Naive Bayes ---
        nb = MultinomialNB()
        nb.fit(X_train, y_train)
        y_pred_nb = nb.predict(X_test)
        res_nb = evaluate_model("NaiveBayes", ngram_label, y_test, y_pred_nb)
        results.append(res_nb)
        if res_nb["f1"] > best_f1:
            best_f1 = res_nb["f1"]
            best_info = ("NaiveBayes", ngram_label, y_test, y_pred_nb)

        # --- Logistic Regression ---
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        res_lr = evaluate_model("LogisticRegression", ngram_label, y_test, y_pred_lr)
        results.append(res_lr)
        if res_lr["f1"] > best_f1:
            best_f1 = res_lr["f1"]
            best_info = ("LogisticRegression", ngram_label, y_test, y_pred_lr)

    # ---- Classification model comparison graph ----
    model_names = [f"{r['model']}\n({r['ngram']})" for r in results]
    f1_scores = [r["f1"] for r in results]

    plt.figure(figsize=(12, 6))
    plt.bar(model_names, f1_scores, color="mediumpurple")
    plt.ylabel("F1-score")
    plt.title("Classification Model Comparison (F1-score)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("graph_model_comparison.png", dpi=120)
    plt.close()
    print("Saved graph: graph_model_comparison.png")

    # ---- Confusion matrix for best model ----
    best_name, best_ngram, y_test_best, y_pred_best = best_info
    print(f"\nBest model: {best_name} on {best_ngram} features (F1={best_f1:.3f})")

    cm = confusion_matrix(y_test_best, y_pred_best, labels=sorted(set(labels)))
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(set(labels)))
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - Best Model ({best_name}, {best_ngram})")
    plt.tight_layout()
    plt.savefig("graph_confusion_matrix.png", dpi=120)
    plt.close()
    print("Saved graph: graph_confusion_matrix.png")

    # ==== K-Means Clustering ====
    vectorizer_all = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
    X_all = vectorizer_all.fit_transform(texts).toarray()

    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_all)

    ari = adjusted_rand_score(labels, cluster_labels)
    print(f"\nK-means Adjusted Rand Index vs true categories: {ari:.3f}")
    print("(closer to 1.0 = clusters match true categories well; "
          "closer to 0 = little agreement)")

    # PCA for 2D visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    label_to_color = {lab: i for i, lab in enumerate(sorted(set(labels)))}
    true_colors = [label_to_color[l] for l in labels]

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=true_colors, cmap="tab10", s=100)
    axes[0].set_title("Documents by TRUE Category (PCA)")
    for i, lab in enumerate(labels):
        axes[0].annotate(lab, (X_pca[i, 0], X_pca[i, 1]), fontsize=7)

    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap="tab10", s=100)
    axes[1].set_title("Documents by K-MEANS Cluster (PCA)")
    for i, c in enumerate(cluster_labels):
        axes[1].annotate(str(c), (X_pca[i, 0], X_pca[i, 1]), fontsize=7)

    plt.tight_layout()
    plt.savefig("graph_kmeans_pca.png", dpi=120)
    plt.close()
    print("Saved graph: graph_kmeans_pca.png")

    print("\nAll classification + clustering results saved.")
