"""
Naive Bayes – Part 2: Model (Multinomial Naive Bayes Classifier)
---------------------------------------------------------------
Implements the text classification model for the IMDb dataset using the
preprocessed data produced in Part 1.

- Load tokenized data and vocabulary.
- Estimate class priors P(y) and conditional word probabilities P(x_i | y)
  using Laplace smoothing.
- Use log probabilities for numerical stability.
- Predict sentiment labels for new documents.
- Compute and print accuracy on the test set.
"""
import os
import json
import numpy as np

# --------------------------
# Configuration and data I/O
# --------------------------
DATA_DIR = os.environ.get("IMDB_DATA_DIR", "./data")
ACL_DIR = os.path.join(DATA_DIR, "aclImdb")
BASE_DIR = os.path.join(ACL_DIR, "preprocessed_baseline")

# Laplace smoothing parameter
ALPHA = 1.7


def load_preprocessed():
    X_train = np.load(os.path.join(BASE_DIR, "X_train.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(BASE_DIR, "y_train.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(BASE_DIR, "X_test.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(BASE_DIR, "y_test.npy"), allow_pickle=True)
    with open(os.path.join(BASE_DIR, "vocab.json"), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    return X_train, y_train, X_test, y_test, vocab


# ------------------
# Naive Bayes Model
# ------------------
class MultinomialNB:
    def __init__(self, alpha: float = 1.0, vocab_size: int | None = None, pad_id: int | None = None):
        self.alpha = float(alpha)
        self.vocab_size = vocab_size  # should be len(vocab)
        self.pad_id = pad_id
        self.class_log_prior_: dict[int, float] | None = None
        self.feature_log_prob_: dict[int, dict[int, float]] | None = None  # per-class log P(token|class)
        self.unseen_log_prob_: dict[int, float] | None = None  # per-class log P(unseen|class)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Estimate parameters P(y) and P(x_i|y) with Laplace smoothing.
        - Uses len(vocab) for smoothing mass (passed via vocab_size) — this is important so
          unseen tokens receive alpha mass even if they never appear in the data.
        - Skips PAD tokens if a pad_id is provided.
        """
        assert self.vocab_size is not None and self.vocab_size > 0, "vocab_size must be set to len(vocab) before fit()"
        classes = np.unique(y)
        self.class_log_prior_ = {}
        self.feature_log_prob_ = {}
        self.unseen_log_prob_ = {}

        N = len(X)
        for c in classes:
            # documents of class c
            X_c = [doc for doc, label in zip(X, y) if label == c]
            token_counts: dict[int, int] = {}
            total_tokens = 0

            for doc in X_c:
                for tok in doc:
                    if self.pad_id is not None and tok == self.pad_id:
                        continue  # ignore padding
                    token_counts[tok] = token_counts.get(tok, 0) + 1
                    total_tokens += 1

            # Laplace smoothing
            denom = total_tokens + self.alpha * self.vocab_size
            log_denom = np.log(denom)
            self.feature_log_prob_[c] = {tok: np.log(count + self.alpha) - log_denom for tok, count in token_counts.items()}
            self.unseen_log_prob_[c] = np.log(self.alpha) - log_denom  # for any token not seen in class c

            # class prior
            self.class_log_prior_[c] = np.log(len(X_c) / N)

    def predict_log_proba(self, X: np.ndarray):
        assert self.class_log_prior_ is not None and self.feature_log_prob_ is not None and self.unseen_log_prob_ is not None
        log_probs = []
        for doc in X:
            doc_log = {}
            for c, log_prior in self.class_log_prior_.items():
                total = log_prior
                feat = self.feature_log_prob_[c]
                unseen = self.unseen_log_prob_[c]
                for tok in doc:
                    if self.pad_id is not None and tok == self.pad_id:
                        continue
                    total += feat.get(tok, unseen)
                doc_log[c] = total
            log_probs.append(doc_log)
        return log_probs

    def predict(self, X: np.ndarray):
        log_probs = self.predict_log_proba(X)
        return np.array([max(lp, key=lp.get) for lp in log_probs])

# --------------
# Evaluation
# --------------
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def main():
    print("Loading preprocessed data…")
    X_train, y_train, X_test, y_test, vocab = load_preprocessed()

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Vocab size: {len(vocab)}")
    model = MultinomialNB(alpha=ALPHA, vocab_size=len(vocab), pad_id=vocab.get("<PAD>"))

    print("Training Naive Bayes model…")
    model.fit(X_train, y_train)

    print("Predicting on test data…")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc * 100:.2f}%")


if __name__ == "__main__":
    main()
