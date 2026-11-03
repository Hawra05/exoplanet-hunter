#setup 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "../data/kepler_data.csv"   # change this if your file has a different name
SEED = 42
np.random.seed(SEED)


df = pd.read_csv(DATA_PATH, comment="#")
print(df.shape)
df.head()

print(df["koi_disposition"].value_counts())

df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)   # 1 = exoplanet, 0 = false positive


FEATURES = [
    "koi_period",     # orbital period (days)
    "koi_duration",   # transit duration (hours)
    "koi_depth",      # transit depth (ppm)
    "koi_prad",       # planetary radius (Earth radii)
    "koi_teq",        # equilibrium temperature (K)
    "koi_insol",      # insolation flux (Earth flux)
    "koi_model_snr",  # signal-to-noise ratio of the transit
    "koi_steff",      # star effective temperature (K)
    "koi_slogg",      # star surface gravity (log10 cm/s^2)
    "koi_srad",       # star radius (solar radii)
    "koi_impact",     # impact parameter
]
LOG_FEATURES = ["koi_period", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr"]

missing = [c for c in FEATURES if c not in df.columns]
assert not missing, f"These columns are not in your file: {missing}"

data = df[FEATURES + ["label"]].dropna()
print("rows kept after dropping missing values:", len(data))
print(data["label"].value_counts())

def transform(raw):
    """Log-transform the skewed columns. raw is a 2D array in FEATURES order."""
    X = raw.astype(float).copy()
    for name in LOG_FEATURES:
        j = FEATURES.index(name)
        X[:, j] = np.log1p(X[:, j])
    return X

X_raw = data[FEATURES].to_numpy()
y = data["label"].to_numpy()

# shuffle, then 80% train / 20% test
idx = np.random.permutation(len(X_raw))
split = int(0.8 * len(idx))
train_idx, test_idx = idx[:split], idx[split:]

X_train_raw, y_train = X_raw[train_idx], y[train_idx]
X_test_raw,  y_test  = X_raw[test_idx],  y[test_idx]

X_train_t = transform(X_train_raw)
X_test_t  = transform(X_test_raw)

# scale with TRAIN statistics only
mean = X_train_t.mean(axis=0)
std = X_train_t.std(axis=0)
std[std == 0] = 1.0

def scale(X_t):
    return (X_t - mean) / std

def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

X_train = add_bias(scale(X_train_t))
X_test = add_bias(scale(X_test_t))
print(X_train.shape, X_test.shape)


def sigmoid(z):
    z = np.clip(z, -500, 500)          # avoid overflow in exp
    return 1.0 / (1.0 + np.exp(-z))

def log_loss(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)       # avoid log(0)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def gradient_ascent(X, y, lr=0.1, n_iter=2000):
    m, n = X.shape
    w = np.zeros(n)
    losses = []
    for _ in range(n_iter):
        p = sigmoid(X @ w)
        gradient = X.T @ (y - p) / m   # gradient of the average log-likelihood
        w = w + lr * gradient          # step UP the log-likelihood
        losses.append(log_loss(y, p))
    return w, losses

w, losses = gradient_ascent(X_train, y_train, lr=0.1, n_iter=2000)

plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Log-loss (train)")
plt.title("Training loss")
plt.show()
print("final train loss:", round(losses[-1], 4))

def predict_proba(X, w):
    return sigmoid(X @ w)

def predict(X, w, threshold=0.5):
    return (predict_proba(X, w) >= threshold).astype(int)

def metrics(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn}

train_m = metrics(y_train, predict(X_train, w))
test_m = metrics(y_test, predict(X_test, w))

baseline = max(y_test.mean(), 1 - y_test.mean())
print(f"Majority-class baseline accuracy: {baseline:.3f}")
pd.DataFrame({"train": train_m, "test": test_m}).round(3)

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train[:, 1:], y_train)          # sklearn adds its own bias term
sk_pred = clf.predict(X_test[:, 1:])

print("from-scratch test accuracy:", round(test_m["accuracy"], 3))
print("scikit-learn test accuracy:", round(float(np.mean(sk_pred == y_test)), 3))

weights = pd.Series(w[1:], index=FEATURES).sort_values()
weights.plot(kind="barh")
plt.title("Learned weights (scaled features)")
plt.xlabel("Weight")
plt.show()

os.makedirs("../models", exist_ok=True)
np.savez(
    "../models/weights.npz",
    w=w,
    mean=mean,
    std=std,
    features=np.array(FEATURES),
    log_features=np.array(LOG_FEATURES),
    defaults=np.median(X_train_raw, axis=0),   # typical values, used as defaults in the app
)
print("saved ../models/weights.npz")

