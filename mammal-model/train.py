"""Train the classification head: a small MLP in pure numpy (no torch/TF needed).

Architecture:  Linear(in_dim -> hidden) -> ReLU -> Dropout -> Linear(hidden -> 6)
Loss: class-weighted cross-entropy (protects data-thin classes like moose/bear).
Optimizer: Adam. Early stopping on validation macro-F1, best weights restored.

Saves: <out_dir>/weights.npz  (W1,b1,W2,b2 + arch meta)
       <out_dir>/history.json (loss / macro-F1 per epoch)
"""
import argparse
import json
import os
import numpy as np

from config import (N_CLASSES, CLASSES, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE,
                    MAX_EPOCHS, EARLY_STOP_PATIENCE, DROPOUT_P, SEED)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y_true, y_pred, n_classes=N_CLASSES):
    fs = []
    for c in range(n_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        fs.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(fs))


class MLPHead:
    def __init__(self, in_dim, hidden=HIDDEN_DIM, n_classes=N_CLASSES,
                 dropout=DROPOUT_P, lr=LEARNING_RATE, seed=SEED):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, hidden)).astype(np.float64)
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, n_classes)).astype(np.float64)
        self.b2 = np.zeros(n_classes)
        self.dropout = dropout
        self.lr = lr
        # Adam state
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.t = 0

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def forward(self, X, train=True, rng=None):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        dropout_mask = None
        if train and self.dropout > 0:
            dropout_mask = (rng.random(a1.shape) > self.dropout) / (1.0 - self.dropout)
            a1 = a1 * dropout_mask
        z2 = a1 @ self.W2 + self.b2
        cache = (X, z1, a1, dropout_mask)
        return softmax(z2), cache

    def predict(self, X):
        p, _ = self.forward(X, train=False)
        return p.argmax(axis=1), p

    def fit(self, Xtr, ytr, Xva, yva, epochs=MAX_EPOCHS, batch=BATCH_SIZE,
            patience=EARLY_STOP_PATIENCE, verbose=True):
        n = len(Xtr)
        counts = np.bincount(ytr, minlength=N_CLASSES).astype(np.float64)
        class_w = n / (N_CLASSES * np.maximum(counts, 1))     # inverse-frequency
        if verbose:
            print("class counts:", dict(zip(CLASSES, counts.astype(int))),
                  "\nclass weights:", np.round(class_w, 3))
        rng = np.random.default_rng(SEED + 1)
        Xtr = Xtr.astype(np.float64)
        history, best_f1, best_params, bad = [], -1, None, 0
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        for ep in range(epochs):
            perm = rng.permutation(n)
            tot_loss, nb = 0.0, 0
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                Xb, yb = Xtr[idx], ytr[idx]
                p, (Xc, z1, a1, dropout_mask) = self.forward(Xb, train=True, rng=rng)
                w = class_w[yb]
                loss = -(w * np.log(p[np.arange(len(yb)), yb] + 1e-12)).sum() / w.sum()
                # backward
                dp = p
                dp[np.arange(len(yb)), yb] -= 1.0
                dp = dp * (w / w.sum())[:, None] * len(yb)
                dW2 = a1.T @ dp / len(yb)
                db2 = dp.sum(axis=0) / len(yb)
                da1 = dp @ self.W2.T
                if dropout_mask is not None:
                    da1 = da1 * dropout_mask
                dz1 = da1 * (z1 > 0)
                dW1 = Xc.T @ dz1 / len(yb)
                db1 = dz1.sum(axis=0) / len(yb)
                # Adam update
                self.t += 1
                for j, g in enumerate([dW1, db1, dW2, db2]):
                    self.m[j] = b1_ * self.m[j] + (1 - b1_) * g
                    self.v[j] = b2_ * self.v[j] + (1 - b2_) * g * g
                    mh = self.m[j] / (1 - b1_ ** self.t)
                    vh = self.v[j] / (1 - b2_ ** self.t)
                    self.params()[j] -= self.lr * mh / (np.sqrt(vh) + eps)
                tot_loss += loss * len(yb)
                nb += len(yb)
            yp_va, _ = self.predict(Xva.astype(np.float64))
            f1 = macro_f1(yva, yp_va)
            history.append({"epoch": ep, "train_loss": float(tot_loss / nb),
                            "val_macro_f1": f1})
            if verbose and (ep % 5 == 0 or ep == epochs - 1):
                print(f"epoch {ep:3d}  loss={tot_loss / nb:.4f}  val_macro_f1={f1:.4f}")
            if f1 > best_f1 + 1e-4:
                best_f1 = f1
                best_params = [p.copy() for p in self.params()]
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    if verbose:
                        print(f"early stop at epoch {ep} (best val macro-F1={best_f1:.4f})")
                    break
        if best_params:
            self.W1, self.b1, self.W2, self.b2 = best_params
        return history, best_f1

    def save(self, path, in_dim):
        np.savez(path, W1=self.W1.astype(np.float32), b1=self.b1.astype(np.float32),
                 W2=self.W2.astype(np.float32), b2=self.b2.astype(np.float32),
                 in_dim=np.int64(in_dim), hidden=np.int64(self.W1.shape[1]),
                 n_classes=np.int64(N_CLASSES),
                 classes=np.array(CLASSES, dtype=object))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="cache_<kind>.npz from dataset.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hidden", type=int, default=HIDDEN_DIM)
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    z = np.load(args.cache, allow_pickle=True)
    X, y, split = z["X"], z["y"], z["split"]
    Xtr, ytr = X[split == 0], y[split == 0]
    Xva, yva = X[split == 1], y[split == 1]
    Xte, yte = X[split == 2], y[split == 2]
    print(f"train={len(Xtr)} val={len(Xva)} test={len(Xte)} dim={X.shape[1]}")
    if not len(Xtr) or not len(Xva):
        raise SystemExit("ERROR: training and validation splits must both contain clips. Rebuild the dataset/cache.")
    if len(np.unique(ytr)) < 2:
        raise SystemExit("ERROR: training split contains fewer than two classes; add verified data before training.")
    if np.any(~np.isfinite(X)):
        raise SystemExit("ERROR: feature cache contains NaN/Inf values; rebuild the cache.")

    model = MLPHead(X.shape[1], hidden=args.hidden)
    history, best_f1 = model.fit(Xtr, ytr, Xva, yva, epochs=args.epochs)
    model.save(os.path.join(args.out_dir, "weights.npz"), X.shape[1])
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    if len(Xte):
        yp_te, _ = model.predict(Xte.astype(np.float64))
        acc = float((yp_te == yte).mean())
        print(f"BEST val macro-F1={best_f1:.4f}   test acc={acc:.4f} (n={len(Xte)})")
    else:
        print(f"BEST val macro-F1={best_f1:.4f}   TEST NOT RUN (empty test split; do not treat this model as validated)")
    print("weights ->", os.path.join(args.out_dir, "weights.npz"))


if __name__ == "__main__":
    main()
