import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

PHASE1_PLAYERS = ["igrac1", "igrac2", "igrac3", "igrac4", "igrac5"]
LEFTY_PLAYERS = ["igrac8_left", "igrac9_left"]

N_RUNS = 3

# Results from earlier experiments, for the final 3-way comparison
NO_AUGMENTATION_RESULTS = {"igrac8_left": 0.00, "igrac9_left": 0.00}
MIRROR_TRANSFORM_RESULTS = {"igrac8_left": 100.00, "igrac9_left": 100.00}
REAL_LEFTY_RESULTS = {"igrac8_left": 0.00, "igrac9_left": 0.00}

X = np.load("X.npy")
y = np.load("y.npy")
players = np.load("players.npy", allow_pickle=True)

SWAP_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10),
    (11, 12), (13, 14), (15, 16), (17, 18),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (27, 28), (29, 30), (31, 32),
]


def mirror_pose(X_in):
    X_out = X_in.copy()
    X_reshaped = X_out.reshape(X_out.shape[0], X_out.shape[1], 33, 2)
    X_reshaped[:, :, :, 0] = -X_reshaped[:, :, :, 0]
    X_swapped = X_reshaped.copy()
    for left_idx, right_idx in SWAP_PAIRS:
        X_swapped[:, :, left_idx, :] = X_reshaped[:, :, right_idx, :]
        X_swapped[:, :, right_idx, :] = X_reshaped[:, :, left_idx, :]
    return X_swapped.reshape(X_out.shape)


def build_model():
    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ------------------------------------------------
# BUILD AUGMENTED TRAINING SET (fixed across all runs and both test players)
# ------------------------------------------------

train_mask = np.isin(players, PHASE1_PLAYERS)
X_train_orig = X[train_mask]
y_train_orig = y[train_mask]

X_train_mirror = mirror_pose(X_train_orig)
y_train_mirror = y_train_orig.copy()

X_train_full = np.concatenate([X_train_orig, X_train_mirror], axis=0)
y_train_full = np.concatenate([y_train_orig, y_train_mirror], axis=0)

print("Original training data:", X_train_orig.shape)
print("Mirror-augmented training data:", X_train_full.shape)
print()

test_data = {}
for p in LEFTY_PLAYERS:
    mask = players == p
    test_data[p] = (X[mask], y[mask])


# ------------------------------------------------
# N_RUNS TRAININGS, EACH EVALUATED ON BOTH LEFTY TEST SETS
# ------------------------------------------------

results = {p: [] for p in LEFTY_PLAYERS}

for run in range(1, N_RUNS + 1):
    print(f"=== Run {run}/{N_RUNS} ===")

    model = build_model()
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train_full, y_train_full,
        epochs=100, batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    for p, (X_test, y_test) in test_data.items():
        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[p].append((acc, cm))

        print(f"  {p:15s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()


# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

print()
print("===================================")
print("SUMMARY: mirror-augmented training set")
print("===================================")
print()
print(f"{'Player':15s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Mean':>8s} | {'Std':>6s}")
print("-" * 65)

mirror_aug_means = {}
for p in LEFTY_PLAYERS:
    accs = [r[0] for r in results[p]]
    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    mean_acc = np.mean(accs) * 100
    mirror_aug_means[p] = mean_acc
    print(f"{p:15s} | {runs_str} | {mean_acc:7.2f}% | {np.std(accs)*100:5.2f}%")

# ------------------------------------------------
# 4-WAY COMPARISON
# ------------------------------------------------

print()
print("===================================")
print("4-WAY COMPARISON")
print("===================================")
print()
print(f"{'Player':15s} | {'No aug.':>9s} | {'Mirror transf.':>14s} | {'Real lefty':>10s} | {'Mirror aug.':>11s}")
print("-" * 75)

for p in LEFTY_PLAYERS:
    print(f"{p:15s} | {NO_AUGMENTATION_RESULTS[p]:8.2f}% | {MIRROR_TRANSFORM_RESULTS[p]:13.2f}% | "
          f"{REAL_LEFTY_RESULTS[p]:9.2f}% | {mirror_aug_means[p]:10.2f}%")