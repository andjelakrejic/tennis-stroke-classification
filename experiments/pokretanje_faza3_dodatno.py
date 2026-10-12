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

# Baseline (no lefty in training) and mirror results, from earlier runs,
# used for the final 3-way comparison table
NO_AUGMENTATION_RESULTS = {"igrac8_left": 0.00, "igrac9_left": 0.00}
MIRROR_RESULTS = {"igrac8_left": 100.00, "igrac9_left": 100.00}

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)


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


# results[held_out_lefty] = list of (accuracy, confusion_matrix) across runs
results = {p: [] for p in LEFTY_PLAYERS}

for held_out in LEFTY_PLAYERS:

    other_lefty = [p for p in LEFTY_PLAYERS if p != held_out][0]

    train_mask = np.isin(players, PHASE1_PLAYERS) | (players == other_lefty)
    test_mask = players == held_out

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print("===================================")
    print(f"HELD OUT (test): {held_out}")
    print(f"Added to training: {other_lefty}")
    print(f"Training set: {X_train.shape}")
    print("===================================")

    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}...")

        model = build_model()
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            epochs=100, batch_size=4,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[held_out].append((acc, cm))

        print(f"    accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()


# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

print()
print("===================================")
print("SUMMARY: training with one real lefty, testing on the other")
print("===================================")
print()
print(f"{'Held out':15s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Mean':>8s} | {'Std':>6s}")
print("-" * 65)

real_lefty_means = {}
for held_out in LEFTY_PLAYERS:
    accs = [r[0] for r in results[held_out]]
    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    mean_acc = np.mean(accs) * 100
    real_lefty_means[held_out] = mean_acc
    print(f"{held_out:15s} | {runs_str} | {mean_acc:7.2f}% | {np.std(accs)*100:5.2f}%")

print()
print("Cumulative confusion matrices:")
for held_out in LEFTY_PLAYERS:
    cms = [r[1] for r in results[held_out]]
    cumulative = sum(cms)
    print(f"{held_out:15s}: {cumulative.tolist()}")

# ------------------------------------------------
# 3-WAY COMPARISON
# ------------------------------------------------

print()
print("===================================")
print("3-WAY COMPARISON: no augmentation vs mirror vs real second lefty")
print("===================================")
print()
print(f"{'Player':15s} | {'No aug.':>10s} | {'Mirror':>10s} | {'Real lefty':>10s}")
print("-" * 60)

for p in LEFTY_PLAYERS:
    print(f"{p:15s} | {NO_AUGMENTATION_RESULTS[p]:9.2f}% | {MIRROR_RESULTS[p]:9.2f}% | {real_lefty_means[p]:9.2f}%")