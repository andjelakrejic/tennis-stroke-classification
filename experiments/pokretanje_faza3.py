import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

PHASE1_PLAYERS = ["igrac1", "igrac2", "igrac3", "igrac4", "igrac5"]
LEFTY_TEST_SETS = ["igrac8_left", "igrac9_left"]

N_RUNS = 3

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

# Training set is fixed across all runs - all 5 base (right-handed) players,
# no lefty data included at all in this baseline (no-augmentation) test.
train_mask = np.isin(players, PHASE1_PLAYERS)
X_train = X[train_mask]
y_train = y[train_mask]

print("Training set (fixed across runs):", X_train.shape)
print()

test_data = {}
for test_set in LEFTY_TEST_SETS:
    mask = players == test_set
    if mask.sum() == 0:
        print(f"WARNING: '{test_set}' not found in dataset, skipping.")
        continue
    test_data[test_set] = (X[mask], y[mask])
    print(f"  {test_set}: {mask.sum()} clips")

print()


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


# results[test_set] = list of (accuracy, confusion_matrix) across runs
results = {ts: [] for ts in test_data}

for run in range(1, N_RUNS + 1):
    print("===================================")
    print(f"RUN {run}/{N_RUNS}")
    print("===================================")

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

    # Save model - needed later for the mirror transformation test
    model.save(f"phase3_model_run{run}.keras")

    for test_set, (X_test, y_test) in test_data.items():
        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[test_set].append((acc, cm))

        print(f"  {test_set:15s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()


# ------------------------------------------------
# SUMMARY TABLE
# ------------------------------------------------

print()
print("===================================")
print("SUMMARY (mean +/- std across", N_RUNS, "runs)")
print("===================================")
print()
print(f"{'Test set':15s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Mean':>8s} | {'Std':>6s}")
print("-" * 65)

for test_set in LEFTY_TEST_SETS:
    if test_set not in results:
        continue
    accs = [r[0] for r in results[test_set]]
    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    print(f"{test_set:15s} | {runs_str} | {np.mean(accs)*100:7.2f}% | {np.std(accs)*100:5.2f}%")

print()
print("Cumulative confusion matrices across all runs:")
for test_set in LEFTY_TEST_SETS:
    if test_set not in results:
        continue
    cms = [r[1] for r in results[test_set]]
    cumulative = sum(cms)
    print(f"{test_set:15s}: {cumulative.tolist()}")