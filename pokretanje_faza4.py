import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------

# Prefiksi imena igraca REZERVISANIH iskljucivo za zavrsnu proveru (4.4).
# Ovi igraci NIKAD ne smeju uci u trening, ni u ovoj ni u bilo kojoj
# ranijoj fazi.
RESERVED_PREFIXES = ("igrac6", "igrac7")

N_RUNS = 3

X = np.load("X.npy")
y = np.load("y.npy")
players = np.load("players.npy", allow_pickle=True)

is_reserved = np.array([
    any(p.startswith(prefix) for prefix in RESERVED_PREFIXES)
    for p in players
])

train_mask = ~is_reserved
X_train = X[train_mask]
y_train = y[train_mask]

print("Finalni trening skup (fiksan za sve runove):", X_train.shape)
print("Igraci ukljuceni u trening:", sorted(np.unique(players[train_mask]).tolist()))
print()

test_sets = sorted(np.unique(players[is_reserved]).tolist())
print("Rezervisani test podskupovi (4.4):", test_sets)
print()

test_data = {}
for ts in test_sets:
    mask = players == ts
    test_data[ts] = (X[mask], y[mask])
    print(f"  {ts}: {mask.sum()} klipova "
          f"({np.sum(y[mask]==0)} bh, {np.sum(y[mask]==1)} fh)")

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


results = {ts: [] for ts in test_sets}

for run in range(1, N_RUNS + 1):
    print(f"=== Run {run}/{N_RUNS} ===")

    model = build_model()
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        epochs=100, batch_size=8,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )

    model.save(f"final_model_run{run}.keras")

    for ts, (X_test, y_test) in test_data.items():
        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[ts].append((acc, cm))

        print(f"  {ts:20s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()


# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

print()
print("===================================")
print("ZAVRSNA PROVERA (4.4) - prosek +/- std kroz", N_RUNS, "runa")
print("===================================")
print()
print(f"{'Test podskup':20s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Prosek':>8s} | {'Std':>6s}")
print("-" * 70)

for ts in test_sets:
    accs = [r[0] for r in results[ts]]
    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    print(f"{ts:20s} | {runs_str} | {np.mean(accs)*100:7.2f}% | {np.std(accs)*100:5.2f}%")

print()
print("Kumulativne matrice konfuzije kroz sve runove:")
for ts in test_sets:
    cms = [r[1] for r in results[ts]]
    cumulative = sum(cms)
    print(f"{ts:20s}: {cumulative.tolist()}")

# Ukupna (agregatna) tacnost kroz SVE rezervisane podatke zajedno
print()
all_correct = 0
all_total = 0
for ts in test_sets:
    cms = [r[1] for r in results[ts]]
    cumulative = sum(cms)
    correct = cumulative[0][0] + cumulative[1][1]
    total = cumulative.sum()
    all_correct += correct
    all_total += total
print(f"UKUPNA agregatna tacnost kroz sve rezervisane podatke i sve runove: "
      f"{all_correct}/{all_total} = {all_correct/all_total*100:.2f}%")