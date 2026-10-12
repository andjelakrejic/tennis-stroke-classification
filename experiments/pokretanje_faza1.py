import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# PODESAVANJA
# ------------------------------------------------

PHASE1_PLAYERS = ["igrac1", "igrac2", "igrac3", "igrac4", "igrac5"]
N_RUNS = 3

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

# Faza 1 koristi samo osnovnih 5 igraca u treningu/testu -
# svi ostali (uglovi, levoruki, rezerva za 4.4) se iskljucuju
# iz cele ove analize, da ne bi curili u trening.
phase1_mask = np.isin(players, PHASE1_PLAYERS)
X = X[phase1_mask]
y = y[phase1_mask]
players = players[phase1_mask]

print("Faza 1 - ukupno klipova:", len(X))
for p in PHASE1_PLAYERS:
    print(f"  {p}: {np.sum(players == p)}")
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


# ------------------------------------------------
# LOPO x N_RUNS ZA SVAKOG IGRACA
# ------------------------------------------------

results = {}  # player -> list of (accuracy, confusion_matrix)

for test_player in PHASE1_PLAYERS:

    train_mask = players != test_player
    test_mask = players == test_player

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print("===================================")
    print(f"TEST PLAYER: {test_player}")
    print("===================================")

    run_accuracies = []
    run_confusions = []

    for run in range(1, N_RUNS + 1):
        print(f"  Run {run}/{N_RUNS}...")

        model = build_model()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=4,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        run_accuracies.append(acc)
        run_confusions.append(cm)

        print(f"    accuracy = {acc:.4f}")
        print(f"    confusion matrix = {cm.tolist()}")

    results[test_player] = (run_accuracies, run_confusions)
    print()


# ------------------------------------------------
# ZBIRNA TABELA - SPREMNA ZA RAD
# ------------------------------------------------

print()
print("===================================")
print("ZBIRNI REZULTATI (prosek +/- std kroz", N_RUNS, "runa)")
print("===================================")
print()
print(f"{'Igrac':10s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Prosek':>8s} | {'Std':>6s}")
print("-" * 60)

all_means = []

for player in PHASE1_PLAYERS:
    accs, cms = results[player]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    all_means.append(mean_acc)

    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    print(f"{player:10s} | {runs_str} | {mean_acc*100:7.2f}% | {std_acc*100:5.2f}%")

print("-" * 60)
print(f"{'PROSEK SVIH':10s} | {'':26s} | {np.mean(all_means)*100:7.2f}% | {np.std(all_means)*100:5.2f}%")

print()
print("Napomena: ako je std kolona blizu 0 za sve igrace, to potvrdjuje")
print("da je rezultat faze 1 stabilan i ne zavisi znacajno od slucajne")
print("inicijalizacije tezina modela.")