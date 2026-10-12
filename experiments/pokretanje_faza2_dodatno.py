import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# PODESAVANJA
# ------------------------------------------------

PHASE1_PLAYERS = ["igrac1", "igrac2", "igrac3", "igrac4", "igrac5"]

ANGLE_SETS = {
    "igrac1": ["igrac1_45_bh", "igrac1_45_fh", "igrac1_angle_bh", "igrac1_angle_fh"],
    "igrac3": ["igrac3_45_bh", "igrac3_45_fh", "igrac3_angle_bh", "igrac3_angle_fh"],
    "igrac5": ["igrac5_45_bh", "igrac5_45_fh", "igrac5_angle_bh", "igrac5_angle_fh"],
}

ANGLE_PLAYERS = list(ANGLE_SETS.keys())  # ["igrac1", "igrac3", "igrac5"]

N_RUNS = 3

# Rezultati BEZ augmentacije, iz prethodnog pokretanja (pokretanje_faza2.py)
# - prosek kroz 3 runa, za poredjenje na kraju
BASELINE_NO_AUGMENTATION = {
    "igrac1_45_bh": 50.00, "igrac1_45_fh": 51.67,
    "igrac1_angle_bh": 50.00, "igrac1_angle_fh": 50.00,
    "igrac3_45_bh": 50.00, "igrac3_45_fh": 56.67,
    "igrac3_angle_bh": 50.00, "igrac3_angle_fh": 45.00,
    "igrac5_45_bh": 55.00, "igrac5_45_fh": 51.67,
    "igrac5_angle_bh": 50.00, "igrac5_angle_fh": 50.00,
}

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


# results[held_out_player][test_set] = lista (accuracy, confusion_matrix) kroz runove
results = {p: {ts: [] for ts in ANGLE_SETS[p]} for p in ANGLE_PLAYERS}

for held_out in ANGLE_PLAYERS:

    other_players = [p for p in ANGLE_PLAYERS if p != held_out]
    augment_sets = ANGLE_SETS[other_players[0]] + ANGLE_SETS[other_players[1]]
    test_sets = ANGLE_SETS[held_out]

    train_mask = np.isin(players, PHASE1_PLAYERS) | np.isin(players, augment_sets)

    X_train = X[train_mask]
    y_train = y[train_mask]

    print("===================================")
    print(f"IZOSTAVLJEN: {held_out}")
    print(f"Dodati u trening (ugaoni podaci): {other_players}")
    print(f"Trening skup: {X_train.shape}")
    print(f"Test skupovi: {test_sets}")
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

        for test_set in test_sets:
            mask = players == test_set
            X_test = X[mask]
            y_test = y[mask]

            probabilities = model.predict(X_test, verbose=0).flatten()
            predictions = (probabilities >= 0.5).astype(int)

            acc = (predictions == y_test).mean()
            cm = confusion_matrix(y_test, predictions, labels=[0, 1])

            results[held_out][test_set].append((acc, cm))

            print(f"    {test_set:20s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()


# ------------------------------------------------
# ZBIRNA TABELA - SA AUGMENTACIJOM
# ------------------------------------------------

print()
print("===================================")
print("ZBIRNI REZULTATI SA AUGMENTACIJOM (prosek +/- std kroz", N_RUNS, "runa)")
print("===================================")
print()
print(f"{'Test skup':20s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Prosek':>8s} | {'Std':>6s}")
print("-" * 70)

augmented_means = {}

for held_out in ANGLE_PLAYERS:
    for test_set in ANGLE_SETS[held_out]:
        accs = [r[0] for r in results[held_out][test_set]]
        runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
        mean_acc = np.mean(accs) * 100
        augmented_means[test_set] = mean_acc
        print(f"{test_set:20s} | {runs_str} | {mean_acc:7.2f}% | {np.std(accs)*100:5.2f}%")

# ------------------------------------------------
# POREDJENJE: BEZ AUGMENTACIJE vs SA AUGMENTACIJOM
# ------------------------------------------------

print()
print("===================================")
print("POREDJENJE: BEZ vs SA AUGMENTACIJOM")
print("===================================")
print()
print(f"{'Test skup':20s} | {'Bez aug.':>10s} | {'Sa aug.':>10s} | {'Razlika':>10s}")
print("-" * 60)

for test_set, no_aug_acc in BASELINE_NO_AUGMENTATION.items():
    with_aug_acc = augmented_means.get(test_set, float('nan'))
    diff = with_aug_acc - no_aug_acc
    print(f"{test_set:20s} | {no_aug_acc:9.2f}% | {with_aug_acc:9.2f}% | {diff:+9.2f}%")