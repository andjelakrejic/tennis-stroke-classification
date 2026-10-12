import numpy as np
from sklearn.metrics import confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# PODESAVANJA
# ------------------------------------------------

PHASE1_PLAYERS = ["igrac1", "igrac2", "igrac3", "igrac4", "igrac5"]

ANGLE_TEST_SETS = [
    "igrac3_angle_fh"
    # "igrac1_45_bh", "igrac1_45_fh", "igrac1_angle_bh", "igrac1_angle_fh",
    # "igrac3_45_bh", "igrac3_45_fh", "igrac3_angle_bh", "igrac3_angle_fh",
    # "igrac5_45_bh", "igrac5_45_fh", "igrac5_angle_bh", "igrac5_angle_fh",
]

N_RUNS = 3

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

# Trening skup je FIKSAN kroz sve testove - svih 5 igraca iz faze 1
# (0 stepeni ugao), bez izostavljanja - jer testiramo generalizaciju
# na NOV UGAO za VEC VIDJENOG igraca, ne na novog igraca.
train_mask = np.isin(players, PHASE1_PLAYERS)
X_train = X[train_mask]
y_train = y[train_mask]

print("Trening skup (fiksan za sve runove):", X_train.shape)
print()

# Test podaci za svaki ugaoni podskup
test_data = {}
for test_set in ANGLE_TEST_SETS:
    mask = players == test_set
    if mask.sum() == 0:
        print(f"UPOZORENJE: '{test_set}' ne postoji u datasetu, preskacem.")
        continue
    test_data[test_set] = (X[mask], y[mask])
    print(f"  {test_set}: {mask.sum()} klipova")

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
# N_RUNS TRENINGA, SVAKI EVALUIRAN NA SVIM TEST SKUPOVIMA
# ------------------------------------------------

# results[test_set] = lista (accuracy, confusion_matrix) kroz runove
results = {test_set: [] for test_set in test_data}

for run in range(1, N_RUNS + 1):
    print("===================================")
    print(f"RUN {run}/{N_RUNS}")
    print("===================================")

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

    # Sacuvaj model ovog runa - koristan za kasniju mirror analizu
    model.save(f"phase2_model_run{run}.keras")

    for test_set, (X_test, y_test) in test_data.items():
        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[test_set].append((acc, cm))

        print(f"  {test_set:20s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()


# ------------------------------------------------
# ZBIRNA TABELA
# ------------------------------------------------

print()
print("===================================")
print("ZBIRNI REZULTATI (prosek +/- std kroz", N_RUNS, "runa)")
print("===================================")
print()
print(f"{'Test skup':20s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Prosek':>8s} | {'Std':>6s}")
print("-" * 70)

for test_set in ANGLE_TEST_SETS:
    if test_set not in results:
        continue
    accs = [r[0] for r in results[test_set]]
    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    print(f"{test_set:20s} | {runs_str} | {np.mean(accs)*100:7.2f}% | {np.std(accs)*100:5.2f}%")

print()
print("Kumulativne (sabrane) matrice konfuzije kroz sve runove, po test skupu:")
print()
for test_set in ANGLE_TEST_SETS:
    if test_set not in results:
        continue
    cms = [r[1] for r in results[test_set]]
    cumulative = sum(cms)
    print(f"{test_set:20s}: {cumulative.tolist()}")