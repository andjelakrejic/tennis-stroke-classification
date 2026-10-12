import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

LEFTY_TEST_SETS = ["igrac8_left", "igrac9_left"]
N_RUNS = 3

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

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


test_data = {}
for test_set in LEFTY_TEST_SETS:
    mask = players == test_set
    test_data[test_set] = (X[mask], y[mask])

results = {ts: [] for ts in test_data}

for run in range(1, N_RUNS + 1):
    model_path = f"phase3_model_run{run}.keras"
    print(f"=== Model: {model_path} ===")
    model = load_model(model_path)

    for test_set, (X_test, y_test) in test_data.items():
        X_mirrored = mirror_pose(X_test)
        probabilities = model.predict(X_mirrored, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[test_set].append((acc, cm))

        print(f"  {test_set:15s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()

print()
print("===================================")
print("MIRROR RESULTS SUMMARY (mean +/- std across", N_RUNS, "runs)")
print("===================================")
print()
print(f"{'Test set':15s} | {'Run1':>8s} {'Run2':>8s} {'Run3':>8s} | {'Mean':>8s} | {'Std':>6s}")
print("-" * 65)

for test_set in LEFTY_TEST_SETS:
    accs = [r[0] for r in results[test_set]]
    runs_str = " ".join(f"{a*100:7.2f}%" for a in accs)
    print(f"{test_set:15s} | {runs_str} | {np.mean(accs)*100:7.2f}% | {np.std(accs)*100:5.2f}%")