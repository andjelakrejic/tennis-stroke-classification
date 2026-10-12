import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model

ZOOMED_TEST_SETS = [
    "igrac3_45_bh_zoomed",
    "igrac3_45_fh_zoomed",
    "igrac3_angle_bh_zoomed",
    "igrac3_angle_fh_zoomed",
]

N_RUNS = 3

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

# Originalni (ne-zumirani) rezultati iz pokretanje_faza2.py, za poredjenje
ORIGINAL_RESULTS = {
    "igrac3_45_bh": 80.00,
    "igrac3_45_fh": 58.33,
    "igrac3_angle_bh": 43.33,
    "igrac3_angle_fh": 50.00,
}

test_data = {}
for test_set in ZOOMED_TEST_SETS:
    mask = players == test_set
    if mask.sum() == 0:
        print(f"UPOZORENJE: '{test_set}' ne postoji u datasetu - da li si pokrenula build_dataset_add_player.py?")
        continue
    test_data[test_set] = (X[mask], y[mask])
    print(f"{test_set}: {mask.sum()} klipova")

print()

results = {ts: [] for ts in test_data}

for run in range(1, N_RUNS + 1):
    model_path = f"phase2_model_run{run}.keras"
    print(f"=== Model: {model_path} ===")
    model = load_model(model_path)

    for test_set, (X_test, y_test) in test_data.items():
        probabilities = model.predict(X_test, verbose=0).flatten()
        predictions = (probabilities >= 0.5).astype(int)

        acc = (predictions == y_test).mean()
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])

        results[test_set].append((acc, cm))

        print(f"  {test_set:30s} accuracy = {acc:.4f}  confusion = {cm.tolist()}")

    print()

print()
print("===================================")
print("POREDJENJE: ORIGINAL vs ZOOMED (prosek kroz 3 runa)")
print("===================================")
print()
print(f"{'Test skup (original)':20s} | {'Original':>10s} | {'Zoomed':>10s} | {'Razlika':>10s}")
print("-" * 65)

for zoomed_name, original_acc in ORIGINAL_RESULTS.items():
    zoomed_key = zoomed_name + "_zoomed"
    if zoomed_key not in results:
        continue
    accs = [r[0] for r in results[zoomed_key]]
    zoomed_mean = np.mean(accs) * 100
    diff = zoomed_mean - original_acc
    print(f"{zoomed_name:20s} | {original_acc:9.2f}% | {zoomed_mean:9.2f}% | {diff:+9.2f}%")