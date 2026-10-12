import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report

# Load the dataset and the model

X = np.load("X.npy")
y = np.load("y.npy")
players = np.load("players.npy", allow_pickle=True)

TEST_PLAYER = "igrac9_left"
MODEL_PATH = "phase3_model_run3.keras"

test_mask = players == TEST_PLAYER
X_test = X[test_mask]
y_test = y[test_mask]

model = load_model(MODEL_PATH) # .keras

# Ogledalo transformacija

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


# Uporedjuje se originalni video sa flip-ovanim

print("ORIGINAL (without mirror transformation)")
probs_orig = model.predict(X_test, verbose=0).flatten()
preds_orig = (probs_orig >= 0.5).astype(int)
acc_orig = (preds_orig == y_test).mean()

print(f"Test player: {TEST_PLAYER}")
print(f"Accuracy: {acc_orig:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, preds_orig))
print()
print(classification_report(y_test, preds_orig,
                             target_names=["backhand", "forehand"],
                             zero_division=0))

print()
print("MIRROR (after horizontal mirror transformation)")
X_mirrored = mirror_pose(X_test)
probs_mirror = model.predict(X_mirrored, verbose=0).flatten()
preds_mirror = (probs_mirror >= 0.5).astype(int)
acc_mirror = (preds_mirror == y_test).mean()

print(f"Accuracy: {acc_mirror:.4f}")
print("Confusion matrix:")
print(confusion_matrix(y_test, preds_mirror))
print()
print(classification_report(y_test, preds_mirror,
                             target_names=["backhand", "forehand"],
                             zero_division=0))

print()
print("INDIVIDUAL PREDICTIONS (mirror)")
for i in range(len(y_test)):
    true_name = "BACKHAND" if y_test[i] == 0 else "FOREHAND"
    pred_name = "BACKHAND" if preds_mirror[i] == 0 else "FOREHAND"
    conf = probs_mirror[i] if preds_mirror[i] == 1 else 1 - probs_mirror[i]
    print(f"{i:02d} | actual={true_name:8s} | "
          f"predicted={pred_name:8s} | confidence={conf:.3f}")