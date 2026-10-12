import numpy as np

from sklearn.model_selection import train_test_split


# ------------------------------------------------
# 1. UČITAVANJE PODATAKA
# ------------------------------------------------

X = np.load("X.npy")
y = np.load("y.npy")
players = np.load("players.npy", allow_pickle=True)
video_paths = np.load("video_paths.npy", allow_pickle=True)


print("===================================")
print("UČITANI PODACI")
print("===================================")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("players shape:", players.shape)
print("video_paths shape:", video_paths.shape)

print()
print("Igrači:", np.unique(players))

print(
    "Backhand:",
    np.sum(y == 0)
)

print(
    "Forehand:",
    np.sum(y == 1)
)


# ------------------------------------------------
# 2. TRAIN / VALIDATION / TEST PODELA
# ------------------------------------------------

# Za ovu prvu workshop-style verziju pravimo random split.
#
# Kasnije cemo napraviti ozbiljniju varijantu gde je
# TEST ceo neviđeni igrač.

X_train_val, X_test, y_train_val, y_test, paths_train_val, paths_test = (
    train_test_split(
        X,
        y,
        video_paths,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


# Od preostalih 80% izdvajamo validation skup.
#
# 0.25 od 80% = 20% ukupnog dataseta.
#
# Konačno približno:
#
# TRAIN      60%
# VALIDATION 20%
# TEST       20%

X_train, X_val, y_train, y_val, paths_train, paths_val = (
    train_test_split(
        X_train_val,
        y_train_val,
        paths_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val
    )
)


print()
print("===================================")
print("PODELA PODATAKA")
print("===================================")

print("Train:")
print("  X:", X_train.shape)
print("  y:", y_train.shape)

print()

print("Validation:")
print("  X:", X_val.shape)
print("  y:", y_val.shape)

print()

print("Test:")
print("  X:", X_test.shape)
print("  y:", y_test.shape)


print()
print("Raspodela klasa:")

print(
    "Train   - BH:",
    np.sum(y_train == 0),
    "| FH:",
    np.sum(y_train == 1)
)

print(
    "Val     - BH:",
    np.sum(y_val == 0),
    "| FH:",
    np.sum(y_val == 1)
)

print(
    "Test    - BH:",
    np.sum(y_test == 0),
    "| FH:",
    np.sum(y_test == 1)
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ------------------------------------------------
# 3. MODEL
# ------------------------------------------------

model = Sequential([
    LSTM(
        64,
        input_shape=(X.shape[1], X.shape[2])
    ),

    Dropout(0.3),

    Dense(
        32,
        activation="relu"
    ),

    Dropout(0.2),

    Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


print()
print("===================================")
print("MODEL")
print("===================================")

model.summary()


# ------------------------------------------------
# 4. CALLBACKS
# ------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)


# ------------------------------------------------
# 5. TRENING
# ------------------------------------------------

history = model.fit(
    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=100,
    batch_size=4,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)

import matplotlib.pyplot as plt


# ------------------------------------------------
# 6. GRAFICI TRENINGA
# ------------------------------------------------

# Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Train accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation accuracy"
)

plt.xlabel("Epoha")
plt.ylabel("Accuracy")
plt.title("Tačnost tokom treninga")
plt.legend()
plt.tight_layout()

plt.savefig("results/training_accuracy.png")
plt.show()


# Loss
plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Train loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation loss"
)

plt.xlabel("Epoha")
plt.ylabel("Loss")
plt.title("Loss tokom treninga")
plt.legend()
plt.tight_layout()

plt.savefig("results/training_loss.png")
plt.show()

from sklearn.metrics import confusion_matrix, classification_report


# ------------------------------------------------
# 7. EVALUACIJA NA TEST SKUPU
# ------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("===================================")
print("TEST REZULTATI")
print("===================================")

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")


# ------------------------------------------------
# 8. PREDIKCIJE
# ------------------------------------------------

probabilities = model.predict(
    X_test,
    verbose=0
).flatten()

predictions = (
    probabilities >= 0.5
).astype(int)


print()
print("Confusion matrix:")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)


print()
print("Classification report:")

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "backhand",
            "forehand"
        ]
    )
)


# ------------------------------------------------
# 9. POJEDINAČNE PREDIKCIJE
# ------------------------------------------------

print()
print("===================================")
print("POJEDINAČNE PREDIKCIJE")
print("===================================")

for i in range(len(y_test)):

    true_name = (
        "BACKHAND"
        if y_test[i] == 0
        else "FOREHAND"
    )

    predicted_name = (
        "BACKHAND"
        if predictions[i] == 0
        else "FOREHAND"
    )

    if predictions[i] == 1:
        confidence = probabilities[i]
    else:
        confidence = 1 - probabilities[i]

    status = (
        "OK"
        if predictions[i] == y_test[i]
        else "GRESKA"
    )

    print(
        f"{i:02d} | "
        f"{status:6s} | "
        f"stvarno={true_name:8s} | "
        f"predikcija={predicted_name:8s} | "
        f"confidence={confidence:.3f} | "
        f"{paths_test[i]}"
    )


# ------------------------------------------------
# 10. SAMO POGRESNE PREDIKCIJE
# ------------------------------------------------

print()
print("===================================")
print("POGRESNO KLASIFIKOVANI KLIPOVI")

error_count = 0

for i in range(len(y_test)):

    if predictions[i] != y_test[i]:

        true_name = (
            "BACKHAND"
            if y_test[i] == 0
            else "FOREHAND"
        )

        predicted_name = (
            "BACKHAND"
            if predictions[i] == 0
            else "FOREHAND"
        )

        if predictions[i] == 1:
            confidence = probabilities[i]
        else:
            confidence = 1 - probabilities[i]

        print()
        print(f"Video: {paths_test[i]}")
        print(f"Stvarno: {true_name}")
        print(f"Predikcija: {predicted_name}")
        print(f"Confidence: {confidence:.3f}")

        error_count += 1


if error_count == 0:
    print("Nema pogresnih predikcija.")
else:
    print()
    print(
        f"Ukupan broj pogresnih predikcija: "
        f"{error_count}/{len(y_test)}"
    )


# ------------------------------------------------
# 11. ČUVANJE MODELA
# ------------------------------------------------

model.save(
    "tennis_stroke_lstm_workshop.keras"
)

print()
print(
    "Model sačuvan kao "
    "tennis_stroke_lstm_workshop.keras"
)