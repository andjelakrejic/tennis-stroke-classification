import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------
# 1. UČITAVANJE PODATAKA
# ------------------------------------------------

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

print("X:", X.shape)
print("y:", y.shape)
print("players:", players.shape)


# ------------------------------------------------
# 2. SHUFFLE FREJMOVA
# ------------------------------------------------

rng = np.random.default_rng(42)

X_shuffled = X.copy()

for i in range(len(X_shuffled)):
    order = rng.permutation(X_shuffled.shape[1])
    X_shuffled[i] = X_shuffled[i, order]


# ------------------------------------------------
# 3. LEAVE-ONE-PLAYER-OUT
# ------------------------------------------------

results = []

for TEST_PLAYER in np.unique(players):

    print()
    print("===================================")
    print(f"TEST PLAYER: {TEST_PLAYER}")
    print("===================================")

    train_mask = players != TEST_PLAYER
    test_mask = players == TEST_PLAYER

    X_train = X_shuffled[train_mask]
    y_train = y[train_mask]

    X_test = X_shuffled[test_mask]
    y_test = y[test_mask]


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


    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )


    model.fit(
        X_train,
        y_train,

        epochs=100,
        batch_size=4,

        validation_split=0.2,

        callbacks=[
            early_stopping
        ],

        verbose=0
    )


    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )


    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()


    predictions = (
        probabilities >= 0.5
    ).astype(int)


    correct = int(
        np.sum(predictions == y_test)
    )

    errors = int(
        np.sum(predictions != y_test)
    )


    print(f"Accuracy: {accuracy:.4f}")
    print(f"Tacno: {correct}/{len(y_test)}")
    print(f"Greske: {errors}")

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


    results.append({
        "player": TEST_PLAYER,
        "samples": len(y_test),
        "correct": correct,
        "errors": errors,
        "accuracy": accuracy
    })


# ------------------------------------------------
# 4. KONAČNI REZULTATI
# ------------------------------------------------

print()
print("===================================")
print("SHUFFLED FRAMES - KONAČNI REZULTATI")
print("===================================")

for result in results:

    print(
        f"{result['player']:8s} | "
        f"uzoraka={result['samples']:3d} | "
        f"tacno={result['correct']:3d} | "
        f"greske={result['errors']:2d} | "
        f"accuracy={result['accuracy']:.4f}"
    )


accuracies = np.array([
    result["accuracy"]
    for result in results
])

total_samples = sum(
    result["samples"]
    for result in results
)

total_correct = sum(
    result["correct"]
    for result in results
)


print()
print(
    f"Prosečna accuracy: "
    f"{np.mean(accuracies):.4f}"
)

print(
    f"Standardna devijacija: "
    f"{np.std(accuracies):.4f}"
)

print(
    f"Ukupna accuracy: "
    f"{total_correct / total_samples:.4f}"
)