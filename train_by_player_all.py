import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ------------------------------------------------
# 1. UČITAVANJE PODATAKA
# ------------------------------------------------

X = np.load("X.npy")
y = np.load("y.npy")
players = np.load("players.npy", allow_pickle=True)

print("===================================")
print("UČITANI PODACI")
print("===================================")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("players shape:", players.shape)

print()
print("Igrači u datasetu:")
print(np.unique(players))

print()
print("Backhand:", np.sum(y == 0))
print("Forehand:", np.sum(y == 1))


# ------------------------------------------------
# 2. SVI IGRACI CE JEDNOM BITI TEST IGRAC
# ------------------------------------------------

TEST_PLAYERS = np.unique(players)

results = []


# ------------------------------------------------
# 3. LEAVE-ONE-PLAYER-OUT EVALUACIJA
# ------------------------------------------------

for TEST_PLAYER in TEST_PLAYERS:

    print()
    print()
    print("===================================")
    print(f"TEST PLAYER: {TEST_PLAYER}")
    print("===================================")

    # Sve osim test igrača ide u trening
    train_mask = players != TEST_PLAYER

    # Samo test igrač ide u test skup
    test_mask = players == TEST_PLAYER

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]


    # ------------------------------------------------
    # 4. PRIKAZ PODELE PODATAKA
    # ------------------------------------------------

    print()
    print("============================")
    print("PODELA PODATAKA")
    print("============================")

    print("Test igrač:", TEST_PLAYER)

    print("Train X:", X_train.shape)
    print("Train y:", y_train.shape)

    print("Test X:", X_test.shape)
    print("Test y:", y_test.shape)

    print()

    print(
        "Train backhand:",
        np.sum(y_train == 0)
    )

    print(
        "Train forehand:",
        np.sum(y_train == 1)
    )

    print(
        "Test backhand:",
        np.sum(y_test == 0)
    )

    print(
        "Test forehand:",
        np.sum(y_test == 1)
    )


    # ------------------------------------------------
    # 5. NOV MODEL ZA SVAKOG TEST IGRAČA
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


    # ------------------------------------------------
    # 6. EARLY STOPPING
    # ------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )


    # ------------------------------------------------
    # 7. TRENING
    # ------------------------------------------------

    model.fit(
        X_train,
        y_train,

        epochs=100,
        batch_size=4,

        validation_split=0.2,

        callbacks=[
            early_stopping
        ],

        verbose=1
    )


    # ------------------------------------------------
    # 8. EVALUACIJA NA POTPUNO NEVIĐENOM IGRAČU
    # ------------------------------------------------

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


    # ------------------------------------------------
    # 9. REZULTATI
    # ------------------------------------------------

    print()
    print("============================")
    print("REZULTAT NA NEVIĐENOM IGRAČU")
    print("============================")

    print(
        f"Test player: {TEST_PLAYER}"
    )

    print(
        f"Test loss: {loss:.4f}"
    )

    print(
        f"Test accuracy: {accuracy:.4f}"
    )


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
    # 10. POJEDINAČNE PREDIKCIJE
    # ------------------------------------------------

    print()
    print("Pojedinačne predikcije:")

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

        confidence = (
            1 - probabilities[i]
            if predictions[i] == 0
            else probabilities[i]
        )

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
            f"confidence={confidence:.3f}"
        )


    # ------------------------------------------------
    # 11. ČUVANJE MODELA
    # ------------------------------------------------

    model_name = (
        f"tennis_stroke_lstm_test_{TEST_PLAYER}.keras"
    )

    model.save(model_name)

    print()
    print(
        f"Model sačuvan kao {model_name}"
    )


    # ------------------------------------------------
    # 12. ČUVANJE REZULTATA OVOG IGRAČA
    # ------------------------------------------------

    correct = int(
        np.sum(
            predictions == y_test
        )
    )

    errors = int(
        np.sum(
            predictions != y_test
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
# 13. KONAČNI REZULTATI ZA SVE IGRAČE
# ------------------------------------------------

print()
print()
print("===================================")
print("KONAČNI REZULTATI")
print("===================================")

for result in results:

    print(
        f"{result['player']:8s} | "
        f"uzoraka={result['samples']:3d} | "
        f"tacno={result['correct']:3d} | "
        f"greske={result['errors']:2d} | "
        f"accuracy={result['accuracy']:.4f}"
    )


# ------------------------------------------------
# 14. PROSEČNA TAČNOST
# ------------------------------------------------

accuracies = np.array([
    result["accuracy"]
    for result in results
])


print()
print(
    f"Prosečna accuracy: "
    f"{np.mean(accuracies):.4f}"
)

print(
    f"Standardna devijacija: "
    f"{np.std(accuracies):.4f}"
)


# ------------------------------------------------
# 15. UKUPNA TAČNOST PREKO SVIH UZORAKA
# ------------------------------------------------

total_samples = sum(
    result["samples"]
    for result in results
)

total_correct = sum(
    result["correct"]
    for result in results
)

total_errors = sum(
    result["errors"]
    for result in results
)

overall_accuracy = (
    total_correct / total_samples
)


print()
print(
    f"Ukupno uzoraka: {total_samples}"
)

print(
    f"Ukupno tacno: {total_correct}"
)

print(
    f"Ukupno gresaka: {total_errors}"
)

print(
    f"Ukupna accuracy: {overall_accuracy:.4f}"
)
