import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)

print("Players in the dataset:")
print(np.unique(players))


TEST_PLAYER = "igrac6_angle_bh"
#
# # train_mask = players != TEST_PLAYER
# # test_mask = players == TEST_PLAYER
#
# EXCLUDE_FROM_TRAIN =    ["igrac1_45_bh","igrac1_45_fh","igrac5_45_bh", "igrac5_45_fh",
#                         "igrac1_angle_bh","igrac1_angle_fh","igrac5_angle_bh", "igrac5_angle_fh",
#                         "igrac3_45_bh","igrac3_45_fh","igrac3_angle_bh", "igrac3_angle_fh",
#                         "igrac8_left", "igrac6_angle_bh", "igrac6_angle_fh", "igrac7_angle_fh", "igrac6_45_bh", "igrac7_angle_bh",
#                          "igrac9_left", "random_test", "final_test_2", "final_test_1", "random_test_2"]

EXCLUDE_FROM_TRAIN = ["igrac2","igrac4","igrac5","igrac9_left", "igrac8_left", "final_test_2", "final_test_1", "random_test_2", "random_test",
                      "igrac6_45_bh", "igrac6_angle_bh", "igrac6_angle_fh", "igrac7_angle_fh", "igrac7_angle_bh",
                      ]
train_mask = ~np.isin(players, EXCLUDE_FROM_TRAIN)
test_mask = players == TEST_PLAYER

# ucitavanje podataka
X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]


print()
print("DATA SPLIT")

print("Test player:", TEST_PLAYER)

print("Train X:", X_train.shape)
print("Train y:", y_train.shape)

print("Test X:", X_test.shape)
print("Test y:", y_test.shape)

print()

print("Train backhand:", np.sum(y_train == 0))
print("Train forehand:", np.sum(y_train == 1))
print("Test backhand:", np.sum(y_test == 0))
print("Test forehand:", np.sum(y_test == 1))


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
    callbacks=[early_stopping],
    verbose=1
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


print()
print("RESULT ON THE UNSEEN PLAYER")

print(f"Test player: {TEST_PLAYER}")

print(f"Test loss: {loss:.4f}")

print(f"Test accuracy: {accuracy:.4f}")

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

print()
print("Individual predictions:")

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

    print(
        f"{i:02d} | "
        f"actual={true_name:8s} | "
        f"predicted={predicted_name:8s} | "
        f"confidence={confidence:.3f}"
    )
model.save(
    f"tennis_stroke_lstm_test_{TEST_PLAYER}.keras"
)

print()
print(f"Model saved as tennis_stroke_lstm_test_{TEST_PLAYER}.keras")




# provera koji igraci su iskorisceni za data set
training_players = np.unique(players[train_mask])
sidecar_path = f"tennis_stroke_lstm_test_{TEST_PLAYER}_train_players.npy"
np.save(sidecar_path, training_players)

print(f"Training-players record saved as {sidecar_path}")
print("Players actually included in training:", training_players)