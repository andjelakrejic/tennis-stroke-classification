import numpy as np

X = np.load("../X.npy")
y = np.load("../y.npy")
players = np.load("../players.npy", allow_pickle=True)
video_paths = np.load("../video_paths.npy", allow_pickle=True)

print("PROVERA DATASETA")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("players shape:", players.shape)
print("video_paths shape:", video_paths.shape)

print("\nLabele:")
labels, counts = np.unique(y, return_counts=True)

for label, count in zip(labels, counts):
    name = "backhand" if label == 0 else "forehand"
    print(f"  {name}: {count}")

print("\nIgrači:")
unique_players, counts = np.unique(
    players,
    return_counts=True
)

for player, count in zip(unique_players, counts):
    print(f"  {player}: {count}")

print("\nVrednosti landmarka:")
print("  Min:", X.min())
print("  Max:", X.max())
print("  Mean:", X.mean())
print("  NaN:", np.isnan(X).sum())
print("  Inf:", np.isinf(X).sum())

print("\nProvera svakog videa:")

for i in range(len(X)):
    zero_ratio = np.mean(X[i] == 0)

    label_name = (
        "backhand"
        if y[i] == 0
        else "forehand"
    )
    print(
        f"{i:02d} | "
        f"{label_name:8s} | "
        f"zeros={zero_ratio:.3f} | "
        f"{video_paths[i]}"
    )

print("\nMaksimalna apsolutna vrednost po videu:")

for i in range(len(X)):
    max_abs = np.max(np.abs(X[i]))

    label_name = "backhand" if y[i] == 0 else "forehand"

    print(
        f"{i:02d} | "
        f"{label_name:8s} | "
        f"max_abs={max_abs:.2f} | "
        f"{video_paths[i]}"
    )