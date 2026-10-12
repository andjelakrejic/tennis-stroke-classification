import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")
players = np.load("players.npy", allow_pickle=True)
video_paths = np.load("video_paths.npy", allow_pickle=True)

print("Before cleanup:", len(X), "clips total")

zoomed_players = ["igrac6_45_bh"]
print("Removing these stale player entries:", zoomed_players)

keep_mask = ~np.isin(players, zoomed_players)

X_clean = X[keep_mask]
y_clean = y[keep_mask]
players_clean = players[keep_mask]
video_paths_clean = video_paths[keep_mask]

np.save("X.npy", X_clean)
np.save("y.npy", y_clean)
np.save("players.npy", players_clean)
np.save("video_paths.npy", video_paths_clean)

print()
print("After cleanup:", len(X_clean), "clips total")
print("Removed:", len(X) - len(X_clean), "clips")
print()
print("Remaining players:", sorted(np.unique(players_clean).tolist()))