# Tennis Stroke Classification from Body Pose

Bachelor's thesis project (School of Electrical Engineering, University of Belgrade, 2026).

Binary classification of tennis **forehand vs. backhand** strokes from video, using only the player's body pose: **MediaPipe PoseLandmarker** keypoints + an **LSTM** classifier (TensorFlow/Keras). No ball or racket detection.

The focus of the thesis is not a single accuracy number, but a systematic study of **how well the model generalizes** when conditions change: a new player, a different camera angle, a different dominant hand.

## Pipeline

1. **Pose extraction.** MediaPipe PoseLandmarker (Tasks API, `VIDEO` mode) returns 33 landmarks per frame. Frames with no detection reuse the last detected pose.
2. **Normalization.** Landmarks are centered on the hip midpoint and scaled by torso length (shoulder midpoint to hip midpoint), which makes the input independent of where the player stands and how far away the camera is. Left/right orientation is deliberately preserved.
3. **Sequence preparation.** Each stroke clip (about 2 s) is linearly interpolated to a fixed length of **30 frames x 66 features** (x, y of 33 landmarks).
4. **Model.** `LSTM(64)` -> `Dropout(0.3)` -> `Dense(32, ReLU)` -> `Dropout(0.2)` -> `Dense(1, sigmoid)`. Early stopping, up to 100 epochs.
5. **Evaluation.** Leave-one-player-out cross-validation (LOPO), each experiment repeated 3 times with different random initialization.

## Experiments and results

| Phase | Question | Result |
|---|---|---|
| 1. Baseline | New player, same camera angle (5 right-handed players, 40 forehands + 40 backhands each) | **99.25%** mean accuracy (LOPO, 3 runs) |
| 2. Camera angle | Train on behind-the-player view, test at 45° and 90° | Drops to about chance level (~50%). The arm is often occluded by the player's own body from these angles, so the discriminative signal is lost. Mirroring and angle augmentation did not fully fix it |
| 3. Dominant hand | Train on right-handed players, test on left-handed players | Without mirroring the model systematically swaps the two classes (~0-2% accuracy). Mirroring the landmark coordinates (flip x, swap left/right landmark pairs) fixes this, with accuracy up to 100% |
| 4. Final check | Three previously unseen players of different ages and levels | Confirms the patterns from phases 2 and 3 outside controlled conditions |

**Main takeaway:** different kinds of distribution shift need qualitatively different fixes. A change of dominant hand is solved by a simple geometric transformation; a change of camera angle needs more data and a different pose representation (e.g. 3D pose).

Results are based on small test subsets (20 clips per subset), so single comparisons should be read with that in mind. That is why every experiment was repeated three times.

Plots: `training_accuracy.png`, `training_loss.png`, `velocity_plot.png`, `combined_velocity_plot.png`.

## Limitations

- Each clip contains exactly one stroke, and clips were cut manually. Automatic stroke segmentation from continuous video (wrist-velocity peak detection) was tried but was not reliable enough.
- Small, self-recorded dataset; all players in the base set are of the same gender, so the effect of gender was not examined.
- Only forehand vs. backhand; no serves or other strokes.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows (PowerShell)
pip install -r requirements.txt
```

Download the MediaPipe pose model and place it in the project root:

```
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## Data

The video dataset is **not** included in this repository (it contains recordings of real people). The expected structure is one folder per player with forehand and backhand clips, which are converted to landmark sequences by the dataset-building scripts.

## Repository overview

| File | Purpose |
|---|---|
| `extract_landmarks.py` | Extracts pose landmarks from video |
| `build_dataset.py`, `build_dataset_add_player.py`, `check_dataset.py` | Builds and checks the landmark dataset |
| `train_by_player.py`, `train_by_player_all.py`, `train_lstm_workshop.py` | LSTM training / leave-one-player-out |
| `pokretanje_faza1.py` ... `pokretanje_faza4.py` | Runs the experiments for phases 1-4 |
| `mirror_test.py` | Mirror transformation test |
| `test_new_angle.py`, `test_shuffled_frames.py` | Camera angle and frame order tests |
| `classify_video.py` | Classifies strokes in a video using a trained model |
| `final_model_run*.keras` | Final trained models |

## Tech

Python, TensorFlow/Keras, MediaPipe, OpenCV, NumPy, SciPy
