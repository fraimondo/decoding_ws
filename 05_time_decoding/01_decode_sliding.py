# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import SlidingEstimator, cross_val_multiscore
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
epochs = mne.read_epochs("../data/sample-epo.fif", preload=True)
epochs = epochs.pick(picks="eeg", exclude="bads")

# %%
# We will train the classifier on left vs right auditory trials on EEG
to_predict = epochs[["auditory/left", "auditory/right"]]
X = to_predict.get_data(copy=False)
y = to_predict.events[:, 2]  # target: auditory left vs auditory right

clf = make_pipeline(
    StandardScaler(), LogisticRegression(solver="liblinear", random_state=41)
)

time_decod = SlidingEstimator(
    clf, n_jobs=None, scoring="roc_auc", verbose=True
)
# here we use cv=3 just for speed
scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# %%›
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AUC ROC")  # Area Under the Curve
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Sensor space decoding (left vs right)")

# %%
to_predict = epochs[["auditory/left", "visual/left"]]
X = to_predict.get_data(copy=False)
y = to_predict.events[:, 2]  # target: auditory left vs visual left

clf = make_pipeline(
    StandardScaler(), LogisticRegression(solver="liblinear", random_state=41)
)

time_decod = SlidingEstimator(
    clf, n_jobs=None, scoring="roc_auc", verbose=True
)
# here we use cv=3 just for speed
scores = cross_val_multiscore(time_decod, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# %%
fig, ax = plt.subplots()
ax.plot(epochs.times, scores, label="score")
ax.axhline(0.5, color="k", linestyle="--", label="chance")
ax.set_xlabel("Times")
ax.set_ylabel("AUC ROC")  # Area Under the Curve
ax.legend()
ax.axvline(0.0, color="k", linestyle="-")
ax.set_title("Sensor space decoding (auditory vs visual)")

# %%
