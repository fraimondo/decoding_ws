# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
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
# define the Temporal generalization object
time_gen = GeneralizingEstimator(
    clf, n_jobs=None, scoring="roc_auc", verbose=True
)

# again, cv=3 just for speed
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=None)

# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

# %%
fig, ax = plt.subplots(1, 1)
im = ax.imshow(
    scores,
    interpolation="lanczos",
    origin="lower",
    cmap="RdBu_r",
    extent=epochs.times[[0, -1, 0, -1]],
    vmin=0.0,
    vmax=1.0,
)
ax.set_xlabel("Testing Time (s)")
ax.set_ylabel("Training Time (s)")
ax.set_title("Temporal generalization")
ax.axvline(0, color="k")
ax.axhline(0, color="k")
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("AUC")
# %%
