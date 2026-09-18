# %%
import matplotlib.pyplot as plt
import mne
import numpy as np
from mne.decoding import (
    Scaler,
    Vectorizer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# %%
epochs = mne.read_epochs("../data/sample-epo.fif", preload=True)
epochs = epochs.pick(picks="eeg", exclude="bads")
# %%

# Choose our target and data
times = [0.1]  # 100ms
time_idx = epochs.time_as_index(times)
to_predict = epochs[["auditory/left", "visual/left"]]

# Get the data at 100ms
X = to_predict.get_data()[:, :, time_idx].squeeze()
y = to_predict.events[:, 2]  # target: auditory left vs visual left

clf = make_pipeline(
    StandardScaler(),
    LogisticRegression(solver="liblinear", random_state=31),
)


scores = cross_validate(
    clf, X, y, cv=5, n_jobs=None, scoring="roc_auc", return_train_score=True
)

# Mean scores across cross-validation splits
test_score = np.mean(scores["test_score"], axis=0)
train_score = np.mean(scores["train_score"], axis=0)
print(
    f"Decoding performance at 100ms: {100 * test_score:0.1f}% "
    f"(train: {100 * train_score:0.1f}%)"
)
print("Number of features used:", X.shape[1])
# %%
# Now we can repeat the same procedure but using the windows between 50ms and
# 150 ms.
to_predict = epochs[["auditory/left", "visual/left"]]
X = to_predict.get_data(picks="eeg", exclude="bads", tmin=0.05, tmax=0.15)
y = to_predict.events[:, 2]  # target: auditory left vs visual left

clf = make_pipeline(
    Scaler(to_predict.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear", random_state=31),
)

scores = cross_validate(
    clf, X, y, cv=5, n_jobs=None, scoring="roc_auc", return_train_score=True
)

# Mean scores across cross-validation splits
test_score = np.mean(scores["test_score"], axis=0)
train_score = np.mean(scores["train_score"], axis=0)
print(
    f"Decoding performance at [50ms-150ms]: {100 * test_score:0.1f}% "
    f"(train: {100 * train_score:0.1f}%)"
)
print("Number of features used:", X.shape[1] * X.shape[2])

# %%
# Now we can repeat the same procedure but using the windows between 50ms and
# 150 ms.
X = to_predict.get_data(picks="eeg", exclude="bads", tmin=0.05, tmax=0.3)
y = to_predict.events[:, 2]  # target: auditory left vs visual left

clf = make_pipeline(
    Scaler(to_predict.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear", random_state=31),
)

scores = cross_validate(
    clf, X, y, cv=5, n_jobs=None, scoring="roc_auc", return_train_score=True
)

# Mean scores across cross-validation splits
test_score = np.mean(scores["test_score"], axis=0)
train_score = np.mean(scores["train_score"], axis=0)
print(
    f"Decoding performance at [50ms-300ms]: {100 * test_score:0.1f}% "
    f"(train: {100 * train_score:0.1f}%)"
)
print("Number of features used:", X.shape[1] * X.shape[2])

# %%
# Use the whole window.
X = to_predict.get_data(picks="eeg", exclude="bads", tmin=0)
y = to_predict.events[:, 2]  # target: auditory left vs visual left

clf = make_pipeline(
    Scaler(to_predict.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear", random_state=31),
)

scores = cross_validate(
    clf, X, y, cv=5, n_jobs=None, scoring="roc_auc", return_train_score=True
)

# Mean scores across cross-validation splits
test_score = np.mean(scores["test_score"], axis=0)
train_score = np.mean(scores["train_score"], axis=0)
print(
    f"Decoding performance at [0ms-700ms]: {100 * test_score:0.1f}% "
    f"(train: {100 * train_score:0.1f}%)"
)
print("Number of features used:", X.shape[1] * X.shape[2])

# %%
# Use the baseline
X = to_predict.get_data(picks="eeg", exclude="bads", tmin=-0.3, tmax=0)
y = to_predict.events[:, 2]  # target: auditory left vs visual left

clf = make_pipeline(
    Scaler(to_predict.info),
    Vectorizer(),
    LogisticRegression(solver="liblinear", random_state=31),
)

scores = cross_validate(
    clf, X, y, cv=5, n_jobs=None, scoring="roc_auc", return_train_score=True
)

# Mean scores across cross-validation splits
test_score = np.mean(scores["test_score"], axis=0)
train_score = np.mean(scores["train_score"], axis=0)
print(
    f"Decoding performance at [-300ms-0ms]: {100 * test_score:0.1f}% "
    f"(train: {100 * train_score:0.1f}%)"
)
print("Number of features used:", X.shape[1] * X.shape[2])
# %%
# Do plots for slides
picks = [
    "EEG 017",
    "EEG 018",
    "EEG 025",
    "EEG 026",
    "EEG 023",
    "EEG 024",
    "EEG 034",
    "EEG 035",
]
fig, axes = plt.subplots(5, 1, figsize=(8, 8), sharex=True, sharey=True)

to_plot = [to_predict["auditory/left"].average(), to_predict["visual/left"].average()]
axes[0].axvline(0.1, color="g", linestyle="-", linewidth=2, alpha=0.3)
mne.viz.plot_compare_evokeds(
    to_plot,
    combine="mean",
    picks=picks,
    show_sensors=True,
    axes=axes[0],
    legend=False,
    show=False,
)

[
    axes[1].axvline(x, color="g", linestyle="-", linewidth=1, alpha=0.3)
    for x in np.arange(0.05, 0.15, 0.001)
]
mne.viz.plot_compare_evokeds(
    to_plot,
    combine="mean",
    picks=picks,
    show_sensors=True,
    axes=axes[1],
    legend=False,
    show=False,
)
axes[1].set_title("")

[
    axes[2].axvline(x, color="g", linestyle="-", linewidth=1, alpha=0.3)
    for x in np.arange(0.05, 0.3, 0.001)
]
mne.viz.plot_compare_evokeds(
    to_plot,
    combine="mean",
    picks=picks,
    show_sensors=True,
    axes=axes[2],
    legend=False,
    show=False,
)
axes[2].set_title("")

[
    axes[3].axvline(x, color="g", linestyle="-", linewidth=1, alpha=0.3)
    for x in np.arange(0, 0.7, 0.001)
]
mne.viz.plot_compare_evokeds(
    to_plot,
    combine="mean",
    picks=picks,
    show_sensors=True,
    axes=axes[3],
    legend=False,
    show=False,
)
axes[3].set_title("")

[
    axes[4].axvline(x, color="g", linestyle="-", linewidth=1, alpha=0.3)
    for x in np.arange(-0.3, 0, 0.001)
]
mne.viz.plot_compare_evokeds(
    to_plot,
    combine="mean",
    picks=picks,
    show_sensors=True,
    axes=axes[4],
    legend=False,
    show=False,
)
axes[4].set_title("")

plt.subplots_adjust(hspace=0.5)
plt.show()
# %%
