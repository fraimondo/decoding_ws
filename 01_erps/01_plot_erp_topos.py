# %% Load data

import matplotlib.pyplot as plt
import mne
import numpy as np

# Load RAW data
root = mne.datasets.sample.data_path() / "MEG" / "sample"
raw_file = root / "sample_audvis_filt-0-40_raw.fif"
raw = mne.io.read_raw_fif(raw_file, preload=False)

events_file = root / "sample_audvis_filt-0-40_raw-eve.fif"
events = mne.read_events(events_file)

raw.crop(tmax=90)  # in seconds (happens in-place)
# discard events >90 seconds (not strictly necessary, but avoids some warnings)
events = events[events[:, 0] <= raw.last_samp]

# Pick only EEG channels
raw.pick(["eeg", "eog"]).load_data()

# Set average ref
mne.set_eeg_reference(raw, ref_channels="average", projection=True)

# Filter
raw.filter(l_freq=0.1, h_freq=None)

# Create epochs
event_dict = {
    "auditory/left": 1,
    "auditory/right": 2,
    "visual/left": 3,
    "visual/right": 4,
    "face": 5,
    "buttonpress": 32,
}

epochs = mne.Epochs(
    raw,
    events,
    event_id=event_dict,
    tmin=-0.3,
    tmax=0.7,
    preload=True,
    baseline=(None, 0),
)

# Reject bad epochs based on EEG and EOG thresholds
reject_criteria = {"eeg": 100e-6, "eog": 200e-6}  # 100 µV, 200 µV
epochs.drop_bad(reject=reject_criteria)

epochs.save("../data/sample-epo.fif", overwrite=True)

# %% Compare evoked responses for left auditory and left visual stimuli
aud_l = epochs["auditory/left"].average()
vis_l = epochs["visual/left"].average()

picks = [
    "EEG 017",
    "EEG 018",
    "EEG 025",
    "EEG 026",
    # "EEG 023",
    # "EEG 024",
    # "EEG 034",
    # "EEG 035",
]
to_compare = {
    "auditory/left": aud_l,
    "visual/left": vis_l,
}
mne.viz.plot_compare_evokeds(
    to_compare, picks=picks, combine="mean", show_sensors=True
)

# %%
fig, ax = plt.subplots(4, 2, figsize=(8, 5), sharex=True, sharey=True)
ax[0, 0].plot(
    epochs["auditory/left"][0].times,
    epochs["auditory/left"][0].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)
ax[1, 0].plot(
    epochs["auditory/left"][1].times,
    epochs["auditory/left"][1].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)
ax[2, 0].plot(
    epochs["auditory/left"][2].times,
    epochs["auditory/left"][2].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)
ax[3, 0].plot(
    epochs["auditory/left"][3].times,
    epochs["auditory/left"][3].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)

ax[3, 0].set_xlabel("Time (s)")
ax[0, 0].set_title("Auditory Left - Epochs 0-3")

ax[0, 1].plot(
    epochs["visual/left"][0].times,
    epochs["visual/left"][0].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)

ax[1, 1].plot(
    epochs["visual/left"][1].times,
    epochs["visual/left"][1].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)
ax[2, 1].plot(
    epochs["visual/left"][2].times,
    epochs["visual/left"][2].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)
ax[3, 1].plot(
    epochs["visual/left"][3].times,
    epochs["visual/left"][3].pick(picks).average().get_data().mean(axis=0)
    * 1e5,
)

ax[3, 1].set_xlabel("Time (s)")
ax[0, 1].set_title("Visual Left - Epochs 0-3")

# %%

times = [0.1]
time_idx = epochs.time_as_index(times)
fig, ax = plt.subplots(5, 10, figsize=(15, 7), sharex=True, sharey=True)

info = mne.pick_info(
    epochs.info,
    [
        i
        for i, sel in enumerate(epochs.ch_names)
        if sel not in epochs.info["bads"]
    ],
)

for i in range(25):
    t_epochs = epochs["auditory/left"]
    t_row = i // 5
    t_col = i % 5
    t_ax = ax[t_row, t_col]
    mne.viz.plot_topomap(
        data=np.squeeze(t_epochs.get_data(picks="eeg")[i, :, time_idx]) * 1e-6,
        pos=info,
        axes=t_ax,
        show=False,
    )

for i in range(25):
    t_epochs = epochs["visual/left"]
    t_row = i // 5
    t_col = i % 5 + 5
    t_ax = ax[t_row, t_col]
    mne.viz.plot_topomap(
        data=np.squeeze(t_epochs.get_data(picks="eeg")[i, :, time_idx]) * 1e-6,
        pos=info,
        axes=t_ax,
        show=False,
    )
# %%
