import warnings

import mne
import moabb
from moabb.datasets import Zhou2016


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")


##############################################################################
# Initializing Datasets
# ---------------------
dataset = Zhou2016().get_data()

# Dataset has got:
# Dictionary with each of the patients (4 in total):
#   Diccionary with each of the sessions of the patient (3 each):
#       Dictionary with each of the runs done the session (2 each):
#           RawCNT object with the data. 14 EEG channels.

patient_one = dataset[1]

session_one_run_one = patient_one["0"]["0"]

display(session_one_run_one)

# Drop non EEG channels
non_eeg_chs = [channel["ch_name"] for channel in session_one_run_one.info["chs"] if channel["kind"] != 2]
session_one_run_one = session_one_run_one.drop_channels(non_eeg_chs)

# Plot data
session_one_run_one.plot()

# Quantity of data
print(f"Quantity of data by using len: {len(session_one_run_one)}")

session_one_run_one_data = session_one_run_one.get_data()
print(f"Nº of channels and quantity of data by using shape after get_data: {session_one_run_one_data.shape}")
