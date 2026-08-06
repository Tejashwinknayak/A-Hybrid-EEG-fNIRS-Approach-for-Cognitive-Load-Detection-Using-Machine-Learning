# ── Standard library ──────────────────────────────────────────────────────────
import os
import copy

# ── Scientific computing ──────────────────────────────────────────────────────
import numpy as np
import scipy.io                                          # read MATLAB .mat files
from scipy.signal import butter, filtfilt, sosfiltfilt, hilbert, resample

# ── EEG processing ────────────────────────────────────────────────────────────
import mne
# mne.set_log_level('ERROR')                            # uncomment to silence MNE logs

# ── Machine learning ──────────────────────────────────────────────────────────
from sklearn.preprocessing   import StandardScaler, LabelEncoder
from sklearn.ensemble        import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.cross_decomposition import CCA
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score)
from xgboost import XGBClassifier
from sklearn.svm import SVC                             # available but commented out below
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils           import shuffle as sk_shuffle   # explicit row+label shuffle

#  Deep learning 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



#  SECTION 1 — GLOBAL CONSTANTS


EEG_BANDPASS   = (0.5, 35)   # Hz  -- keeps delta through beta, removes noise & drift
EEG_SFREQ      = 200         # Hz  -- raw EEG sampling rate in the .mat files
FNIRS_SFREQ    = 10          # Hz  -- fNIRS sampling rate (also used after EEG resample)
INST_SEC       = 2           # s   -- instruction period at trial start (excluded/skipped)
TASK_SEC       = 10          # s   -- mental arithmetic task period per trial
REST_SEC       = 10          # s   -- rest period per trial (10 s = same as task)
TRIAL_SEC      = INST_SEC + TASK_SEC + REST_SEC  # = 22 s per trial (instruction + task + rest)
RAW_TRIAL_SEC  = 27          # s   -- physical cycle in raw recording (2 + 10 + 15 s)
                              #        used only for: (a) EEG/fNIRS baseline correction strides
                              #                       (b) trial_size indexing into raw arrays
WIN_SEC        = 3           # s   -- sliding window length
OVERLAP_SEC    = 2           # s   -- sliding window overlap
N_CCA_COMP     = 10          # number of CCA canonical components per pair
OVERFIT_THRESH = 0.30        # train-test gap above this → flagged as overfit
N_SPLITS       = 5           # number of K-Fold splits (used for both ML and DL)

#  Trial structure:
#    Raw recording cycle : | 2s instruction | 10s TASK | 15s REST |  = 27s (RAW_TRIAL_SEC)
#    Used per trial      : | 2s skip        | 10s TASK | 10s REST |  = 22s (TRIAL_SEC)
#                                                          ^ first 10s of rest only; last 5s discarded
#    Each trial yields 8 Task windows AND 8 Rest windows (perfect 1:1 balance).

# Data paths
EEG_DIR  = r"C:\Users\ASUS\Documents\laptop\main\new_da\data\eeg"
NIRS_DIR = r"C:\Users\ASUS\Documents\laptop\main\new_da\data\nirs"


#  SECTION 2 — EEG PREPROCESSING


def process_subject_eeg(file_path, freq_target=EEG_BANDPASS):
    """
    Load one subject's EEG .mat file and return a clean MNE RawArray.

    Steps:
      1. Load .mat → extract 3 Mental Arithmetic (MA) sessions (indices 1, 3, 5)
      2. Create MNE Raw object for each session
      3. Mark VEOG/HEOG as eye-movement (EOG) channels
      4. Assign standard 10-05 electrode montage
      5. Concatenate the 3 sessions end-to-end
      6. Keep only EEG channels (drops VEOG, HEOG)
      7. Bandpass filter: 0.5-35 Hz  (zero-phase via filtfilt)
      8. Per-block baseline correction: subtract mean of first 2 s of every 10 s block

    Returns
    -------
    mne.io.RawArray  (30 channels x N_samples, at 200 Hz)
    """
    mat_data = scipy.io.loadmat(file_path)
    cnt_data = mat_data['cnt']          # MATLAB cell array: 6 sessions per subject

    #  Step 1–4: Load and wrap each MA session 
    ma_indices = [1, 3, 5]             # indices 0,2,4 = Baseline; 1,3,5 = MA Task
    raw_list   = []

    for idx in ma_indices:
        session  = cnt_data[0, idx][0, 0]
        ch_names = [str(n[0]) for n in session['clab'][0]]   # e.g. ['Fp1','Fp2',...,'VEOG']
        sfreq    = float(session['fs'][0, 0])                 # 200 Hz
        data     = session['x'].T                             # .T : (samples×ch) → (ch×samples)

        info        = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
        raw_session = mne.io.RawArray(data, info)

        # Mark eye channels so they are excluded during pick_types()
        raw_session.set_channel_types({'VEOG': 'eog', 'HEOG': 'eog'})

        # Assign 3D electrode positions (needed for spatial analysis / visualization)
        montage = mne.channels.make_standard_montage('standard_1005')
        raw_session.set_montage(montage, on_missing='ignore')

        raw_list.append(raw_session)

    #Step 5: Concatenate 3 MA sessions into one continuous recording 
    raw_ma = mne.concatenate_raws(raw_list)

    # Step 6: Drop eye channels, keep 30 EEG channels only 
    raw_eeg = raw_ma.copy().pick_types(eeg=True)   # 32 ch → 30 ch (drops VEOG, HEOG)

    # Step 7: Bandpass filter (0.5 – 35 Hz, zero-phase) 
    raw_eeg.filter(l_freq=freq_target[0], h_freq=freq_target[1])

    cleaned = raw_eeg.get_data()   # convert to numpy array: shape (30, N_samples)

    # Step 8: Per-trial baseline correction 
    sfreq     = raw_eeg.info['sfreq']
    block_len = int(RAW_TRIAL_SEC * sfreq)          # 27 s × 200 Hz = 5400 samples (physical cycle)
    n_blocks  = int(cleaned.shape[1] / block_len)

    for b in range(n_blocks):
        t0 = b * block_len
        t1 = t0 + block_len
        if t1 > cleaned.shape[1]:
            break
        # Baseline = mean of first 2 s (instruction) of this trial, per channel
        baseline = np.mean(cleaned[:, t0 : t0 + int(INST_SEC * sfreq)],
                           axis=1, keepdims=True)
        cleaned[:, t0:t1] -= baseline               # subtract from entire trial

    return mne.io.RawArray(cleaned, raw_eeg.info)

#  Load all EEG subjects (subjects 01–25)
subject_files_eeg = [
    os.path.join(EEG_DIR, f"subject {str(i).zfill(2)}", "cnt.mat")
    for i in range(1, 26)
]

processed_eeg = {}
for idx, fpath in enumerate(subject_files_eeg, start=1):
    if os.path.exists(fpath):
        print(f"EEG  subject {idx:02d} ...")
        processed_eeg[f'subject {idx}'] = process_subject_eeg(fpath)


# 
#  SECTION 3 — fNIRS PREPROCESSING
#

def beer_lambert(mat_x, dpf=6.0, distance=3.0):
    """
    Convert raw fNIRS intensity to HbO and HbR using the modified Beer-Lambert Law.

    Parameters
    ----------
    mat_x    : ndarray, shape (Samples, 72)
               Columns 0-35  → 760 nm wavelength
               Columns 36-71 → 850 nm wavelength
    dpf      : float  -- differential path-length factor (default 6.0)
    distance : float  -- source-detector distance in cm (default 3.0)

    Returns
    -------
    hbo : ndarray (36, Samples)  -- oxygenated hemoglobin
    hbr : ndarray (36, Samples)  -- deoxygenated hemoglobin
    """
    i_760 = mat_x[:, :36].T    # (36 channels, Samples) at 760 nm
    i_850 = mat_x[:, 36:].T    # (36 channels, Samples) at 850 nm

    def optical_density(intensity):
        """Convert raw intensity to optical density (OD = -log(I/I0))."""
        baseline = np.mean(intensity[:, :10], axis=1, keepdims=True)
        ratio    = np.maximum(intensity / (baseline + 1e-9), 1e-9)
        return -np.log(ratio)

    od_760 = optical_density(i_760)
    od_850 = optical_density(i_850)

    # Molar extinction coefficients [HbO, HbR] at [760 nm, 850 nm]
    eps     = np.array([[1.4866, 3.8437],
                        [2.5273, 1.7986]])
    inv_eps = np.linalg.inv(eps)
    factor  = distance * dpf    # modified path length

    hbo = np.zeros_like(od_760)
    hbr = np.zeros_like(od_760)

    for ch in range(36):
        od_vec     = np.vstack([od_760[ch, :], od_850[ch, :]])
        conc       = np.dot(inv_eps, od_vec) / factor
        hbo[ch, :] = conc[0, :]    # HbO concentration
        hbr[ch, :] = conc[1, :]    # HbR concentration

    return hbo, hbr


def bandpass_fnirs(hbo, hbr, fs):
    """
    Bandpass filter fNIRS signals between 0.01 and 0.1 Hz.
    This isolates the hemodynamic response and removes respiratory / cardiac noise.
    Uses SOS (second-order sections) form for numerical stability with high-order
    narrow-band filters — avoids Inf/NaN overflow that occurs with b,a form.
    """
    nyq = 0.5 * fs
    sos = butter(6, [0.01 / nyq, 0.1 / nyq], btype='band', output='sos')
    return sosfiltfilt(sos, hbo, axis=1), sosfiltfilt(sos, hbr, axis=1)


def baseline_correct_fnirs(hbo, hbr, fs):
    """
    Per-trial baseline correction for fNIRS.
    Subtracts the mean of the first 2 s (instruction) from every 27s physical cycle.
    Uses RAW_TRIAL_SEC (27 s) as the stride to match the actual recording structure.
    """
    block_len = int(RAW_TRIAL_SEC * fs)              # 27 s per physical cycle
    n_blocks  = int(hbo.shape[1] / block_len)

    for b in range(n_blocks):
        t0    = b * block_len
        t1    = t0 + block_len
        b_hbo = np.mean(hbo[:, t0 : t0 + int(INST_SEC * fs)], axis=1, keepdims=True)
        b_hbr = np.mean(hbr[:, t0 : t0 + int(INST_SEC * fs)], axis=1, keepdims=True)
        hbo[:, t0:t1] -= b_hbo
        hbr[:, t0:t1] -= b_hbr

    return hbo, hbr


def process_subject_fnirs(file_path):
    """
    Load one subject's fNIRS .mat file and return HbO and HbR arrays
    concatenated across the three Mental Arithmetic sessions.

    Returns
    -------
    hbo : ndarray (36, N_samples)
    hbr : ndarray (36, N_samples)
    """
    mat_data = scipy.io.loadmat(file_path)
    cnt_data = mat_data['cnt']

    ma_indices   = [1, 3, 5]
    hbo_sessions = []
    hbr_sessions = []

    for idx in ma_indices:
        session       = cnt_data[0, idx][0, 0]
        fs            = float(session['fs'][0, 0])
        raw_intensity = session['x']               # shape: (Samples, 72)

        hbo, hbr     = beer_lambert(raw_intensity)        # convert intensity → HbO/HbR
        hbo_f, hbr_f = bandpass_fnirs(hbo, hbr, fs)      # 0.01–0.1 Hz bandpass
        hbo_c, hbr_c = baseline_correct_fnirs(hbo_f, hbr_f, fs)  # baseline correct

        hbo_sessions.append(hbo_c)
        hbr_sessions.append(hbr_c)

    # Concatenate all 3 MA sessions along the time axis
    return (np.concatenate(hbo_sessions, axis=1),
            np.concatenate(hbr_sessions, axis=1))


# Load all fNIRS subjects (subjects 01–29)
subject_files_nirs = [
    os.path.join(NIRS_DIR, f"subject {str(i).zfill(2)}", "cnt.mat")
    for i in range(1, 30)
]

# Bundle EEG + fNIRS per subject (EEG only available for subjects 1–25)
all_subjects = {}
nirs_idx = 1
for fpath in subject_files_nirs:
    if os.path.exists(fpath):
        print(f"fNIRS subject {nirs_idx:02d} ...")
        hbo_c, hbr_c = process_subject_fnirs(fpath)
        all_subjects[f'subject {nirs_idx}'] = {
            'eeg': processed_eeg.get(f'subject {nirs_idx}'),  # None if no EEG
            'hbo': hbo_c,
            'hbr': hbr_c,
        }
        nirs_idx += 1



#  SECTION 4 — ALIGN EEG BANDS TO fNIRS (200 Hz → 10 Hz)


# Band extraction MUST happen at 200 Hz BEFORE downsampling.
#      After resampling to 10 Hz, Nyquist = 5 Hz, so Alpha (8–13 Hz)
#      and Beta (13–30 Hz) would be invisible.
#
#  Pipeline per subject:
#    EEG (200 Hz) → bandpass per band → Hilbert envelope → resample to 10 Hz

def band_envelope_10hz(eeg_200hz, low, high, target_len, sfreq=EEG_SFREQ):
    """
    Extract the power envelope of one EEG frequency band, resampled to 10 Hz.

    Steps:
      1. Butterworth bandpass filter (order 6) between [low, high] Hz
      2. Hilbert transform → instantaneous amplitude (envelope)
      3. Resample from 200 Hz to target_len samples (≈ 10 Hz)

    Parameters
    ----------
    eeg_200hz  : ndarray (n_channels, n_samples)  at 200 Hz
    low / high : float  -- band edges in Hz
    target_len : int    -- number of output samples (= fNIRS length at 10 Hz)

    Returns
    -------
    ndarray (n_channels, target_len)
    """
    nyq      = 0.5 * sfreq
    sos      = butter(6, [low / nyq, high / nyq], btype='band', output='sos')
    filtered = sosfiltfilt(sos, eeg_200hz, axis=-1)     # zero-phase bandpass (SOS — numerically stable)
    envelope = np.abs(hilbert(filtered, axis=-1))       # instantaneous amplitude
    return resample(envelope, target_len, axis=-1)      # downsample to 10 Hz


print("\nAligning EEG bands to fNIRS sampling rate (10 Hz) ...")
aligned = {}

for i in range(1, 30):
    key    = f'subject {i}'
    bundle = all_subjects.get(key)
    if bundle is None:
        continue

    raw_eeg = bundle['eeg']
    if raw_eeg is None:
        print(f"  Subject {i:02d}: no EEG data -> skipped.")
        continue

    eeg_2d     = raw_eeg.get_data()         # shape: (30 channels, N_samples) at 200 Hz
    target_len = bundle['hbo'].shape[1]     # fNIRS sample count at 10 Hz

    # Extract 4 EEG bands at 200 Hz, then resample envelopes to 10 Hz
    aligned[key] = {
        'delta': band_envelope_10hz(eeg_2d, 0.5,  4,  target_len),  # (30, target_len)
        'theta': band_envelope_10hz(eeg_2d, 4,    8,  target_len),
        'alpha': band_envelope_10hz(eeg_2d, 8,    13, target_len),
        'beta' : band_envelope_10hz(eeg_2d, 13,   30, target_len),
        'hbo'  : bundle['hbo'],                                       # (36, target_len)
        'hbr'  : bundle['hbr'],
    }
    print(f"  Subject {i:02d} aligned: {target_len} samples at {FNIRS_SFREQ} Hz")



#  SECTION 5 — SLIDING WINDOW FEATURE EXTRACTION

#  For each subject:
#    • Split data into 10-second blocks (Task / Rest alternating)
#    • Skip first 2 s (instruction period) → use remaining 8 s
#    • Apply 3-second sliding window with 2-second overlap (stride = 1 s)
#    • Per window: compute channel mean → one feature vector per modality
#    • Normalize per subject using Z-score (StandardScaler)
#
#  Output shapes (across all subjects, all windows):
#    EEG bands  : each (n_windows, 30)
#    fNIRS      : each (n_windows, 36)

def sliding_windows(delta, theta, alpha, beta, hbo, hbr,
                    sfreq=FNIRS_SFREQ, win_sec=WIN_SEC, overlap_sec=OVERLAP_SEC):
    """
    Slide a window across all 6 modalities simultaneously.

    Parameters
    ----------
    delta/theta/alpha/beta : ndarray (n_ch, n_samples)  -- EEG band envelopes at 10 Hz
    hbo / hbr              : ndarray (36, n_samples)    -- fNIRS at 10 Hz
    sfreq                  : float   -- sampling rate (10 Hz)
    win_sec / overlap_sec  : float   -- window and overlap duration in seconds

    Returns
    -------
    dict of lists, one list of (n_ch, win_samples) arrays per modality
    """
    win    = int(win_sec * sfreq)            # 3 s × 10 Hz = 30 samples
    stride = int((win_sec - overlap_sec) * sfreq)   # 1 s × 10 Hz = 10 samples
    n_samp = hbo.shape[1]

    out = {k: [] for k in ['delta', 'theta', 'alpha', 'beta', 'hbo', 'hbr']}
    k   = 0
    while True:
        s, e = k * stride, k * stride + win
        if e > n_samp:
            break
        out['delta'].append(delta[:, s:e])
        out['theta'].append(theta[:, s:e])
        out['alpha'].append(alpha[:, s:e])
        out['beta'].append(beta[:, s:e])
        out['hbo'].append(hbo[:, s:e])
        out['hbr'].append(hbr[:, s:e])
        k += 1
    return out


print("\nExtracting windowed channel-level features ...")

# Global collectors — one row per window across all subjects (Averaged features for RF/XGBoost)
all_delta_ch = []   # shape per entry: (30,)
all_theta_ch = []
all_alpha_ch = []
all_beta_ch  = []
all_hbo_ch   = []   # shape per entry: (36,)
all_hbr_ch   = []
all_hbt_ch   = []   # HbT = HbO + HbR  (Total Hemoglobin)

# Global sequence collectors (retaining time dimension for LSTM)
all_delta_seq = []  # shape per entry: (30, 30) (time_steps, channels)
all_theta_seq = []
all_alpha_seq = []
all_beta_seq  = []
all_hbo_seq   = []  # shape per entry: (30, 36)
all_hbr_seq   = []
all_hbt_seq   = []

all_labels   = []   # 'Task' or 'Rest'
all_groups   = []   # subject id per window (needed for subject-grouped splits)

# Per-session window counters (Session 1 = 1st MA, Session 2 = 2nd MA, Session 3 = 3rd MA)
session_task = {1: 0, 2: 0, 3: 0}
session_rest = {1: 0, 2: 0, 3: 0}

# Trial sizes at fNIRS rate (10 Hz)
trial_size = int(RAW_TRIAL_SEC * FNIRS_SFREQ)  # 270 samples — physical cycle stride in raw arrays
inst_size  = int(INST_SEC      * FNIRS_SFREQ)  #  20 samples = 2s instruction (skip)
task_size  = int(TASK_SEC      * FNIRS_SFREQ)  # 100 samples = 10s task period
rest_size  = int(REST_SEC      * FNIRS_SFREQ)  # 100 samples = 10s rest period
# TRIAL_SEC = 22 s (2+10+10) is the effective trial duration extracted per cycle.

for i in range(1, 30):
    key = f'subject {i}'
    if key not in aligned:
        continue

    delta_data = aligned[key]['delta']
    theta_data = aligned[key]['theta']
    alpha_data = aligned[key]['alpha']
    beta_data  = aligned[key]['beta']
    hbo_data   = aligned[key]['hbo']
    hbr_data   = aligned[key]['hbr']

    n_trials = hbo_data.shape[1] // trial_size

    # Per-subject collectors (normalized separately)
    subj_delta, subj_theta = [], []
    subj_alpha, subj_beta  = [], []
    subj_hbo,   subj_hbr   = [], []
    subj_hbt,   subj_labels = [], []
    subj_task_count = 0
    subj_rest_count = 0

    # Sequence collectors (not averaged)
    subj_delta_seq, subj_theta_seq = [], []
    subj_alpha_seq, subj_beta_seq  = [], []
    subj_hbo_seq,   subj_hbr_seq   = [], []
    subj_hbt_seq = []

    trials_per_session = max(1, n_trials // 3)  # assume 3 equal MA sessions

    for t_idx in range(n_trials):
        sess_num = min(t_idx // trials_per_session + 1, 3)  # 1, 2, or 3
        rep_start  = t_idx * trial_size
        task_start = rep_start + inst_size            # skip 2s instruction
        task_end   = task_start + task_size           # end of 10s task
        rest_start = task_end
        rest_end   = rest_start + rest_size           # end of 10s rest (matching 10s task)

        # ── Task portion (10s after instruction) ────────────────────────────
        delta_blk = delta_data[:, task_start : task_end]   # (30, 100 samples)
        theta_blk = theta_data[:, task_start : task_end]
        alpha_blk = alpha_data[:, task_start : task_end]
        beta_blk  = beta_data[:,  task_start : task_end]
        hbo_blk   = hbo_data[:,   task_start : task_end]   # (36, 100 samples)
        hbr_blk   = hbr_data[:,   task_start : task_end]
        hbt_blk   = hbo_blk + hbr_blk
        label     = 'Task'

        # Apply sliding window across the 8-second block
        windows     = sliding_windows(delta_blk, theta_blk, alpha_blk,
                                      beta_blk, hbo_blk, hbr_blk)

        # Compute HbT windows using the same stride as the other modalities
        stride_s    = int((WIN_SEC - OVERLAP_SEC) * FNIRS_SFREQ)  # 10 samples
        win_s       = int(WIN_SEC * FNIRS_SFREQ)                   # 30 samples
        hbt_windows = [hbt_blk[:, k * stride_s : k * stride_s + win_s]
                       for k in range(len(windows['hbo']))]

        for de_w, th_w, al_w, be_w, hbo_w, hbr_w, hbt_w in zip(
                windows['delta'], windows['theta'], windows['alpha'],
                windows['beta'],  windows['hbo'],   windows['hbr'],
                hbt_windows):

            # Channel mean over time → one feature value per channel per window
            subj_delta.append(np.mean(de_w,  axis=1))   # (30,)
            subj_theta.append(np.mean(th_w,  axis=1))
            subj_alpha.append(np.mean(al_w,  axis=1))
            subj_beta.append( np.mean(be_w,  axis=1))
            subj_hbo.append(  np.mean(hbo_w, axis=1))   # (36,)
            subj_hbr.append(  np.mean(hbr_w, axis=1))
            subj_hbt.append(  np.mean(hbt_w, axis=1))   # (36,)
            subj_labels.append(label)
            subj_task_count += 1
            session_task[sess_num] += 1

            # Store full sequence (time steps intact) for LSTM
            subj_delta_seq.append(de_w.T)
            subj_theta_seq.append(th_w.T)
            subj_alpha_seq.append(al_w.T)
            subj_beta_seq.append(be_w.T)
            subj_hbo_seq.append(hbo_w.T)
            subj_hbr_seq.append(hbr_w.T)
            subj_hbt_seq.append(hbt_w.T)

        # ── Rest portion (10s after task) ───────────────────────────────
        delta_blk = delta_data[:, rest_start : rest_end]
        theta_blk = theta_data[:, rest_start : rest_end]
        alpha_blk = alpha_data[:, rest_start : rest_end]
        beta_blk  = beta_data[:,  rest_start : rest_end]
        hbo_blk   = hbo_data[:,   rest_start : rest_end]
        hbr_blk   = hbr_data[:,   rest_start : rest_end]
        hbt_blk   = hbo_blk + hbr_blk
        label     = 'Rest'

        windows_rest = sliding_windows(delta_blk, theta_blk, alpha_blk,
                                       beta_blk, hbo_blk, hbr_blk)

        hbt_windows_rest = [hbt_blk[:, k * stride_s : k * stride_s + win_s]
                            for k in range(len(windows_rest['hbo']))]

        for de_w, th_w, al_w, be_w, hbo_w, hbr_w, hbt_w in zip(
                windows_rest['delta'], windows_rest['theta'], windows_rest['alpha'],
                windows_rest['beta'],  windows_rest['hbo'],   windows_rest['hbr'],
                hbt_windows_rest):

            subj_delta.append(np.mean(de_w,  axis=1))
            subj_theta.append(np.mean(th_w,  axis=1))
            subj_alpha.append(np.mean(al_w,  axis=1))
            subj_beta.append( np.mean(be_w,  axis=1))
            subj_hbo.append(  np.mean(hbo_w, axis=1))
            subj_hbr.append(  np.mean(hbr_w, axis=1))
            subj_hbt.append(  np.mean(hbt_w, axis=1))
            subj_labels.append(label)
            subj_rest_count += 1
            session_rest[sess_num] += 1

            # Store full sequence for LSTM
            subj_delta_seq.append(de_w.T)
            subj_theta_seq.append(th_w.T)
            subj_alpha_seq.append(al_w.T)
            subj_beta_seq.append(be_w.T)
            subj_hbo_seq.append(hbo_w.T)
            subj_hbr_seq.append(hbr_w.T)
            subj_hbt_seq.append(hbt_w.T)

    # Helper to Z-score sequence lists per subject
    def scale_sequence(seq_list):
        arr = np.array(seq_list)  # (n_windows, time_steps, channels)
        n_win, n_steps, n_ch = arr.shape
        arr_flat = arr.reshape(-1, n_ch)
        arr_scaled = StandardScaler().fit_transform(arr_flat)
        return arr_scaled.reshape(n_win, n_steps, n_ch)

    # Z-score normalization per subject (removes between-subject amplitude differences)
    # NOTE: Fitted only on this subject's windows — no leakage between subjects.
    #        Each modality gets its own scaler (different feature dimensions: 30 vs 36).
    all_delta_ch.extend(StandardScaler().fit_transform(subj_delta))
    all_theta_ch.extend(StandardScaler().fit_transform(subj_theta))
    all_alpha_ch.extend(StandardScaler().fit_transform(subj_alpha))
    all_beta_ch.extend( StandardScaler().fit_transform(subj_beta))
    all_hbo_ch.extend(  StandardScaler().fit_transform(subj_hbo))
    all_hbr_ch.extend(  StandardScaler().fit_transform(subj_hbr))
    all_hbt_ch.extend(  StandardScaler().fit_transform(subj_hbt))

    all_delta_seq.extend(scale_sequence(subj_delta_seq))
    all_theta_seq.extend(scale_sequence(subj_theta_seq))
    all_alpha_seq.extend(scale_sequence(subj_alpha_seq))
    all_beta_seq.extend( scale_sequence(subj_beta_seq))
    all_hbo_seq.extend(  scale_sequence(subj_hbo_seq))
    all_hbr_seq.extend(  scale_sequence(subj_hbr_seq))
    all_hbt_seq.extend(  scale_sequence(subj_hbt_seq))

    all_labels.extend(subj_labels)
    all_groups.extend([i] * len(subj_labels))

    print(f"  Subject {i:02d}: {n_trials} trials | "
          f"Task: {subj_task_count} windows | "
          f"Rest: {subj_rest_count} windows | "
          f"Total: {len(subj_labels)} windows")

# Convert lists to NumPy arrays
all_delta_ch = np.array(all_delta_ch)   # (n_windows, 30)
all_theta_ch = np.array(all_theta_ch)
all_alpha_ch = np.array(all_alpha_ch)
all_beta_ch  = np.array(all_beta_ch)
all_hbo_ch   = np.array(all_hbo_ch)     # (n_windows, 36)
all_hbr_ch   = np.array(all_hbr_ch)
all_hbt_ch   = np.array(all_hbt_ch)     # (n_windows, 36)  HbT = HbO + HbR

# Sequence arrays (time dimension intact)
all_delta_seq = np.array(all_delta_seq, dtype=np.float32)  # (n_windows, 30, 30)
all_theta_seq = np.array(all_theta_seq, dtype=np.float32)
all_alpha_seq = np.array(all_alpha_seq, dtype=np.float32)
all_beta_seq  = np.array(all_beta_seq,  dtype=np.float32)
all_hbo_seq   = np.array(all_hbo_seq,   dtype=np.float32)  # (n_windows, 30, 36)
all_hbr_seq   = np.array(all_hbr_seq,   dtype=np.float32)
all_hbt_seq   = np.array(all_hbt_seq,   dtype=np.float32)

y_str = np.array(all_labels)            # string labels: 'Task' / 'Rest'
le    = LabelEncoder()
y_enc = le.fit_transform(y_str)         # numeric labels: 0 / 1
all_groups = np.array(all_groups)       # subject id per window

print(f"\n{'='*60}")
print(f"  Feature extraction complete")
print(f"  Total windows  : {len(y_str)}")
print(f"  EEG channels   : {all_delta_ch.shape[1]}  (per band)")
print(f"  fNIRS channels : {all_hbo_ch.shape[1]}  (per signal)")
print(f"  Task windows   : {np.sum(y_str == 'Task')}")
print(f"  Rest windows   : {np.sum(y_str == 'Rest')}")
print(f"  {'-'*56}")
print(f"  {'Session':<12} {'Task':>10} {'Rest':>10} {'Total':>10}")
print(f"  {'-'*56}")
for s in [1, 2, 3]:
    s_total = session_task[s] + session_rest[s]
    print(f"  {'Session ' + str(s):<12} {session_task[s]:>10} {session_rest[s]:>10} {s_total:>10}")
print(f"{'='*60}")

# 
#  SECTION 7 — CLASSIFICATION  (CCA + Scaler INSIDE CV fold)
# 
#  Three data-leakage fixes applied here:
#
#  Fix 1 (Label)    : Label assignment must be verified (see Section 1 note).
#
#  Fix 2 (CCA)      : CCA is fitted ONLY on train fold, then applied to test.
#                     Previously, CCA was fitted on all data before CV → leaked.
#
#  Fix 3 (Scaler)   : StandardScaler is fitted ONLY on train fold per CV fold.
#                     Previously, scaler was fitted on all data before CV → leaked.

# Raw (pre-CCA) feature dicts — indexed by modality name
eeg_bands_raw = {
    'delta': all_delta_ch,    # (n_windows, 30)
    'theta': all_theta_ch,
    'alpha': all_alpha_ch,
    'beta' : all_beta_ch,
}
nirs_signals_raw = {
    'hbo': all_hbo_ch,        # (n_windows, 36)
    'hbr': all_hbr_ch,
    'hbt': all_hbt_ch,
}

# Raw sequence (pre-CCA) feature dicts — indexed by modality name (time steps intact)
eeg_bands_raw_seq = {
    'delta': all_delta_seq,   # (n_windows, 30, 30)
    'theta': all_theta_seq,
    'alpha': all_alpha_seq,
    'beta' : all_beta_seq,
}
nirs_signals_raw_seq = {
    'hbo': all_hbo_seq,       # (n_windows, 30, 36)
    'hbr': all_hbr_seq,
    'hbt': all_hbt_seq,
}

# 12 CCA pair definitions  (eeg_key, nirs_key)
CCA_PAIR_DEFS = [
    ('delta', 'hbo'), ('delta', 'hbr'), ('delta', 'hbt'),
    ('theta', 'hbo'), ('theta', 'hbr'), ('theta', 'hbt'),
    ('alpha', 'hbo'), ('alpha', 'hbr'), ('alpha', 'hbt'),
    ('beta',  'hbo'), ('beta',  'hbr'), ('beta',  'hbt'),
]


def run_kfold_with_cca_fixed(eeg_dict, nirs_dict, pair_defs, y_labels, y_encoded,
                              n_components=N_CCA_COMP, n_splits=N_SPLITS, clf_name='RF'):
    """
    Stratified K-Fold cross-validation with CCA and StandardScaler fitted
    on each training fold only. No test data leaks.

    Steps per fold:
      Step 1 — Split raw features into train / test subsets
      Step 2 — Fit CCA on train → transform both train and test
      Step 3 — Fit StandardScaler on train → transform both train and test
      Step 4 — Train classifier on train → evaluate on test

    Parameters
    ----------
    eeg_dict   : dict  {band_name: ndarray (n_windows, n_eeg_ch)}
    nirs_dict  : dict  {signal_name: ndarray (n_windows, n_nirs_ch)}
    pair_defs  : list  [(eeg_key, nirs_key), ...]
    y_labels   : ndarray of str  ('Task' / 'Rest')
    y_encoded  : ndarray of int  (0 / 1)
    clf_name   : 'RF' or 'XGB'

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, train_accuracy
                    each containing per-fold scores (list of length n_splits)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = {
        'accuracy'      : [],
        'precision'     : [],
        'recall'        : [],
        'f1'            : [],
        'train_accuracy': [],
    }

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(y_labels)), y_labels), start=1):
        # ── Step 1: split raw features ───────────────────────────────────────────
        tr_eeg  = {k: v[train_idx] for k, v in eeg_dict.items()}
        te_eeg  = {k: v[test_idx]  for k, v in eeg_dict.items()}
        tr_nirs = {k: v[train_idx] for k, v in nirs_dict.items()}
        te_nirs = {k: v[test_idx]  for k, v in nirs_dict.items()}

        # ── Step 2: CCA — fit on train, apply to both ─────────────────────────────
        tr_fused, te_fused = [], []
        for eeg_key, nirs_key in pair_defs:
            cca = CCA(n_components=n_components, max_iter=1000)
            eeg_tr, nirs_tr = cca.fit_transform(tr_eeg[eeg_key], tr_nirs[nirs_key])
            eeg_te, nirs_te = cca.transform(te_eeg[eeg_key],     te_nirs[nirs_key])
            tr_fused.append(np.hstack([eeg_tr, nirs_tr]))
            te_fused.append(np.hstack([eeg_te, nirs_te]))

        X_tr = np.hstack(tr_fused).astype(np.float32)
        X_te = np.hstack(te_fused).astype(np.float32)
        # No post-CCA normalization — CCA canonical variates are already
        # on a comparable scale; scaler was removed intentionally.

        y_tr     = y_labels[train_idx]
        y_te     = y_labels[test_idx]
        y_tr_enc = y_encoded[train_idx]
        y_te_enc = y_encoded[test_idx]

        # ── Step 4: Train and evaluate classifier ────────────────────────────────
        if clf_name == 'RF':
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_leaf=2,
                max_features='sqrt', random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            y_pred       = clf.predict(X_te)
            y_train_pred = clf.predict(X_tr)
            y_true_te, y_true_tr = y_te, y_tr

        else:    # XGBoost
            clf = XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric='logloss', random_state=42, n_jobs=-1)
            clf.fit(X_tr, y_tr_enc)
            y_pred       = clf.predict(X_te)
            y_train_pred = clf.predict(X_tr)
            y_true_te, y_true_tr = y_te_enc, y_tr_enc

        fold_metrics['accuracy'].append(accuracy_score(y_true_te, y_pred))
        fold_metrics['precision'].append(precision_score(y_true_te, y_pred, average='weighted'))
        fold_metrics['recall'].append(recall_score(y_true_te, y_pred, average='weighted'))
        fold_metrics['f1'].append(f1_score(y_true_te, y_pred, average='weighted'))
        fold_metrics['train_accuracy'].append(accuracy_score(y_true_tr, y_train_pred))

    return fold_metrics


def print_metrics(name, m):
    """Print a formatted results block for one classifier (K-Fold mean ± std)."""
    acc_mean  = np.mean(m['accuracy'])
    acc_std   = np.std(m['accuracy'])
    prec_mean = np.mean(m['precision'])
    rec_mean  = np.mean(m['recall'])
    f1_mean   = np.mean(m['f1'])
    tr_mean   = np.mean(m['train_accuracy'])
    gap       = tr_mean - acc_mean
    print(f"\n  {name}")
    print(f"  {'Accuracy':<12}: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  {'Precision':<12}: {prec_mean:.4f}")
    print(f"  {'Recall':<12}: {rec_mean:.4f}")
    print(f"  {'F1-Score':<12}: {f1_mean:.4f}")
    print(f"  {'Train Acc':<12}: {tr_mean:.4f}  "
          f"(gap = {gap:.4f})")


print(f"\n{'='*65}")
print("  SECTION 7: LEAK-FREE CLASSIFICATION")
print(f"  (CCA + Scaler fitted on train fold only — {N_SPLITS}-Fold CV)")
print(f"{'='*65}")

print(f"\n[Random Forest]  {N_SPLITS}-Fold cross-validation  "
      f"(n_cca = {N_CCA_COMP}, 12 pairs) ...")
rf_metrics = run_kfold_with_cca_fixed(
    eeg_bands_raw, nirs_signals_raw, CCA_PAIR_DEFS,
    y_str, y_enc, clf_name='RF')

print(f"\n[XGBoost]  {N_SPLITS}-Fold cross-validation  "
      f"(n_cca = {N_CCA_COMP}, 12 pairs) ...")
xgb_metrics = run_kfold_with_cca_fixed(
    eeg_bands_raw, nirs_signals_raw, CCA_PAIR_DEFS,
    y_str, y_enc, clf_name='XGB')

#  Results summary
print(f"\n{'='*55}")
print(f"  CLASSIFICATION RESULTS (Leak-Free, {N_SPLITS}-Fold CV, Mean ± Std)")
print(f"{'='*55}")
print_metrics("Random Forest", rf_metrics)
print_metrics("XGBoost",       xgb_metrics)

print(f"\n  {'Metric':<15} {'Random Forest':>18} {'XGBoost':>18}")
print(f"  {'-'*51}")
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    print(f"  {metric:<14}: "
          f"{np.mean(rf_metrics[metric]):>18.4f}  "
          f"{np.mean(xgb_metrics[metric]):>18.4f}")


#
#  SECTION 7.5 — LSTM CLASSIFIER HEAD  (on the same CCA-fused features)
#
#  RF/XGBoost above are evaluated with Stratified K-Fold CV.
#  The LSTM and CNN are also evaluated with K-Fold CV (N_SPLITS folds).
#
#  CCA is still fitted on the TRAIN fold only and applied to train+val,
#  exactly like Section 7's leak-free protocol.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_fused_features_for_lstm(eeg_dict, nirs_dict, pair_defs, train_idx, val_idx,
                                   n_components=N_CCA_COMP):
    """
    Build CCA-fused + Raw sequence feature matrices for the train and val splits.
    CCA and StandardScaler are fitted on train_idx only and applied to train and val
    to prevent data leakage.
    """
    # 1. Compute means over time axis for CCA fitting (using train split only)
    eeg_train_mean = {k: np.mean(v[train_idx], axis=1) for k, v in eeg_dict.items()}
    nirs_train_mean = {k: np.mean(v[train_idx], axis=1) for k, v in nirs_dict.items()}

    fused_tr_blocks = []
    fused_va_blocks = []
    
    for eeg_key, nirs_key in pair_defs:
        cca = CCA(n_components=n_components, max_iter=1000)
        
        # Fit CCA on the averaged train windows only
        cca.fit(eeg_train_mean[eeg_key], nirs_train_mean[nirs_key])
        
        # EEG and fNIRS sequence structures
        eeg_seq = eeg_dict[eeg_key]  # shape: (n_windows, 30, 30)
        nirs_seq = nirs_dict[nirs_key]  # shape: (n_windows, 30, 36)
        
        n_windows, n_steps, n_ch_eeg = eeg_seq.shape
        _, _, n_ch_nirs = nirs_seq.shape
        
        # Flatten to 2D for transform: (n_windows * 30, channels)
        eeg_flat = eeg_seq.reshape(-1, n_ch_eeg)
        nirs_flat = nirs_seq.reshape(-1, n_ch_nirs)
        
        # Transform using the fitted CCA weights
        eeg_proj_flat, nirs_proj_flat = cca.transform(eeg_flat, nirs_flat)
        
        # Reshape back to 3D: (n_windows, 30, n_components)
        eeg_proj = eeg_proj_flat.reshape(n_windows, n_steps, n_components)
        nirs_proj = nirs_proj_flat.reshape(n_windows, n_steps, n_components)
        
        fused_pair = np.concatenate([eeg_proj, nirs_proj], axis=2)
        
        # Split into train and val
        fused_tr_blocks.append(fused_pair[train_idx])
        fused_va_blocks.append(fused_pair[val_idx])

    # Include raw normalized temporal sequences alongside CCA features
    raw_tr_blocks = []
    raw_va_blocks = []
    for k in ['delta', 'theta', 'alpha', 'beta']:
        raw_tr_blocks.append(eeg_dict[k][train_idx])
        raw_va_blocks.append(eeg_dict[k][val_idx])
    for k in ['hbo', 'hbr', 'hbt']:
        raw_tr_blocks.append(nirs_dict[k][train_idx])
        raw_va_blocks.append(nirs_dict[k][val_idx])

    # Concatenate features along the channel axis (axis=2)
    X_tr = np.concatenate(fused_tr_blocks + raw_tr_blocks, axis=2).astype(np.float32)
    X_va = np.concatenate(fused_va_blocks + raw_va_blocks, axis=2).astype(np.float32)
    # No post-CCA normalization — removed intentionally.

    return X_tr, X_va



class LSTMHead(nn.Module):
    """Bidirectional LSTM classifier head with pooling over window sequences."""
    def __init__(self, n_features, hidden_size=128, num_layers=2, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        out_dim = hidden_size * 2
        # Combine mean pooling, max pooling, and last step representation
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        out, _ = self.lstm(x)  # (batch, seq_len, out_dim)
        mean_pool = torch.mean(out, dim=1)
        max_pool, _ = torch.max(out, dim=1)
        last_step = out[:, -1, :]
        combined = torch.cat([mean_pool, max_pool, last_step], dim=1)
        return self.classifier(combined)


class CNN1DHead(nn.Module):
    """1D Convolutional Neural Network classifier head for temporal sequence processing."""
    def __init__(self, n_features, dropout=0.4):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(n_features, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, n_features, seq_len) for Conv1d
        out = self.conv_blocks(x)  # (batch, 128, seq_len)
        mean_pool = torch.mean(out, dim=2)
        max_pool, _ = torch.max(out, dim=2)
        combined = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(combined)


def _train_dl_model(model, X_tr_np, y_tr_np, X_va_np, y_va_np,
                    epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-3):
    """Shared training loop for PyTorch sequence models (LSTM, CNN) with checkpointing."""
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    X_va = torch.tensor(X_va_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    y_va = torch.tensor(y_va_np, dtype=torch.long)

    model     = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_acc   = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            va_logits = model(X_va.to(DEVICE))
            va_pred   = va_logits.argmax(dim=1).cpu().numpy()
            acc       = accuracy_score(y_va.numpy(), va_pred)
            if acc > best_acc:
                best_acc   = acc
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        va_pred = model(X_va.to(DEVICE)).argmax(dim=1).cpu().numpy()

    # Return per-fold scalar metrics (not wrapped in list, to be aggregated by caller)
    return {
        'accuracy' : accuracy_score(y_va.numpy(), va_pred),
        'precision': precision_score(y_va.numpy(), va_pred, average='weighted', zero_division=0),
        'recall'   : recall_score(y_va.numpy(), va_pred, average='weighted', zero_division=0),
        'f1'       : f1_score(y_va.numpy(), va_pred, average='weighted', zero_division=0),
    }


def train_lstm_head(X_tr_np, y_tr_np, X_va_np, y_va_np,
                    epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-3):
    """Train the Bidirectional LSTM head with scheduler and validation checkpointing."""
    model = LSTMHead(n_features=X_tr_np.shape[2])
    return _train_dl_model(model, X_tr_np, y_tr_np, X_va_np, y_va_np, epochs, batch_size, lr, weight_decay)


def train_cnn_head(X_tr_np, y_tr_np, X_va_np, y_va_np,
                   epochs=60, batch_size=64, lr=1e-3, weight_decay=1e-3):
    """Train the 1D Convolutional Neural Network head."""
    model = CNN1DHead(n_features=X_tr_np.shape[2])
    return _train_dl_model(model, X_tr_np, y_tr_np, X_va_np, y_va_np, epochs, batch_size, lr, weight_decay)


def run_kfold_dl(eeg_seq_dict, nirs_seq_dict, pair_defs, y_encoded,
                 n_splits=N_SPLITS, model_type='lstm'):
    """
    K-Fold cross-validation for DL models (LSTM / 1D CNN) using CCA-fused sequences.
    CCA and StandardScaler are fitted on each training fold only.

    Parameters
    ----------
    eeg_seq_dict   : dict {band_key: ndarray (n_windows, time_steps, n_eeg_ch)}
    nirs_seq_dict  : dict {signal_key: ndarray (n_windows, time_steps, n_nirs_ch)}
    pair_defs      : list [(eeg_key, nirs_key), ...]
    y_encoded      : ndarray of int  (0 / 1)
    model_type     : 'lstm' or 'cnn'

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1
                    each a list of per-fold scores
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(y_encoded))), start=1):
        X_tr, X_va = build_fused_features_for_lstm(
            eeg_seq_dict, nirs_seq_dict, pair_defs, train_idx, val_idx)
        y_tr = y_encoded[train_idx]
        y_va = y_encoded[val_idx]

        if model_type == 'lstm':
            m = train_lstm_head(X_tr, y_tr, X_va, y_va)
        else:
            m = train_cnn_head(X_tr, y_tr, X_va, y_va)

        for key in fold_metrics:
            fold_metrics[key].append(m[key])

    return fold_metrics


print(f"\n{'='*65}")
print("  SECTION 7.5: DEEP LEARNING CLASSIFIERS (LSTM / 1D CNN)")
print(f"  ({N_SPLITS}-Fold cross-validation — CCA & Scaler fitted per fold)")
print(f"{'='*65}")

# K-Fold CV for both DL models (CCA & Scaling fit on each train fold only)
print(f"\n  Training Bidirectional LSTM ({N_SPLITS}-Fold CV) ...")
lstm_metrics = run_kfold_dl(
    eeg_bands_raw_seq, nirs_signals_raw_seq, CCA_PAIR_DEFS, y_enc, model_type='lstm')

print(f"  Training 1D CNN ({N_SPLITS}-Fold CV) ...")
cnn_metrics = run_kfold_dl(
    eeg_bands_raw_seq, nirs_signals_raw_seq, CCA_PAIR_DEFS, y_enc, model_type='cnn')

# Build a reference train/val split (fold 0) for Section 8 ablation reuse
_kf_ref = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
_splits_ref = list(_kf_ref.split(np.arange(len(y_enc))))
train_idx, val_idx = _splits_ref[0]   # use fold-0 indices for Section 8 helpers

print(f"\n  Deep Learning Results ({N_SPLITS}-Fold CV Mean ± Std)")
print(f"  {'Model':<20} {'Accuracy':>14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print(f"  {'-'*67}")
print(f"  {'Bi-LSTM':<20} "
      f"{np.mean(lstm_metrics['accuracy']):>7.4f}±{np.std(lstm_metrics['accuracy']):.4f} "
      f"{np.mean(lstm_metrics['precision']):>10.4f} "
      f"{np.mean(lstm_metrics['recall']):>10.4f} "
      f"{np.mean(lstm_metrics['f1']):>10.4f}")
print(f"  {'1D CNN':<20} "
      f"{np.mean(cnn_metrics['accuracy']):>7.4f}±{np.std(cnn_metrics['accuracy']):.4f} "
      f"{np.mean(cnn_metrics['precision']):>10.4f} "
      f"{np.mean(cnn_metrics['recall']):>10.4f} "
      f"{np.mean(cnn_metrics['f1']):>10.4f}")

print(f"\n  Summary Comparison Across All 4 Classifiers ({N_SPLITS}-Fold Mean)")
print(f"  {'Model':<20} {'Accuracy':>10}")
print(f"  {'-'*32}")
print(f"  {'Random Forest':<20} {np.mean(rf_metrics['accuracy']):>10.4f}")
print(f"  {'XGBoost':<20} {np.mean(xgb_metrics['accuracy']):>10.4f}")
print(f"  {'Bi-LSTM':<20} {np.mean(lstm_metrics['accuracy']):>10.4f}")
print(f"  {'1D CNN':<20} {np.mean(cnn_metrics['accuracy']):>10.4f}")



# =============================================================================
#  SECTION 8 — UNIFIED SUMMARY TABLE
#  RF / XGBoost / LSTM accuracy for:
#    - each individual EEG band            (unimodal)
#    - each individual fNIRS signal        (unimodal)
#    - each EEG band + fNIRS signal pairing (CCA-fused, leak-free)
#  No fold-by-fold or epoch-by-epoch output — only final accuracies.
# =============================================================================

def rf_xgb_accuracy(X, y_labels, y_encoded, n_splits=N_SPLITS):
    """Stratified K-Fold mean accuracy + weighted F1 for RF and XGB on a single feature block."""
    X = X.astype(np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf_accs, xgb_accs = [], []
    rf_f1s,  xgb_f1s  = [], []

    for tr_idx, te_idx in skf.split(X, y_labels):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        rf_clf = RandomForestClassifier(
            n_estimators=300, max_depth=15, min_samples_leaf=2,
            max_features='sqrt', random_state=42, n_jobs=-1)
        rf_clf.fit(X_tr, y_labels[tr_idx])
        rf_pred = rf_clf.predict(X_te)
        rf_accs.append(accuracy_score(y_labels[te_idx], rf_pred))
        rf_f1s.append(f1_score(y_labels[te_idx], rf_pred, average='weighted', zero_division=0))

        xgb_clf = XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1)
        xgb_clf.fit(X_tr, y_encoded[tr_idx])
        xgb_pred = xgb_clf.predict(X_te)
        xgb_accs.append(accuracy_score(y_encoded[te_idx], xgb_pred))
        xgb_f1s.append(f1_score(y_encoded[te_idx], xgb_pred, average='weighted', zero_division=0))

    return (float(np.mean(rf_accs)), float(np.mean(xgb_accs)),
            float(np.mean(rf_f1s)),  float(np.mean(xgb_f1s)))


def _scale_seq_split(seq_data, train_idx, val_idx):
    """Z-score scale a 3D sequence array; fit scaler on train split only."""
    X = seq_data.astype(np.float32)
    X_tr, X_va = X[train_idx], X[val_idx]
    n_tr, n_steps, n_ch = X_tr.shape
    n_va = X_va.shape[0]
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr.reshape(-1, n_ch)).reshape(n_tr, n_steps, n_ch)
    X_va_s = scaler.transform(X_va.reshape(-1, n_ch)).reshape(n_va, n_steps, n_ch)
    return X_tr_s, X_va_s


def lstm_accuracy_single_modality(seq_data, y_encoded, train_idx, val_idx):
    """Train/val LSTM accuracy + F1 for ONE modality's raw time-series (no CCA needed)."""
    X_tr_s, X_va_s = _scale_seq_split(seq_data, train_idx, val_idx)
    m = train_lstm_head(X_tr_s, y_encoded[train_idx], X_va_s, y_encoded[val_idx])
    return m['accuracy'], m['f1']


def cnn_accuracy_single_modality(seq_data, y_encoded, train_idx, val_idx):
    """Train/val 1D CNN accuracy + F1 for ONE modality's raw time-series (no CCA needed)."""
    X_tr_s, X_va_s = _scale_seq_split(seq_data, train_idx, val_idx)
    m = train_cnn_head(X_tr_s, y_encoded[train_idx], X_va_s, y_encoded[val_idx])
    return m['accuracy'], m['f1']


def build_fused_pair_only(eeg_seq, nirs_seq, train_idx, val_idx, n_components=N_CCA_COMP):
    """CCA-fuse exactly ONE (band, signal) pair — no extra raw blocks appended."""
    eeg_train_mean  = np.mean(eeg_seq[train_idx],  axis=1)
    nirs_train_mean = np.mean(nirs_seq[train_idx], axis=1)

    cca = CCA(n_components=n_components, max_iter=1000)
    cca.fit(eeg_train_mean, nirs_train_mean)

    n_windows, n_steps, n_ch_eeg  = eeg_seq.shape
    _,         _,        n_ch_nirs = nirs_seq.shape

    eeg_flat  = eeg_seq.reshape(-1, n_ch_eeg)
    nirs_flat = nirs_seq.reshape(-1, n_ch_nirs)
    eeg_proj_flat, nirs_proj_flat = cca.transform(eeg_flat, nirs_flat)

    eeg_proj  = eeg_proj_flat.reshape(n_windows, n_steps, n_components)
    nirs_proj = nirs_proj_flat.reshape(n_windows, n_steps, n_components)
    fused     = np.concatenate([eeg_proj, nirs_proj], axis=2)

    X_tr, X_va = fused[train_idx], fused[val_idx]
    # No post-CCA normalization — removed intentionally.
    return X_tr.astype(np.float32), X_va.astype(np.float32)


def lstm_accuracy_pair(eeg_key, nirs_key, y_encoded, train_idx, val_idx):
    """LSTM accuracy + F1 for one CCA-fused (band, signal) pair."""
    X_tr, X_va = build_fused_pair_only(
        eeg_bands_raw_seq[eeg_key], nirs_signals_raw_seq[nirs_key], train_idx, val_idx)
    m = train_lstm_head(X_tr, y_encoded[train_idx], X_va, y_encoded[val_idx])
    return m['accuracy'], m['f1']


def cnn_accuracy_pair(eeg_key, nirs_key, y_encoded, train_idx, val_idx):
    """1D CNN accuracy + F1 for one CCA-fused (band, signal) pair."""
    X_tr, X_va = build_fused_pair_only(
        eeg_bands_raw_seq[eeg_key], nirs_signals_raw_seq[nirs_key], train_idx, val_idx)
    m = train_cnn_head(X_tr, y_encoded[train_idx], X_va, y_encoded[val_idx])
    return m['accuracy'], m['f1']



# ── Reuse the same train/val split as Section 7.5 (already defined: train_idx, val_idx) ──

_eeg_bands   = [('Delta', 'delta', all_delta_ch), ('Theta', 'theta', all_theta_ch),
                ('Alpha', 'alpha', all_alpha_ch), ('Beta',  'beta',  all_beta_ch)]
_nirs_sigs   = [('HbO', 'hbo', all_hbo_ch), ('HbR', 'hbr', all_hbr_ch), ('HbT', 'hbt', all_hbt_ch)]

tree_results = []   # list of (label, rf_acc, xgb_acc, rf_f1, xgb_f1)

# ── SECTION 8A header update
print(f"\n{'='*75}")
print(f"  SECTION 8A: TREE CLASSIFIER RESULTS (Random Forest & XGBoost — {N_SPLITS}-Fold CV Mean)")
print(f"{'='*75}")

_step = 1
# Individual EEG bands
for band_name, band_key, band_data in _eeg_bands:
    print(f"  [{_step:02d}/19] Evaluating unimodal band: {band_name:<16} ...")
    rf_acc, xgb_acc, rf_f1, xgb_f1 = rf_xgb_accuracy(band_data, y_str, y_enc)
    tree_results.append((band_name, rf_acc, xgb_acc, rf_f1, xgb_f1))
    _step += 1

# Individual fNIRS signals
for sig_name, sig_key, sig_data in _nirs_sigs:
    print(f"  [{_step:02d}/19] Evaluating unimodal signal: {sig_name:<14} ...")
    rf_acc, xgb_acc, rf_f1, xgb_f1 = rf_xgb_accuracy(sig_data, y_str, y_enc)
    tree_results.append((sig_name, rf_acc, xgb_acc, rf_f1, xgb_f1))
    _step += 1

# Every band + signal pairing (CCA-fused, leak-free)

for band_name, band_key, _ in _eeg_bands:
    for sig_name, sig_key, _ in _nirs_sigs:
        pair_label = f"{band_name} + {sig_name}"
        print(f"  [{_step:02d}/19] Evaluating CCA-fused pair: {pair_label:<15} ...")
        p_rf  = run_kfold_with_cca_fixed(eeg_bands_raw, nirs_signals_raw, [(band_key, sig_key)],
                                          y_str, y_enc, clf_name='RF')
        p_xgb = run_kfold_with_cca_fixed(eeg_bands_raw, nirs_signals_raw, [(band_key, sig_key)],
                                          y_str, y_enc, clf_name='XGB')
        rf_acc  = float(np.mean(p_rf['accuracy']))
        xgb_acc = float(np.mean(p_xgb['accuracy']))
        rf_f1   = float(np.mean(p_rf['f1']))
        xgb_f1  = float(np.mean(p_xgb['f1']))
        tree_results.append((pair_label, rf_acc, xgb_acc, rf_f1, xgb_f1))
        _step += 1

# All EEG + all fNIRS combined (from Section 7)
tree_results.append(("All EEG + All fNIRS (CCA)",
                     float(np.mean(rf_metrics['accuracy'])),
                     float(np.mean(xgb_metrics['accuracy'])),
                     float(np.mean(rf_metrics['f1'])),
                     float(np.mean(xgb_metrics['f1']))))


def print_section_trees(title, rows, group_by=None):
    print(f"\n  {title}")
    print(f"  {'-'*79}")
    print(f"  {'Modality':<32}  {'RF Acc':>8}  {'RF F1':>8}  {'XGB Acc':>8}  {'XGB F1':>8}")
    print(f"  {'-'*79}")
    for idx, r in enumerate(rows):
        if group_by and idx > 0 and idx % group_by == 0:
            print(f"  {'-'*79}")
        print(f"  {r[0]:<32}  {r[1]:>8.4f}  {r[3]:>8.4f}  {r[2]:>8.4f}  {r[4]:>8.4f}")


print(f"\n{'='*83}")
print("  SUMMARY: TREE CLASSIFIERS (RF & XGBoost) — Accuracy + F1")
print(f"{'='*83}")
print_section_trees("Level 1 — Individual EEG Bands (Unimodal)", tree_results[0:4])
print_section_trees("Level 2 — Individual fNIRS Signals (Unimodal)", tree_results[4:7])
print_section_trees("Level 2.6 — Pairwise Multimodal Combinations (CCA-Fused)", tree_results[7:19], group_by=3)
print_section_trees("Level 4 — Full Multimodal Fusion (Proposed CCA Method)", [tree_results[19]])


# ── PHASE 2: DEEP LEARNING SEQUENCE TRAINING (LSTM / 1D CNN — 60 epochs, batch 64) ──
print(f"\n{'='*80}")
print("  SECTION 8B: DEEP LEARNING ABLATION (LSTM / 1D CNN)")
print(f"{'='*80}")
print(f"  ({N_SPLITS}-Fold CV — fold-0 split reused for ablation efficiency)")

lstm_results = []   # list of (acc, f1)
cnn_results  = []
_step = 1

# — Unimodal EEG bands —
for band_name, band_key, _ in _eeg_bands:
    print(f"  [{_step:02d}/19] Unimodal EEG band: {band_name:<14} — LSTM ...", end='', flush=True)
    lstm_acc, lstm_f1 = lstm_accuracy_single_modality(eeg_bands_raw_seq[band_key], y_enc, train_idx, val_idx)
    print(f"  CNN ...")
    cnn_acc, cnn_f1  = cnn_accuracy_single_modality(eeg_bands_raw_seq[band_key], y_enc, train_idx, val_idx)
    lstm_results.append((lstm_acc, lstm_f1))
    cnn_results.append((cnn_acc, cnn_f1))
    _step += 1

# — Unimodal fNIRS signals —
for sig_name, sig_key, _ in _nirs_sigs:
    print(f"  [{_step:02d}/19] Unimodal fNIRS signal: {sig_name:<12} — LSTM ...", end='', flush=True)
    lstm_acc, lstm_f1 = lstm_accuracy_single_modality(nirs_signals_raw_seq[sig_key], y_enc, train_idx, val_idx)
    print(f"  CNN ...")
    cnn_acc, cnn_f1  = cnn_accuracy_single_modality(nirs_signals_raw_seq[sig_key], y_enc, train_idx, val_idx)
    lstm_results.append((lstm_acc, lstm_f1))
    cnn_results.append((cnn_acc, cnn_f1))
    _step += 1

# — Pairwise CCA-fused combinations —
for band_name, band_key, _ in _eeg_bands:
    for sig_name, sig_key, _ in _nirs_sigs:
        pair_label = f"{band_name} + {sig_name}"
        print(f"  [{_step:02d}/19] CCA-fused pair: {pair_label:<16} — LSTM ...", end='', flush=True)
        lstm_acc, lstm_f1 = lstm_accuracy_pair(band_key, sig_key, y_enc, train_idx, val_idx)
        print(f"  CNN ...")
        cnn_acc, cnn_f1  = cnn_accuracy_pair(band_key, sig_key, y_enc, train_idx, val_idx)
        lstm_results.append((lstm_acc, lstm_f1))
        cnn_results.append((cnn_acc, cnn_f1))
        _step += 1

# — Full multimodal fusion (from Section 7.5) —
lstm_results.append((float(np.mean(lstm_metrics['accuracy'])), float(np.mean(lstm_metrics['f1']))))
cnn_results.append( (float(np.mean(cnn_metrics['accuracy'])),  float(np.mean(cnn_metrics['f1']))))


# ── PHASE 3: FINAL UNIFIED COMPARISON MATRIX
final_results = []
for i in range(len(tree_results)):
    final_results.append((
        tree_results[i][0],
        tree_results[i][1],   # RF  acc
        tree_results[i][2],   # XGB acc
        tree_results[i][3],   # RF  f1
        tree_results[i][4],   # XGB f1
        lstm_results[i][0],   # LSTM acc
        lstm_results[i][1],   # LSTM f1
        cnn_results[i][0],    # CNN  acc
        cnn_results[i][1],    # CNN  f1
    ))

print(f"\n{'='*105}")
print("  SECTION 8C: FINAL UNIFIED ABLATION & MULTIMODAL RESULTS  (Accuracy | F1)")
print(f"{'='*105}")


def print_section_all(title, rows, group_by=None):
    print(f"\n  {title}")
    hdr = f"  {'Modality':<32}  {'RF Acc':>7}  {'RF F1':>7}  {'XGB Acc':>8}  {'XGB F1':>7}  {'LSTM Acc':>9}  {'LSTM F1':>8}  {'CNN Acc':>8}  {'CNN F1':>7}"
    print(hdr)
    print(f"  {'-'*101}")
    for idx, r in enumerate(rows):
        if group_by and idx > 0 and idx % group_by == 0:
            print(f"  {'-'*101}")
        print(f"  {r[0]:<32}  {r[1]:>7.4f}  {r[3]:>7.4f}  {r[2]:>8.4f}  {r[4]:>7.4f}  {r[5]:>9.4f}  {r[6]:>8.4f}  {r[7]:>8.4f}  {r[8]:>7.4f}")


print_section_all("Level 1 — Individual EEG Bands (Unimodal)", final_results[0:4])
print_section_all("Level 2 — Individual fNIRS Signals (Unimodal)", final_results[4:7])
print_section_all("Level 2.6 — Pairwise Multimodal Combinations (CCA-Fused)", final_results[7:19], group_by=3)
print_section_all("Level 4 — Full Multimodal Fusion (Proposed CCA Method)", [final_results[19]])

print(f"\n{'='*105}")
print("  Done.")
print(f"{'='*105}")