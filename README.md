# STAGSMOTE

**STAGSMOTE** is a state-of-the-art (SOTA) **asymmetric time-then-graph GNN encoder** with a **DCRNN decoder architecture** for self-supervised learning on EEG data.

---

## 🚀 Running the Pipeline for the CHB-MIT Dataset

Follow these steps to set up and run the pipeline.

---

### ✅ Step 1: Create a Conda Environment

```bash
conda create -n my_env python=3.10
conda activate my_env
```

---

### ✅ Step 2: Install Dependencies

> ⚠️ `requirements.txt` may not capture all dependencies. You might need to install some packages manually.

```bash
pip install -r requirements.txt
```

---

### ✅ Step 3: Filter Raw EEG Data

From the **root directory** of the repo, run:

```bash
python -m data.filter_signals --raw_edf_dir <path_to_raw_data_root_dir>
```

#### Example

```bash
python -m data.filter_signals --raw_edf_dir /path/to/chb-mit-scalp-eeg-database-1.0.0
```

The above step will store the filtered signals in:

```
<repo_root>/data/filtered_data
```

---

### ✅ Step 4: Run the SSL + Downstream Pipeline

Use the provided shell script to run both the pretraining (SSL) and downstream tasks:

```bash
sh run_ssl_plus_downstream.sh <num_participants> <num_epochs_downstream> <num_epochs_downstream> <num_epochs_ssl> <dataset> <filtered_data_dir> <input_len>
```

#### Argument Description

- `<num_participants>`: Number of participants to run in leave-one-out manner  
- `<num_epochs_downstream>`: Number of epochs for downstream task (SSL is fixed to 2 epochs)  
- `<save_dir_ssl>`: Directory to log the SSL task results  
- `<save_dir_downstream>`: Directory to log the downstream task results  
- `<task>`: `detection` or `prediction`  
  *(⚠️ `prediction` pipeline is not implemented yet)*  
- `<dataset>`: `CHBMIT` or `TUSZ`  
  *(⚠️ TUSZ support is not available yet)*  
- `<raw_data_dir>`: Path to the raw data directory  

---

## 🛠️ Additional Scripts

To run **only** the SSL task:

```bash
sh run_ssl.sh <args>
```

To run **only** the downstream task:

```bash
sh run.sh <args> 
```

---
