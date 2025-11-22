import sys

sys.path.append("../")
from .data_utils import resampleData, getEDFsignals, getOrderedChannels
from tqdm import tqdm
import argparse
import numpy as np
import os
import h5py
import mne
selected_channels_chbmit = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-1",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]
selected_channels_tusz = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-1",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
    "P7-T7",
    "T7-FT9",
    "FT9-FT10",
    "FT10-T8",
]


def get_filtered_data(raw):
    raw.notch_filter([60], fir_design='firwin') 
    raw.filter(0.5, 50, fir_design='firwin')
    return raw
def normalise_data(data):
    return ((data - np.mean(data, axis=1, keepdims=True)) / np.std(data,axis=1, keepdims=True))

def filter_all(raw_edf_dir, dataset, output_dir):
    if not os.path.exists(raw_edf_dir):
        Exception("raw edf dir does not exist. Please check the path provided...")
    os.makedirs(output_dir, exist_ok=True)
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    failed_files = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]

        if edf_fn.split(".")[-1] == "seizures": continue

        save_fn = os.path.join(output_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            continue
        try:
            raw = mne.io.read_raw_edf(edf_fn,preload=True)
            if dataset == "chbmit":
                raw.pick(selected_channels_chbmit)
            elif dataset == "TUSZ":
                raw.pick(selected_channels_tusz)
            else: 
                Exception("Wrong dataset, Note dataset value must be either [TUSZ or chbmit]")
            raw = get_filtered_data(raw)
            data, times = raw[:]
            data = normalise_data(data)
            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("filtered_signal", data=data)
                hf.create_dataset("filtered_freq", data=256)

        except BaseException as e:
            print(f"An error occurred: {e}")  
            print(f"Failed to process {edf_fn}")
            failed_files.append(edf_fn)

    print("DONE. {} files failed.".format(len(failed_files)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Filter.")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        default=None,
        help="Full path to raw edf files...",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="dataset name to be used, Choose from [TUSZ, chbmit]...",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output directory with filtered signals... ",
    )
    args = parser.parse_args()

    filter_all(args.raw_edf_dir, args.dataset, args.output_dir)
