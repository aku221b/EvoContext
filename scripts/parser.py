import os
import pandas as pd
import h5py
import argparse
from pathlib import Path

# Get the repo root (i.e., STAGSMOTE) assuming this file is inside the repo
repo_root = Path(__file__).resolve().parent.parent  # adjust .parent levels as needed

freq = 256
PIL_interval = 30*60
SPH_interval = 5*60

def chunk_h5_signal(labels_final_base_path,h5_path, chunk_duration_sec, label, start_time, end_time, label_save_path, task):
    os.makedirs(labels_final_base_path, exist_ok=True)

    if not os.path.exists(h5_path):
        return 

    with h5py.File(h5_path, "r") as hf:
        signal = hf["filtered_signal"][:]  # shape: (channels, samples)


    samples_per_chunk = chunk_duration_sec * freq
    num_channels, total_samples = signal.shape
    chunk_count = total_samples // samples_per_chunk

    if num_channels != 22: return

    file_base_name = h5_path.split("/")[-1]
    buffer = 5*60*freq

    if task == "detection": 
        for i in range(chunk_count):
            start_ictal_sample = int(start_time)*freq
            end_ictal_sample = int(end_time)*freq

            start_clip_sample = i * samples_per_chunk
            end_clip_sample = (i + 1) * samples_per_chunk

            if label == 1:
                if start_clip_sample > end_ictal_sample or end_clip_sample-1 < start_ictal_sample:
                    label_clip = 0
                else: label_clip = 1
            else : label_clip = 0

            row_entry = [file_base_name, str(i), str(label_clip)]
            with open(label_save_path, "a") as f:
                f.write(",".join(row_entry) + "\n")
            
    elif task == "prediction":
    
        for i in range(chunk_count):
            start_ictal_sample = int(start_time)*freq
            start_PIL_sample = int(start_time)*freq - PIL_interval*freq
            start_SPH_sample = int(start_time)*freq - SPH_interval*freq
            end_ictal_sample = int(end_time)*freq

            start_clip_sample = i * samples_per_chunk
            end_clip_sample = (i + 1) * samples_per_chunk

            if label == 1 and end_clip_sample < start_SPH_sample and start_clip_sample > start_PIL_sample:
                label_clip = 1
            elif label == 1 and end_clip_sample < start_PIL_sample:
                label_clip = 0
            elif label == 0:
                label_clip = 0
            else: continue

            row_entry = [file_base_name, str(i), str(label_clip)]
            with open(label_save_path, "a") as f:
                f.write(",".join(row_entry) + "\n")

        

def parser_chbmit(p_array,filtered_data_path, labels_final_base_path, clip_size, split, task):
    label_base_path = repo_root / "labels" / "parsed_labels_chbmit"

    if not os.path.exists(label_base_path):
        raise Exception("parsed chbmit labels dont exist...")

    # If you need strings:
    label_base_path = str(label_base_path)
    data_base_path = str(filtered_data_path)
    labels_final_base_path = str(labels_final_base_path)

        
    for i in p_array:
        ind = str(i).zfill(2)
        label_file = f"chb{ind}_labels.csv"
        df = pd.read_csv(os.path.join(label_base_path, label_file))
        rows = list(df.iterrows())
        n = len(rows)
        for (index, row) in rows:
            file_name = row["File_names"]
            label = row["Labels"]
            start_time = row["Start_time"]
            end_time = row["End_time"]

            file_folder = file_name.split("_")[0]

            file_path = os.path.join(data_base_path, file_name.split(".")[0] + ".h5")

            label_save_path = os.path.join(labels_final_base_path, f"{split}_{task}_{clip_size}s.txt")

            is_seizure_next = False
            if index < n-1 and rows[index+1][1]["Labels"] == "1": is_seizure_next = True

            is_seizure_prev = False
            if index > 0 and rows[index-1][1]["Labels"] == "1": is_seizure_prev = True

            if task == "prediction" and label == "1" and is_seizure_prev: continue
            if task == "prediction" and label == "0" and (is_seizure_prev or is_seizure_next): continue

            chunk_h5_signal(labels_final_base_path,file_path, clip_size, label, start_time, end_time, label_save_path, task)


if __name__ == "__main__":
    t_parser = argparse.ArgumentParser(description="parses script arguments")

    t_parser.add_argument("--test_participants", type=int, nargs="+", help="space seperated list of participants to use for testing")
    t_parser.add_argument("--task", type=str, help="enter the task you want to run")
    t_parser.add_argument("--dataset", type=str, help="enter the dataset")
    t_parser.add_argument("--clip_size", type=int, help="enter the clip size")
    t_parser.add_argument("--filtered_data_path", type=str, help="enter filtered data path")
    t_parser.add_argument("--labels_final_base_path", type=str, help="enter final labels dir")

    args = t_parser.parse_args()

    splits = ["train", "dev", "test"]

    if args.dataset == "CHBMIT":
        total_participants = 24
    else: 
        raise Exception("dataset not implementeted")



    test_participants = args.test_participants
    train_participants = [ i for i in range(1,total_participants+1) if not i in test_participants]

    for split in splits:
        if split == "train":
            if args.dataset == "CHBMIT":
                parser_chbmit(train_participants,args.filtered_data_path, args.labels_final_base_path,  args.clip_size, split, args.task)
            else:
                raise Exception("dataset not implemented")
        elif split == "dev":
            if args.dataset == "CHBMIT":
                parser_chbmit(test_participants,args.filtered_data_path, args.labels_final_base_path, args.clip_size, split, args.task)
            else:
                raise Exception("dataset not implemented")

        else: 
            if args.dataset == "CHBMIT":
                parser_chbmit(test_participants, args.filtered_data_path, args.labels_final_base_path, args.clip_size, split, args.task)
            else:
                raise Exception("dataset not implemented")
