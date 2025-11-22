import os
import pandas as pd
import h5py
import argparse
import random
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
FREQUENCY = 256

def chunk_h5_signal(args):

    if not os.path.exists(args["file_path"]):
        return 

    with h5py.File(args["file_path"], "r") as hf:
        signal = hf["filtered_signal"][:]  # shape: (channels, samples)


    samples_per_chunk = args["clip_size"] * FREQUENCY
    num_channels, total_samples = signal.shape
    chunk_count = total_samples // samples_per_chunk

    if num_channels != 22: return

    file_base_name = args["file_path"].split("/")[-1]

    for i in range(chunk_count-1):

        row_entry = [file_base_name, str(i), str(i+1)]
        with open(args["label_save_path"], "a") as f:
            f.write(",".join(row_entry) + "\n")
            

def parser_CHBMIT(filtered_data_path, final_label_dir, p_array, clip_size, split):
    label_base_path = repo_root / "labels" / "parsed_labels_chbmit"
    label_base_path = str(label_base_path)
    print(label_base_path)

    if not os.path.exists(label_base_path):
        raise Exception(f"parsed labels for CHBMIT dataset not found...{label_base_path}")

    os.makedirs(final_label_dir, exist_ok=True)
    
    for i in p_array:
        ind = str(i).zfill(2)
        label_file = f"chb{ind}_labels.csv"
        df = pd.read_csv(os.path.join(label_base_path, label_file))
        rows = list(df.iterrows())
        n = len(rows)
        for (index, row) in rows:
            file_name = row["File_names"]

            file_folder = file_name.split("_")[0]

            file_path = os.path.join(filtered_data_path, file_name.split(".")[0] + ".h5")

            label_save_path = os.path.join(final_label_dir, f"{split}Set_seq2seq_{clip_size}s.txt")

            chunk_h5_signal({
                "file_path": file_path, 
                "clip_size": clip_size, 
                "label_save_path": label_save_path, 
            })


if __name__ == "__main__":

    t_parser = argparse.ArgumentParser(description="parses script arguments")
    t_parser.add_argument("--filtered_data_path", type=str, help="path to the filtered data")
    t_parser.add_argument("--test_participants", type=int, help="number of participants to use for testing out of 24")
    t_parser.add_argument("--val_participants", type=int,  help="number of participants to use for validation out of 24")
    t_parser.add_argument("--leave_one_out",action="store_true",help="Enable leave-one-out training mode")
    t_parser.add_argument("--participant_number", type=int, help="participant number")
    t_parser.add_argument("--dataset", type=str, help="dataset name")
    t_parser.add_argument("--clip_size", type=int, help="input length")
    t_parser.add_argument("--final_label_dir", type=str, help="final label path dir")


    args = t_parser.parse_args()


    splits = ["train", "dev", "test"]

    if args.dataset == "CHBMIT" :
        total_participants = 24
    else: 
        raise Exception("dataset not implemented...")


    
    if args.leave_one_out:
        if args.participant_number is None:
            raise Exception("--participant_number is required when --leave_one_out is set.")
        val_participants = test_participants = [args.participant_number]
        train_participants = [ i for i in range(1,total_participants+1) if not i in val_participants]
    else:
        if args.test_participants is None or args.val_participants is None:
            raise Exception.error("--test_participants and --val_participants are required when --leave_one_out is not set.")
        test_participants = random.sample(range(1, total_participants+1), args.test_participants)
        remaining_particpants = [ i for i in range(1,total_participants+1) if not i in test_participants]
        val_participants = random.sample(remaining_particpants, args.val_participants)
        remaining_particpants = [ i for i in remaining_particpants if not i in val_participants]
        train_participants = remaining_particpants

    for split in splits:
        if split == "train":
            if args.dataset == "CHBMIT":
                parser_CHBMIT(args.filtered_data_path, args.final_label_dir, train_participants, args.clip_size, split)
            else: 
                raise Exception("parser not configured for this dataset")
        elif split == "dev":
            if args.dataset == "CHBMIT":        
                parser_CHBMIT(args.filtered_data_path,args.final_label_dir, val_participants, args.clip_size, split)
        else: 
            if args.dataset == "CHBMIT":  
                parser_CHBMIT(args.filtered_data_path, args.final_label_dir, test_participants, args.clip_size, split)