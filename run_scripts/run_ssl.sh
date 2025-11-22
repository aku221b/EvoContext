#!/bin/bash
test_participants=$1
val_participants=$2
# create_fresh_labels=$3
raw_data_dir="/Users/aksagaga/Desktop/Personal_Learning/chb-mit-scalp-eeg-database-1.0.0"
save_dir="/Users/aksagaga/Desktop/Personal_Learning/logs_ssl"
filtered_data="/Users/aksagaga/Desktop/Personal_Learning/STAGSMOTE/data/filtered_data"
cached_graphs_dir="/Users/aksagaga/Desktop/Personal_Learning/STAGSMOTE/graph_cache_chb_ssl"
final_label_dir="/Users/aksagaga/Desktop/Personal_Learning/STAGSMOTE/data/file_markers_chb_mit_ssl"


cleanup() {

    if [ -n "$cached_graphs_dir" ] && [ -d "$cached_graphs_dir" ]; then
        rm -rf $cached_graphs_dir
    fi

    if [ -n "$final_label_dir" ] && [ -d "$final_label_dir" ]; then
        rm -rf $final_label_dir
    fi

}

run_parser() {
    python3 /Users/aksagaga/Desktop/Personal_Learning/STAGSMOTE/data/parser_ssl.py --test_participants "$test_participants" --val_participants "$val_participants"
}

run_training() {
    python3 -m train_ssl --dataset "CHBMIT" --input_dir "$filtered_data" --raw_data_dir "$raw_data_dir" --save_dir "$save_dir" 
}

main(){
    echo "cleaning up the old cache..."
    cleanup
    echo "preparing labels..."
    run_parser 
    echo "starting training..."
    run_training
}

main
