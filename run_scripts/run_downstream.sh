#!/bin/bash
test_participant=$1
task=$2
epochs_per_iteration=$3
load_model_path=$4
filtered_data=$5
save_dir=$6
dataset=$7
input_len=$8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cached_graphs_dir="$REPO_ROOT/cache/graph_cache_chb_supervised"
final_label_dir="$REPO_ROOT/labels/file_markers_$dataset"

cleanup() {
    if [ -n "$cached_graphs_dir" ] && [ -d "$cached_graphs_dir" ]; then
        rm -rf $cached_graphs_dir
    fi

    if [ -n "$final_label_dir" ] && [ -d "$final_label_dir" ]; then
        rm -rf $final_label_dir
    fi
}

run_parser() {
    python3 -m scripts.parser --test_participants "$test_participant" --task "$task" --clip_size "$input_len" --dataset "$dataset" --filtered_data_path "$filtered_data" --labels_final_base_path "$final_label_dir"
}

run_training() {
    # python3 -m train --dataset "CHBMIT" --input_dir "$filtered_data" --raw_data_dir "$raw_data_dir" --save_dir "$save_dir" --task "$task" --model_name "$model_name" --num_epochs "$epochs_per_iteration" --load_model_path "$load_model_path"
     python3 -m scripts.train --dataset "CHBMIT" --input_dir "$filtered_data" --save_dir "$save_dir" --task "$task" --num_epochs "$epochs_per_iteration" --load_model_path "$load_model_path" --final_label_dir "$final_label_dir" --cached_graphs_dir "$cached_graphs_dir" --fine_tune 
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
