#!/bin/bash
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

participant_number=$1
save_dir=$2
dataset=$3
filtered_data_dir=$4
num_epochs_ssl=$5
input_len=$6

cached_graphs_dir="$REPO_ROOT/cache/graph_cache_${dataset}_ssl"
final_label_dir="$REPO_ROOT/labels/file_markers_${dataset}_ssl"


cleanup() {

    if [ -n "$cached_graphs_dir" ] && [ -d "$cached_graphs_dir" ]; then
        rm -rf $cached_graphs_dir
    fi

    if [ -n "$final_label_dir" ] && [ -d "$final_label_dir" ]; then
        rm -rf $final_label_dir
    fi

}

run_parser() {
    python3 -m scripts.parser_ssl --leave_one_out --participant_number "$participant_number" --dataset "$dataset" --filtered_data_path "$filtered_data_dir" --clip_size "$input_len" --final_label_dir "$final_label_dir"
}

run_training() {
    python3 -m scripts.train_ssl --dataset "$dataset" --input_dir "$filtered_data_dir" --save_dir "$model_save_dir" --num_epochs "$num_epochs_ssl" --labels_dir "$final_label_dir" --max_seq_len "$input_len" --cache_dir "$cached_graphs_dir"
}

main(){
    echo "cleaning up the old cache and labels if found..."
    cleanup
    echo "preparing labels..."
    run_parser 
    echo "starting training..."
    run_training
}

main
