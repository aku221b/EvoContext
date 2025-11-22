SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
timestamp=$(date +%s)
#model_save_dir="$REPO_ROOT/model_checkpoints/ssl_$timestamp"
model_save_dir="$REPO_ROOT/model_checkpoints/ssl_test"
num_partcipants=$1
num_epochs_downstream=$2
num_epochs_ssl=$3
dataset=$4
filtered_data_dir=$5
input_len=$6

run_ssl="$SCRIPT_DIR/run_ssl_leave_one_out.sh"
run_downstream="$SCRIPT_DIR/run_downstream.sh"


main(){
    for i in $(seq 1 $num_partcipants)
    do
        mkdir -p "$model_save_dir/$i"
        echo "Running the pipeline for participant $i..."
        #echo "Starting pretraining pipeline..."
        "$run_ssl" "$i" "$model_save_dir/$i" "$dataset" "$filtered_data_dir" "$num_epochs_ssl" "$input_len" 
        echo "Starting the downstream training..."
        "$run_downstream" "$i" "detection" "$num_epochs_downstream" "$model_save_dir/$i/best.pth.tar" "$filtered_data_dir" "$model_save_dir" "$dataset" "$input_len"
        echo "Trainging done..."
    done
}
main