set -e
GPU_ID=3

python whole_train_baseline_multimodel.py \
    --name whole_train_Track1_Mandarin \
    --checkpoints_dir A_checkpoints/Track1/Mandarin \
    --log_dir A_logs/Track1/Mandarin \
    --device $GPU_ID