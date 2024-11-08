set -e
GPU_ID=2

python whole_train_baseline_multimodel.py \
    --name whole_train_Track1_Mandarin \
    --checkpoints_dir A_checkpoints/Track1/Mandarin \
    --log_dir A_logs/Track1/Mandarin \
    --feature_dirs datasets/MC-EIU_Track1_Mandarin_Processed/Features/wav2vec-large-c-FRA \
    datasets/MC-EIU_Track1_Mandarin_Processed/Features/resnet50face_FRA \
    datasets/MC-EIU_Track1_Mandarin_Processed/Features/roberta-base-4-FRA \
    --input_feature_dims 512 512 768 \
    --device $GPU_ID