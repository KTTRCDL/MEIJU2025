set -e
GPU_ID=3

python whole_train_baseline_multimodel.py \
    --name train_Track2_Mandarin \
    --dataset_filelist_dir filelists/Track2_Mandarin \
    --checkpoints_dir A_checkpoints/Track2/Mandarin \
    --log_dir A_logs/Track2/Mandarin \
    --feature_dirs data/MC-EIU/Processed/Track2/Mandarin/features/chinese-hubert-large-FRA \
    data/MC-EIU/Processed/Track2/Mandarin/features/resnet50face_FRA \
    data/MC-EIU/Processed/Track2/Mandarin/features/Baichuan-13B-Base-4-prompt-FRA \
    --input_feature_dims 1024 512 5120 \
    --load_model_prefix best_Iuar \
    --eval_freq 900 \
    --save_freq 900 \
    --device $GPU_ID