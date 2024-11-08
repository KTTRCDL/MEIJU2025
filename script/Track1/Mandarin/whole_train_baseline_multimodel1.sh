set -e
GPU_ID=1

python whole_train_baseline_multimodel.py \
    --name whole_train_Track1_Mandarin_WF1 \
    --checkpoints_dir A_checkpoints/Track1/Mandarin \
    --log_dir A_logs/Track1/Mandarin \
    --dataset_filelist_dir filelists/Track1_Mandarin_origin \
    --feature_dirs datasetsdata/MC-EIU/Processed/Track1/Mandarin/Annotation/features/wav2vec-large-c-FRA \
    datasetsdata/MC-EIU/Processed/Track1/Mandarin/Annotation/features/resnet50face_FRA \
    datasetsdata/MC-EIU/Processed/Track1/Mandarin/Annotation/features/chinese-hubert-large-FRA \
    --input_feature_dims 512 512 1024 \
    --device $GPU_ID