set -e
GPU_ID=1

python whole_train_baseline_multimodel.py \
    --name train_Track2_English_baseline \
    --dataset_filelist_dir filelists/Track2_English \
    --checkpoints_dir A_checkpoints/Track2/English \
    --log_dir A_logs/Track2/English \
    --feature_dirs datasetsdata/MC-EIU/Processed/Track2/English/features/wav2vec-large-c-FRA \
    datasetsdata/MC-EIU/Processed/Track2/English/features/resnet50face_FRA \
    datasetsdata/MC-EIU/Processed/Track2/English/features/roberta-base-4-FRA \
    --input_feature_dims 512 512 768 \
    --load_model_prefix best_Iuar \
    --cls_dropout 0.5 \
    --eval_freq 900 \
    --save_freq 900 \
    --device $GPU_ID