set -e
GPU_ID=0

python test_baseline_multimodel.py \
    --name Track2_English_predict_emo_int_sep \
    --dataset_filelist_dir filelists/Track2_English \
    --checkpoints_dir A_checkpoints/Track2/English \
    --log_dir A_logs/Track2/English \
    --feature_dirs datasetsdata/MC-EIU/Processed/Track2/English/features/wav2vec-large-c-FRA \
    datasetsdata/MC-EIU/Processed/Track2/English/features/resnet50face_FRA \
    datasetsdata/MC-EIU/Processed/Track2/English/features/roberta-base-4-FRA \
    --input_feature_dims 512 512 768 \
    --cls_dropout 0.5 \
    --eval_freq 979 \
    --save_freq 979 \
    --device $GPU_ID