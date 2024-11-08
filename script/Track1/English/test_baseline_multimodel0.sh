set -e
GPU_ID=0

python test_baseline_multimodel.py \
    --name test_val_best_F1 \
    --checkpoints_dir A_checkpoints/Track1/English \
    --log_dir A_logs/Track1/English \
    --feature_dirs datasets/MC-EIU_Track1_English_Processed/Features/wav2vec-large-c-FRA \
    datasets/MC-EIU_Track1_English_Processed/Features/resnet50face_FRA \
    datasets/MC-EIU_Track1_English_Processed/Features/roberta-base-4-FRA \
    --load_model_path A_checkpoints/Track1/English/whole_train_Track1_English_WF1_2024-11-03_02-49-52_variant_multimodal/train_1 \
    --load_model_prefix best_F1 \
    --input_feature_dims 512 512 768 \
    --no-is_train \
    --device $GPU_ID