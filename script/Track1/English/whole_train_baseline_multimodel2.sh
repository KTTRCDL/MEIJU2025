set -e
GPU_ID=2

# python whole_train_baseline_multimodel.py \
#     --name whole_train_Track1_English_Iuar_attn_head_2 \
#     --checkpoints_dir A_checkpoints/Track1/English \
#     --log_dir A_logs/Track1/English \
#     --feature_dirs datasets/MC-EIU_Track1_English_Processed/Features/wav2vec-large-c-FRA \
#     datasets/MC-EIU_Track1_English_Processed/Features/resnet50face_FRA \
#     datasets/MC-EIU_Track1_English_Processed/Features/roberta-base-4-FRA \
#     --load_model_prefix best_Iuar \
#     --input_feature_dims 512 512 768 \
#     --device $GPU_ID

python whole_train_baseline_multimodel.py \
    --name whole_train_Track1_English_epoch_more_steps \
    --checkpoints_dir A_checkpoints/Track1/English \
    --log_dir A_logs/Track1/English \
    --feature_dirs datasets/MC-EIU_Track1_English_Processed/Features/wav2vec-large-c-FRA \
    datasets/MC-EIU_Track1_English_Processed/Features/resnet50face_FRA \
    datasets/MC-EIU_Track1_English_Processed/Features/roberta-base-4-FRA \
    --load_model_prefix best_F1 \
    --input_feature_dims 512 512 768 \
    --eval_freq 99 \
    --pretrain_niter 30 \
    --pretrain_niter_decay 60 \
    --niter 25 \
    --niter_decay 55 \
    --device $GPU_ID