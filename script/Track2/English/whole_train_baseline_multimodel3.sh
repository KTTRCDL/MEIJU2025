set -e
GPU_ID=3

# python whole_train_baseline_multimodel.py \
#     --name train_Track2_English_baseline_best_F1 \
#     --dataset_filelist_dir filelists/Track2_English \
#     --checkpoints_dir A_checkpoints/Track2/English \
#     --log_dir A_logs/Track2/English \
#     --feature_dirs data/MC-EIU/Processed/Track2/English/features/wav2vec-large-c-FRA \
#     data/MC-EIU/Processed/Track2/English/features/resnet50face_FRA \
#     data/MC-EIU/Processed/Track2/English/features/roberta-base-4-FRA \
#     --input_feature_dims 512 512 768 \
#     --load_model_prefix best_F1 \
#     --cls_dropout 0.5 \
#     --eval_freq 979 \
#     --save_freq 979 \
#     --device $GPU_ID

python whole_train_baseline_multimodel.py \
    --name train_Track2_English_hidden_128_ICL \
    --model variant_multimodal_hidden \
    --dataset_filelist_dir filelists/Track2_English \
    --checkpoints_dir A_checkpoints/Track2/English \
    --log_dir A_logs/Track2/English \
    --feature_dirs data/MC-EIU/Processed/Track2/English/features/wav2vec-large-c-FRA \
    data/MC-EIU/Processed/Track2/English/features/resnet50face_FRA \
    data/MC-EIU/Processed/Track2/English/features/roberta-base-4-FRA \
    data/MC-EIU/Processed/Track2/English/features/chinese-hubert-large-FRA \
    data/MC-EIU/Processed/Track2/English/features/clip-vit-large-patch14-FRA \
    data/MC-EIU/Processed/Track2/English/features/manet_FRA \
    data/MC-EIU/Processed/Track2/English/features/wav2vec-large-z-FRA \
    --embed_methods maxpool maxpool NULL maxpool maxpool NULL maxpool \
    --emotion_network_names LSTMEncoder LSTMEncoder TextCNN LSTMEncoder LSTMEncoder TextCNN LSTMEncoder \
    --intent_network_names LSTMEncoder LSTMEncoder TextCNN LSTMEncoder LSTMEncoder TextCNN LSTMEncoder \
    --input_feature_dims 512 512 768 1024 768 1024 512 \
    --embed_dims 128 128 128 256 128 256 128 \
    --load_model_prefix best_F1 \
    --use_ICL \
    --eval_freq 245 \
    --save_freq 245 \
    --pretrain_niter 40 --pretrain_niter_decay 80 \
    --niter 30 --niter_decay 90 \
    --batch_size 128 \
    --device $GPU_ID