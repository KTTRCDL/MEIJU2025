set -e

gpu_id=3

python train_two_stage_pretrain_feature_baseline.py --name my_baseline \
    --log_dir logs --checkpoint_dir checkpoints --gpu_id $gpu_id \
    --model baseline --input_feature_dims 512 1024 768 \
    --emotion_network_names LSTMEncoder TextCNN LSTMEncoder \
    --emotion_network_embed_dims 128 128 128 \
    --intent_network_names LSTMEncoder TextCNN LSTMEncoder \
    --intent_network_embed_dims 128 128 128 \
    --hidden_size 128 \
    --dataset_filelist_dir filelists/Track1_English_origin \
    --feature_dirs datasets/MC-EIU_Track1_English_Processed/Features/wav2vec-large-c-FRA \
     datasets/MC-EIU_Track1_English_Processed/Features/manet_FRA \
     datasets/MC-EIU_Track1_English_Processed/Features/roberta-base-4-FRA \
    --niter 20 --niter_decay 40 --print_freq 10 --batch_size 32 
    