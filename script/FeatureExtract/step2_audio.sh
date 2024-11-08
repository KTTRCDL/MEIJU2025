## acoustic feature extraction
cd feature_extraction/audio
CUDA_VISIBLE_DEVICES=0 python -u extract_transformers_embedding.py  --dataset=Track2_Mandarin --feature_level='UTTERANCE' --model_name='chinese-hubert-large'
CUDA_VISIBLE_DEVICES=0 python -u extract_transformers_embedding.py  --dataset=Track2_English --feature_level='UTTERANCE' --model_name='chinese-hubert-large'

CUDA_VISIBLE_DEVICES=0 python -u extract_transformers_embedding.py  --dataset=Track2_English --feature_level='FRAME' --model_name='chinese-hubert-large'
CUDA_VISIBLE_DEVICES=0 python -u extract_transformers_embedding.py  --dataset=Track2_Mandarin --feature_level='FRAME' --model_name='chinese-hubert-large'
