## lexical feature extraction
python feature_extraction_main.py generate_transcription_files_asr \
 'data/MC-EIU/Processed/Track1/English/NoAnnotation/Audios' \
  'data/MC-EIU/Processed/Track1/English/NoAnnotation/transcription.csv' \
  'English'

cd feature_extraction/text

python extract_text_embedding_LZ.py --dataset=Track2_Mandarin --feature_level='UTTERANCE' --model_name=roberta-base --gpu=3