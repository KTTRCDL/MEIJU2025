########################################################################
######################## step0: environment preprocess #################
# 手动输入以下指令来配置环境 #
########################################################################
#conda create --name MER_test python=3.8 -y
#
#conda activate MER_test

#conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
#
#pip install scikit-image fire opencv-python tqdm matplotlib pandas soundfile wenetruntime fairseq==0.9.0 numpy==1.23.5 transformers paddlespeech pytest-runner paddlepaddle whisper -i https://pypi.tuna.tsinghua.edu.cn/simple


########################################################################
######################## step1: dataset preprocess #####################
########################################################################
### Processing training set and validation set
# !!!!!!!!!!!!!!!!!!!!!! origin csv data header name is not right, normalize_dataset_format need to be fixed: Subtitle -> sentence, FileName -> name (or change it in extract_text_embedding_LZ.py)
# !!!!!!!!!!!!!!!! not correct, validation_transcription.csv Subtitle(sentence) not gathered, need to be fixed
python feature_extraction_main.py normalize_dataset_format --data_root='data/MC-EIU/Track1/English' --save_root='data/MC-EIU/Processed/Track1/English' --track=1
python feature_extraction_main.py normalize_dataset_format --data_root='data/MC-EIU/Track1/Mandarin' --save_root='data/MC-EIU/Processed/Track1/Mandarin' --track=1

python feature_extraction_main.py normalize_dataset_format --data_root='data/MC-EIU/Track2/English' --save_root='data/MC-EIU/Processed/Track2/English' --track=2
python feature_extraction_main.py normalize_dataset_format --data_root='data/MC-EIU/Track2/Mandarin' --save_root='data/MC-EIU/Processed/Track2/Mandarin' --track=2
