############################################################################
################# step2: multimodal feature extraction #####################
# you can also extract utterance-level features setting --feature_level='UTTERANCE'#
############################################################################
cd feature_extraction/visual

### for building openface in ubuntu, please reference https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation and https://blog.csdn.net/m0_47623548/article/details/138171121
# !!! attention !!! use g++-9 and gcc-9 to compile openface, otherwise will cause error
# for me, my ubuntu version is Ubuntu 20.04.6 LTS

python extract_openface_ubuntu_multiprocess.py --dataset=Track2_Mandarin --type=videoOne

CUDA_VISIBLE_DEVICES=2 python -u extract_vision_huggingface.py --dataset=Track2_Mandarin --feature_level='UTTERANCE' --model_name='videomae-large'