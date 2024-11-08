from opts.get_opts import Options
from utils.logger import get_logger, ResultRecorder
from os.path import join
from data import create_dataset_with_args
from models import create_model
from tqdm import tqdm
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import os
import numpy as np
import math
import pandas as pd

def eval_model(model, dataset, logger, steps, opt):
    model.eval()
    total_emo_pred = []
    total_emo_label = []
    total_int_pred = []
    total_int_label = []

    # model.test()
    for data in dataset:
        model.set_input(data)
        model.test()
        emo_pred = model.emo_pred.argmax(dim=1).detach().cpu().numpy()
        int_pred = model.int_pred.argmax(dim=1).detach().cpu().numpy()
        emo_label = data['emotion_label']
        int_label = data['intent_label']

        total_emo_pred.append(emo_pred)
        total_emo_label.append(emo_label)
        total_int_pred.append(int_pred)
        total_int_label.append(int_label)

    total_emo_pred = np.concatenate(total_emo_pred)
    total_emo_label = np.concatenate(total_emo_label)
    total_int_pred = np.concatenate(total_int_pred)
    total_int_label = np.concatenate(total_int_label)
    
    if 'Track1' in opt.log_dir:
        average_method = 'weighted'
    elif 'Track2' in opt.log_dir:
        average_method = 'micro'
    else:
        average_method = 'weighted'

    emo_acc = accuracy_score(total_emo_label, total_emo_pred)
    # emo_uar = recall_score(total_emo_label, total_emo_pred, average=average_method)
    emo_uar = recall_score(total_emo_label, total_emo_pred, average='macro')
    int_acc = accuracy_score(total_int_label, total_int_pred)
    # int_uar = recall_score(total_int_label, total_int_pred, average=average_method)
    int_uar = recall_score(total_int_label, total_int_pred, average='macro')
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)
    int_cm = confusion_matrix(total_int_label, total_int_pred)
    f1_emo = f1_score(total_emo_label, total_emo_pred, average=average_method)
    f1_int = f1_score(total_int_label, total_int_pred, average=average_method)
    F1 = 2 * f1_emo * f1_int / (f1_emo + f1_int)
    model.train()

    logger.info('Step {}'.format(steps) + ' Emotion Acc {:.4f} UAR {:.4f} F1 {:.4f}'.format(emo_acc, emo_uar, f1_emo))
    logger.info('Step {}'.format(steps) + ' Intent Acc {:.4f} UAR {:.4f} F1 {:.4f}'.format(int_acc, int_uar, f1_int))
    logger.info('Step {}'.format(steps) + ' Joint F1 {:.4f}'.format(F1))
    logger.info('\n{}'.format(emo_cm))
    logger.info('\n{}'.format(int_cm))

    return emo_acc, emo_uar, f1_emo, int_acc, int_uar, f1_int, F1

def predict(opt, output_file_name='submission.csv'):
    opt.is_train = False
    logger = get_logger(join(opt.log_dir, 'predict'), opt.model)
    option.log_save_options(logger)
    dataset_test = create_dataset_with_args(opt, set_name='test')
    dataset_test_size = len(dataset_test)
    logger.info('The number of test samples = %d' % dataset_test_size)
    model = create_model(opt, logger=logger)
    model.setup()
    model.eval()
    total_filename = []
    total_emo_pred = []
    total_int_pred = []

    for data in dataset_test:
        total_filename.append(data['filename'])
        model.set_input(data)
        model.test()
        emo_pred = model.emo_pred.argmax(dim=1).detach().cpu().numpy()
        int_pred = model.int_pred.argmax(dim=1).detach().cpu().numpy()
        total_emo_pred.append(emo_pred)
        total_int_pred.append(int_pred)

    total_emo_pred = np.concatenate(total_emo_pred)
    total_int_pred = np.concatenate(total_int_pred)
    total_filename = np.concatenate([np.array(batch_filename) for batch_filename in total_filename])

    emotions = ['happy', 'surprise', 'sad', 'disgust', 'anger', 'fear', 'neutral']
    intents = ['questioning', 'agreeing', 'acknowledging', 'encouraging', 'consoling', 'suggesting', 'wishing', 'neutral']

    # df = pd.DataFrame({'filename': total_filename, 'emo_pred': total_emo_pred, 'int_pred': total_int_pred})
    df = pd.DataFrame({'filename': total_filename, 'emo_pred': [emotions[i] for i in total_emo_pred], 'int_pred': [intents[i] for i in total_int_pred]})
    df.to_csv(join(opt.checkpoints_dir, output_file_name), index=False)
    print(f'Predict done! save the {output_file_name} in {opt.checkpoints_dir}')
    logger.info(f'Predict done! save the {output_file_name} in {opt.checkpoints_dir}')

    model.train()
    opt.is_train = True

if __name__ == '__main__':
    option = Options()
    opt = option.parse()

    opt.load_model_path='A_checkpoints/Track2/English/train_Track2_English_baseline_best_F1_2024-11-04_23-11-51_variant_multimodal/train_1'
    opt.load_model_prefix='best_emo'
    predict(opt, output_file_name='submission_emo.csv')

    opt.load_model_prefix='best_int'
    predict(opt, output_file_name='submission_int.csv')
    # logger = get_logger(join(opt.log_dir, f'test_{opt.load_model_path.split("/")[-1]}'), opt.model)

    # model = create_model(opt, logger=logger)
    # model.setup()

    # dataset_train, dataset_val = create_dataset_with_args(opt, set_name=['train', 'valid'])

    # eval_model(model, dataset_val, logger, -1, opt)
