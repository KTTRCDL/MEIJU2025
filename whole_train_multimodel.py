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

    emo_acc = accuracy_score(total_emo_label, total_emo_pred)
    emo_uar = recall_score(total_emo_label, total_emo_pred, average=average_method)
    int_acc = accuracy_score(total_int_label, total_int_pred)
    int_uar = recall_score(total_int_label, total_int_pred, average=average_method)
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

def clean_checkpoints(checkpoint_dir, best_eval_steps):
    model_prefix_list = ['best_emo', 'best_int', 'best_F1']
    for file in os.listdir(checkpoint_dir):
        if not ((any([file.startswith(str(step) + '_' + model_prefix_list[i]) for i, step in enumerate(best_eval_steps)]) and file.endswith('.pth'))
                or (any([file.startswith(f'pretrained_{step}_{model_prefix_list[i]}') for i, step in enumerate(best_eval_steps)]) and file.endswith('.pth'))
                or 'latest' in file):
            os.remove(join(checkpoint_dir, file))

def train(opt):
    if opt.load_model_path == 'None':
        result_file_name = 'pretrain_result.tsv'
    else:
        result_file_name = 'train_result.tsv'
    result_recorder = ResultRecorder(join(opt.checkpoints_dir, result_file_name), total_cv=opt.total_cv)
    for cv in range(1, opt.total_cv + 1):
        if opt.load_model_path == 'None':
            logger = get_logger(join(opt.log_dir, f'pretrain_{cv}'), opt.model)
        else:
            logger = get_logger(join(opt.log_dir, f'train_{cv}'), opt.model)
        option.log_save_options(logger)

        # for different cv, we need to change the checkpoints_dir
        if opt.load_model_path == 'None':
            opt.checkpoints_dir = join(opt.checkpoints_dir, f'pretrain_{cv}')
        else:
            opt.checkpoints_dir = join(opt.checkpoints_dir, f'train_{cv}')

        os.makedirs(opt.checkpoints_dir, exist_ok=True)
        
        dataset_train, dataset_val = create_dataset_with_args(opt, set_name=['train', 'valid'])
        
        dataset_train_size = len(dataset_train)
        dataset_val_size = len(dataset_val)
        logger.info('The number of training samples = %d' % dataset_train_size)
        logger.info('The number of validation samples = %d' % dataset_val_size)

        model = create_model(opt, logger=logger)
        model.setup()

        total_steps = 0
        best_emo_metric, best_int_metric, best_metric = 0, 0, 0
        best_eval_steps = [-1, -1, -1]
        niter = opt.niter + opt.niter_decay if opt.load_model_path != 'None' else opt.pretrain_niter + opt.pretrain_niter_decay

        for epoch in range(opt.epoch_count, opt.epoch_count + niter):
            for data in tqdm(dataset_train, desc=f'Epoch:{epoch} CV:{cv} Step:{total_steps}', total=math.ceil(dataset_train_size/opt.batch_size)):
                total_steps += 1
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.loss_freq == 0:
                    losses = model.get_current_losses()
                    logger.info('Cur steps {}'.format(total_steps) + ' loss ' +
                                ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
                    
                if total_steps % opt.eval_freq == 0:
                    emo_acc, emo_uar, f1_emo, int_acc, int_uar, f1_int, F1 = eval_model(model, dataset_val, logger, total_steps, opt=opt)
                    if f1_emo > best_emo_metric:
                        best_emo_metric = f1_emo
                        logger.info('Saving the best emotion model at step {} with f1_emo {}'.format(total_steps, f1_emo))
                        model.save_networks(f'{total_steps}_best_emo')
                        best_eval_steps[0] = total_steps
                    if f1_int > best_int_metric:
                        best_int_metric = f1_int
                        logger.info('Saving the best intent model at step {} with f1_int {}'.format(total_steps, f1_int))
                        model.save_networks(f'{total_steps}_best_int')
                        best_eval_steps[1] = total_steps
                    if F1 > best_metric:
                        best_metric = F1
                        model.save_networks(f'{total_steps}_best_F1')
                        logger.info('Saving the best F1 model at step {} with F1 {}'.format(total_steps, F1))
                        best_eval_steps[2] = total_steps

                if total_steps % opt.save_freq == 0:
                    model.save_networks('latest')
                    logger.info('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                
                clean_checkpoints(opt.checkpoints_dir, best_eval_steps)

            model.update_learning_rate()

        result_recorder.write_result_to_tsv({'emo_metric': best_emo_metric, 
                                             'int_metric': best_int_metric, 
                                             'joint_metric': best_metric}, cv)
        
        # clean_checkpoints(opt.checkpoints_dir, best_eval_steps)
        logger.info('End of pretraining cv %d' % cv)

        opt.checkpoints_dir = os.path.dirname(opt.checkpoints_dir)

def find_and_set_load_model_path_for_next(opt):
    load_model_prefix_2_result_record_metric_name = {'best_emo': 'emo_metric', 'best_int': 'int_metric', 'best_F1': 'joint_metric'}
    # judge whether pretrain -> train or train -> predict
    if opt.load_model_path == 'None':
        result_file_name = 'pretrain_result.tsv'
    else:
        result_file_name = 'train_result.tsv'
    result_df = pd.read_csv(join(opt.checkpoints_dir, result_file_name), sep='\t')
    # pd.read_csv('checkpoints/train_Track2_Mandarin_pretrain_use_ICL_2024-11-02_03-43-48_variant_multimodal/pretrain_result.tsv', sep='\t')
    prefix_result = result_df[load_model_prefix_2_result_record_metric_name[opt.load_model_prefix]][:-1]
    # choose the index of the largest pretrain_prefix_result, 1 is the shift
    best_prefix_cv = prefix_result.idxmax() + 1
    # set the load_model_path for the next step
    if opt.load_model_path == 'None':
        opt.load_model_path = join(opt.checkpoints_dir, f'pretrain_{best_prefix_cv}')
    else:
        opt.load_model_path = join(opt.checkpoints_dir, f'train_{best_prefix_cv}')

def predict(opt):
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
    df.to_csv(join(opt.checkpoints_dir, 'submission.csv'), index=False)
    print(f'Predict done! save the submission.csv in {opt.checkpoints_dir}')
    logger.info(f'Predict done! save the submission.csv in {opt.checkpoints_dir}')

    model.train()
    opt.is_train = True

if __name__ == '__main__':
    option = Options()
    opt = option.parse()

    # pretrain model
    train(opt)
    
    # get the best pretrain model for train
    find_and_set_load_model_path_for_next(opt)

    # train model
    train(opt)

    # get the best train model for predict
    find_and_set_load_model_path_for_next(opt)
    
    # predict
    predict(opt)

