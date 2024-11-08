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

def eval_model(model, dataset, logger, steps):
    model.eval()
    total_emo_pred = []
    total_emo_label = []
    total_int_pred = []
    total_int_label = []

    for i, data in enumerate(dataset):
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
    
    emo_acc = accuracy_score(total_emo_label, total_emo_pred)
    emo_uar = recall_score(total_emo_label, total_emo_pred, average='macro')
    int_acc = accuracy_score(total_int_label, total_int_pred)
    int_uar = recall_score(total_int_label, total_int_pred, average='macro')
    emo_cm = confusion_matrix(total_emo_label, total_emo_pred)
    int_cm = confusion_matrix(total_int_label, total_int_pred)
    f1_emo = f1_score(total_emo_label, total_emo_pred, average='macro')
    f1_int = f1_score(total_int_label, total_int_pred, average='macro')
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

if __name__ == '__main__':
    option = Options()
    opt = option.parse()
    result_file_name = 'result.tsv'
    if opt.load_model_path == 'None':
        result_file_name = 'pretrain_result.tsv'
    result_recorder = ResultRecorder(join(opt.checkpoints_dir, result_file_name), total_cv=opt.total_cv)
    for cv in range(1, opt.total_cv + 1):
        logger = get_logger(join(opt.log_dir, str(cv)), opt.model)
        option.log_save_options(logger)

        # for different cv, we need to change the checkpoints_dir
        opt.checkpoints_dir = join(opt.checkpoints_dir, str(cv))
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
        best_loss = 100
        curr_metric = 0

        # for epoch in range(1, opt.niter + opt.niter_decay + 1):
        for epoch in range(opt.epoch_count, opt.epoch_count + opt.niter + opt.niter_decay):
            for i, data in enumerate(tqdm(dataset_train, desc=f'Epoch:{epoch} CV:{cv} Step:{total_steps}', total=math.ceil(dataset_train_size/opt.batch_size))):
                total_steps += 1
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.loss_freq == 0:
                    losses = model.get_current_losses()
                    logger.info('Cur steps {}'.format(total_steps) + ' loss ' +
                                ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
                    
                if total_steps % opt.eval_freq == 0:
                    emo_acc, emo_uar, f1_emo, int_acc, int_uar, f1_int, F1 = eval_model(model, dataset_val, logger, total_steps)
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
                    # model.save_networks(str(total_steps))
                    logger.info('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                
                clean_checkpoints(opt.checkpoints_dir, best_eval_steps)

            model.update_learning_rate()

        result_recorder.write_result_to_tsv({'emo_metric': best_emo_metric, 
                                             'int_metric': best_int_metric, 
                                             'joint_metric': best_metric}, cv)
        
        # clean_checkpoints(opt.checkpoints_dir, best_eval_steps)
        logger.info('End of training cv %d' % cv)

        if len(str(cv)) == 1:
            opt.checkpoints_dir = opt.checkpoints_dir[:-1]
        elif len(str(cv)) == 2:
            opt.checkpoints_dir = opt.checkpoints_dir[:-2]
