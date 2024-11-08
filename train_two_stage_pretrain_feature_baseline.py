from opts.get_option import Options
import os
from utils.logger import get_logger
from data.multimodal_dataset import create_dataloader_with_args
from models import create_model
import time
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

def train(opt, train_dataset, val_dataset, model: torch.nn.Module, logger):
    device = torch.device(f'cuda:{opt.gpu_id}') if int(opt.gpu_id) >= 0 else torch.device('cpu')
    model = model.to(device)
    model.train()
    total_iters = 0
    best_F1_JRBM = 0
    best_eval_epoch = -1

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_focal = torch.nn.CrossEntropyLoss()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(train_dataset):
            iter_start_time = time.time()
            total_iters += 1
            epoch_iter += opt.batch_size

            emo_label = data['emotion_label'].to(device)
            int_label = data['intent_label'].to(device)
            features = data['features']
            for j in range(len(features)):
                features[j] = features[j].to(device)

            # data = [data[key].to(device) for key in data if key != 'filename']
            emo_logits, int_logits, emo_logits_fusion, int_logits_fusion = model(features)
            optimizer.zero_grad()
            loss_emo_CE = criterion_ce(emo_logits, emo_label)
            loss_int_CE = criterion_ce(int_logits, int_label)
            loss_EmoF_CE = criterion_focal(emo_logits_fusion, emo_label)
            loss_IntF_CE = criterion_focal(int_logits_fusion, int_label)
            loss = loss_emo_CE + loss_int_CE + loss_EmoF_CE + loss_IntF_CE
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if total_iters % opt.print_freq == 0:
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info(f'Epoch: {epoch}, Iter: {total_iters}, '
                            f'Loss: {loss.item()}, Time: {t_comp}, '
                            f'loss_emo_CE: {loss_emo_CE.item()}, loss_int_CE: {loss_int_CE.item()}, '
                            f'loss_EmoF_CE: {loss_EmoF_CE.item()}, loss_IntF_CE: {loss_IntF_CE.item()}')

        if epoch % opt.save_epoch_freq == 0:
            logger.info(f'saving the model at the end of epoch {epoch}')
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, f'{epoch}_{total_iters}.pth'))
        
        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            
        int_acc, int_uar, int_cm, emo_acc, emo_uar, emo_cm = evaluate(opt, val_dataset, model)
        logger.info('Val result of epoch %d / %d emo_acc %.4f emo_uar %.4f int_acc %.4f int_uar %.4f' % (
            epoch, opt.niter + opt.niter_decay, int_acc, int_uar, emo_acc, emo_uar))
        logger.info('\n{}'.format(emo_cm))
        logger.info('\n{}'.format(int_cm))

        F1_emo = 2 * emo_acc * emo_uar / (emo_acc + emo_uar + 1e-7)
        F1_int = 2 * int_acc * int_uar / (int_acc + int_uar + 1e-7)
        F1_JRBM = 2 * (F1_emo * F1_int) / (F1_emo + F1_int + 1e-7)
        if F1_JRBM > best_F1_JRBM:
            best_F1_JRBM = F1_JRBM
            best_eval_epoch = epoch
            logger.info(f'Best F1_JRBM: {best_F1_JRBM} at epoch {best_eval_epoch}')
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, 'best.pth'))

    # clean_checkpoint(opt.checkpoint_dir, 'best.pth')

def evaluate(opt, val_dataset, model: torch.nn.Module):
    device = torch.device(f'cuda:{opt.gpu_id}') if int(opt.gpu_id) >= 0 else torch.device('cpu')
    model.eval()
    total_emo_pred = []
    total_emo_label = []
    total_int_pred = []
    total_int_label = []

    for i, data in enumerate(val_dataset):
        # data = [data[key].to(device) for key in data if key != 'filename']
        features = data['features']
        for j in range(len(features)):
            features[j] = features[j].to(device)
        emo_logits, int_logits, _, _ = model(features)
        emo_pred = F.softmax(emo_logits, dim=-1).argmax(dim=1).detach().cpu().numpy()
        int_pred = F.softmax(int_logits, dim=-1).argmax(dim=1).detach().cpu().numpy()

        emo_lebel = data['emotion_label'].detach().cpu().numpy()
        int_label = data['intent_label'].detach().cpu().numpy()

        total_emo_pred.append(emo_pred)
        total_emo_label.append(emo_lebel)
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

    return int_acc, int_uar, int_cm, emo_acc, emo_uar, emo_cm

def clean_checkpoint(checkpoint_dir, best_model_name):
    for file in os.listdir(checkpoint_dir):
        if file != best_model_name:
            os.remove(os.path.join(checkpoint_dir, file))

if __name__ == '__main__':
    opt = Options().parse()
    # cvNo indicates the cross-validation number
    logger_path = opt.log_dir
    os.makedirs(logger_path, exist_ok=True)

    # !!!!!!!!!! TODO: what is opt.corpus_name?
    # total_cv = 3 if opt.corpus_name != 'MSP' else 12

    # result_recorder = ResultRecorder(os.path.join(opt.log_dir, opt.name, 'result.tsv'),
    #                                  total_cv=total_cv)
    # !!!!!!!!! modify, suffix -> opt.model
    logger = get_logger(logger_path, opt.model)

    dataset_train = create_dataloader_with_args(opt, set_name='train')
    dataset_val = create_dataloader_with_args(opt, set_name='valid')
    if opt.has_test:
        dataset_test = create_dataloader_with_args(opt, set_name='test')

    logger.info('The number of training samples = %d' % len(dataset_train))
    logger.info('The number of validation samples = %d' % len(dataset_val))
    if opt.has_test:
        logger.info('The number of testing samples = %d' % len(dataset_test))

    # create a model given opt.model and other options
    model = create_model(opt, logger)
    train(opt, dataset_train, dataset_val, model, logger)
