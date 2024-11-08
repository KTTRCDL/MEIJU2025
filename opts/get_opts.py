import argparse
import os
import torch
import models
import data
import json
import time
import numpy as np
import random

# def str2bool(v):
#     """string to boolean"""
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
    
class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--name', type=str, required=True,
                            help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--device', type=str, default='0', help='device: e.g. 0. use -1 for CPU, unsupport multiple GPUs')
        parser.add_argument('--checkpoints_dir', type=str, default='A_checkpoints', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='A_logs', help='logs are saved here')
        parser.add_argument('--is_train', default=True, action=argparse.BooleanOptionalAction, 
                            help='train or test (for model init), default: True (Train), use --no-is_train for test')
        parser.add_argument('--seed', type=int, default=42, help='random seed')
        parser.add_argument('--total_cv', type=int, default=3, help='total cross validation')

        # model basic parameters
        parser.add_argument('--model', type=str, default='variant_multimodal', choices=['baseline', 'variant_multimodal', 'variant_multimodal_hidden'],
                            help='chooses which model to use. [baseline, variant_multimodel], default: variant_multimodal')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--load_model_path', type=str, default='None', 
                            help='where to load pretrained encoder network or the whole trained model, if None, train from scratch (means start pretrain)')
        parser.add_argument('--load_model_prefix', type=str, default='best_F1',
                            help='prefix of pretrained model or the whole trained model, e.g. best_F1, best_emo, best_int ...')

        
        # dataset basic parameters
        parser.add_argument('--dataset_name', type=str, default='variant_multimodal', choices=['baseline_multimodal', 'variant_multimodal'],
                            help='chooses which dataset to use. [baseline_multimodal, variant_multimodal], default: variant_multimodal')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--serial_batches', default=False, action=argparse.BooleanOptionalAction, 
                            help='if true, takes images in order to make batches, otherwise takes them randomly, default: False')
        parser.add_argument('--drop_last', default=False, action=argparse.BooleanOptionalAction,
                            help='drop the last batch if it is not complete, default: False')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        
        # additional parameters
        parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction, help='if specified, print more debugging information')

        # training parameter
        parser.add_argument('--pretrain_niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--pretrain_niter_decay', type=int, default=40,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--niter', type=int, default=15, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=45,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--loss_freq', type=int, default=1, help='loss frequency')
        parser.add_argument('--eval_freq', type=int, default=100, help='evaluation frequency')
        parser.add_argument('--save_freq', type=int, default=100, help='save frequency')
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr_policy', type=str, default='linear',
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations, for step lr policy')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        
        # get the basic options
        opt, _ = parser.parse_known_args()

        # model-related parser options
        model_name = opt.model
        if opt.verbose:
            print(f'load {model_name} option')
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser)

        # dataset-related parser options
        dataset_name = opt.dataset_name
        if opt.verbose:
            print(f'load {dataset_name} option')
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser)

        self.parser = parser
        return parser.parse_args()

    def log_save_options(self, logger):
        """log and save options"""
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        # save to the disk
        os.makedirs(self.opt.checkpoints_dir, exist_ok=True)
        if self.opt.load_model_path == 'None':
            file_name = os.path.join(self.opt.checkpoints_dir, 'pretrain_opt.txt')
        else:
            file_name = os.path.join(self.opt.checkpoints_dir, 'train_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        # log to the logger
        logger.info(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        
        opt.name = opt.name + '_' + str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))) \
                    + '_' + str(opt.model)
        opt.log_dir = os.path.join(opt.log_dir, opt.name)
        opt.checkpoints_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(opt.log_dir, exist_ok=True)
        os.makedirs(opt.checkpoints_dir, exist_ok=True)

        if opt.verbose:
            print('------------ Options -------------')
            print(json.dumps(vars(opt), indent=4))
            print('-------------- End ----------------')
        
        opt.device = torch.device('cuda:{}'.format(opt.device) if torch.cuda.is_available() and opt.device != '-1' else 'cpu')

        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        np.random.seed(opt.seed)
        random.seed(opt.seed)



        self.opt = opt
        return self.opt
        
