import argparse
import os
import torch
import models
import json
import time

class Options():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
    
    def initialize(self, parser):
        # basic parameters
        parser.add_argument('--name', type=str, required=True,
                            help='name of the experiment. It decides where to store samples and models')
        # !!!!!!!!!!!! modify gpu_ids to gpu_id
        parser.add_argument('--gpu_id', type=str, default='3', help='gpu ids: e.g. 3, default: 3, use -1 for CPU')
        # !!!!!!!! modify, chekpoints_dir -> checkpoint_dir
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='models are saved here')
        parser.add_argument('--log_dir', type=str, default='log', help='logs are saved here')
        # parser.add_argument('--shared_dir', type=str, default='shared', help='shared are saved here')
        # parser.add_argument('--cuda_benchmark', action='store_true', help='use torch cudnn benchmark')
        parser.add_argument('--has_test', default=False, action=argparse.BooleanOptionalAction,
                            help='whether have test, default is False, if set to true, use --has_test; if False, do not point out or use --no-has_test.')

        # model parameters
        parser.add_argument('--model', type=str, default='baseline', choices=['baseline'],
                            help='chooses which model to use. [baseline]')
        parser.add_argument('--input_feature_dims', type=int, nargs='+', default=[512, 1024, 768],
                    help='feature dimensions in order using list, e.g. --input_feature_dims 512 1024 768')
        parser.add_argument('--emotion_network_names', type=str, nargs='+', default=['LSTMEncoder', 'TextCNN', 'LSTMEncoder'],
                    help='emotion_network_names using list, e.g. --emotion_network_names LSTMEncoder TextCNN LSTMEncoder')
        parser.add_argument('--emotion_network_embed_dims', type=int, nargs='+', default=[128, 128, 128],
                    help='emotion_network_embed_dims using list, e.g. --emotion_network_embed_dims 128 128 128')
        parser.add_argument('--intent_network_names', type=str, nargs='+', default=['LSTMEncoder', 'TextCNN', 'LSTMEncoder'],
                    help='intent_network_names using list, e.g. --intent_network_names LSTMEncoder TextCNN LSTMEncoder')
        parser.add_argument('--intent_network_embed_dims', type=int, nargs='+', default=[128, 128, 128],
                    help='intent_network_embed_dims using list, e.g. --intent_network_embed_dims 128 128 128')
        ## parameter of pretrain
        parser.add_argument('--pretrained_path', type=str, default='None',help='where to load pretrained encoder network')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay when training')
        parser.add_argument('--init_type', type=str, default='kaiming',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.012,
                            help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--ablation', type=str, default='normal', help='which module should be ablate')
        
        # dataset parameters
        # parser.add_argument('--dataset_mode', type=str, default='multimodal', choices=['multimodal'],
        #                     help='chooses how datasets are loaded. [multimodal]')
        parser.add_argument('--dataset_filelist_dir', type=str, default='filelists/Track1_English_origin',
                            help='path to dataset filelist directory')
        # add dataset argument feature_dirs, indicata the feature directories using list
        parser.add_argument('--feature_dirs', type=str, nargs='+', default=['path/to/feature1', 'path/to/feature2'],
                            help='feature directories using list, e.g. --feature_dirs path/to/feature1 path/to/feature2')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--drop_last', default=False, action=argparse.BooleanOptionalAction,
                            help='drop the last batch if it is not full')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        # !!!!!!!!! TODO: may need to modify, may not use
        # parser.add_argument('--suffix', default='', type=str,
        #                     help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # training parameter
        # !!!!!!!!! TODO: may not use, show in log
        parser.add_argument('--print_freq', type=int, default=100,
                            help='frequency of showing training results on console')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1,
                            help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # !!!!!!!!! TODO: may not use, same as `mode` in MISA parameter
        parser.add_argument('--phase', type=str, default='train', choices=['train', 'val', 'test'], help='train, val, test')
        parser.add_argument('--niter', type=int, default=20, help='#(number) of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=80,
                            help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        # !!!!!!!! TODO: may not use, same as `learning_rate` in MISA parameter
        parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', choices=['linear', 'step', 'plateau', 'cosine'],
                            help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50,
                            help='multiply by a gamma every lr_decay_iters iterations')

        # MISA parameter
        # mode
        # !!!!!!!!! TODO: may not use, same as `phase` in training parameter
        parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'], help='mode, e.g. train, test, default is train')
        parser.add_argument('--runs', type=int, default=5, help='number of runs, default is 5')

        # bert
        # Python 3.9 and above, using --no-use_bert to set use_bert=False
        parser.add_argument('--use_bert', default=True, action=argparse.BooleanOptionalAction)
        parser.add_argument('--use_cmd_sim', default=True, action=argparse.BooleanOptionalAction)

        # Train
        parser.add_argument('--num_classes', type=int, default=10)
        parser.add_argument('--eval_batch_size', type=int, default=10)
        parser.add_argument('--n_epoch', type=int, default=500)
        parser.add_argument('--patience', type=int, default=6)

        parser.add_argument('--diff_weight', type=float, default=0.15)  # default:0.3  3 0.5 0.5 15
        parser.add_argument('--sim_weight', type=float, default=0.025)  # default:1
        parser.add_argument('--sp_weight', type=float, default=0.0)  #
        parser.add_argument('--recon_weight', type=float, default=0.025)  # default:1
        parser.add_argument('--cls_weight', type=float, default=1)  # default:1

        # !!!!!!!! TODO: may not use, same as `lr` in training parameter
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--clip', type=float, default=1.0)    

        parser.add_argument('--rnncell', type=str, default='lstm')
        parser.add_argument('--embedding_size', type=int, default=300)
        parser.add_argument('--hidden_size', type=int, default=128)
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--reverse_grad_weight', type=float, default=1.0)

        # Selectin activation from 'elu', "hardshrink", "hardtanh", "leakyrelu", "prelu", "relu", "rrelu", "tanh"
        parser.add_argument('--activation', type=str, default='relu', 
                            choices=['elu', 'hardshrink', 'hardtanh', 'leakyrelu', 'prelu', 'relu', 'rrelu', 'tanh'],
                            help='activation function, default is relu')

        parser.add_argument('--random_seed', default=42, type=int, help='# the random seed')
        # !!!!!!!! TODO: what is the help?
        parser.add_argument('--track', default=1, type=int, help='# the random seed')

        # expriment parameter
        # !!!!!!!! TODO: may not use, use default log name: name + time
        parser.add_argument('--run_idx', type=str, help='experiment number; for repeat experiment')

        # !!!!!!!! TODO: self.isTrain = True, useless?
        self.isTrain = True
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
        # modify model-related parser options
        model_name = opt.model
        if opt.verbose:
            print('model_name: ', model_name)
        # model_option_setter = models.get_option_setter(model_name)
        # parser = model_option_setter(parser, self.isTrain)

        # parse again with new defaults
        # opt, _ = parser.parse_known_args()
        # dataset_name = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()
        
    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoint_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'

        if opt.verbose:
            print(message)

        # save to the disk
        expr_dir = opt.checkpoint_dir
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        # log dir
        log_dir = opt.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        # save opt as txt file
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def save_json(self, opt):
        dictionary = {}
        for k, v in sorted(vars(opt).items()):
            dictionary[k] = v

        expr_dir = opt.checkpoint_dir
        save_path = os.path.join(expr_dir, '{}_opt.conf'.format(opt.phase))
        json.dump(dictionary, open(save_path, 'w'), indent=4)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # !!!!!!!!!! modified, delete process opt.suffix
        # if opt.suffix:
        #     suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
        #     opt.name = opt.name + suffix
        
        opt.name = opt.name + '_' + str(time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))) \
                    + '_' + str(opt.model)
        # !!!!!!!!!! modified, change opt.checkpoints_dir and opt.log_dir
        opt.log_dir = os.path.join(opt.log_dir, opt.name)
        opt.checkpoint_dir = os.path.join(opt.checkpoint_dir, opt.name)

        if opt.verbose:
            print("Expr Name:", opt.name)

        self.print_options(opt)

        # set gpu ids
        # str_ids = opt.gpu_ids.split(',')
        # opt.gpu_ids = []
        # for str_id in str_ids:
        #     id = int(str_id)
        #     if id >= 0:
        #         opt.gpu_ids.append(id)
        # if len(opt.gpu_ids) > 0:
        #     torch.cuda.set_device(opt.gpu_ids[0])

        # !!!!!!!!!!! modify gpu_ids to gpu_id
        if opt.gpu_id != '-1':
            torch.cuda.set_device(int(opt.gpu_id))

        if opt.isTrain:
            self.save_json(opt)

        self.opt = opt
        return self.opt