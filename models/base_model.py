from os.path import join
from abc import ABC, abstractmethod
import torch
from .networks import tools
from collections import OrderedDict
import argparse
import os

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt, logger=None):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        # !!! self.gpu_ids should be one
        # self.device = torch.device('cuda:{}'.format(opt.device) if torch.cuda.is_available() and opt.device != '-1' else 'cpu')
        self.device = opt.device
        self.is_train = opt.is_train
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.logger = logger
        self.verbose = opt.verbose
    
    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser

        Returns:
            the modified parser.
        """            
        return parser
    
    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    @abstractmethod
    def setup(self):
        """Load and print networks; create schedulers"""
        # if self.is_train:
        #     self.schedulers = [tools.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
        #     if self.opt.load_model_path != 'None':
        #         for name in self.model_names:
        #             net = getattr(self, 'net' + name)
        #             net = tools.init_net(net, init_type=self.opt.init_type, 
        #                                 init_gain=self.opt.init_gain, device=self.opt.device)
        #             setattr(self, 'net' + name, net)
        # else:
        #     self.eval()
        
        # self.print_networks()
        # self.post_process()
        pass

    def cuda(self):
        assert(torch.cuda.is_available())
        for name in self.model_names:
            net = getattr(self, 'net' + name)
            net.to(self.device)

    def eval(self):
        """Make models eval mode during test time"""
        self.isTrain = False
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()
    
    def train(self):
        """Make models back to train mode after test time"""
        self.isTrain = True
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        with torch.no_grad():
            self.forward()

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        
        lr = self.optimizers[0].param_groups[0]['lr']
        self.logger.info('learning rate = %.7f' % lr)

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, prefix):
        """Save all the networks to the disk.

        Parameters:
        name (str or int) -- the name of the network. It's usually the current step of training.
            step (int) -- current step; used in the file name '%s_net_%s.pth' % (step, name)
            'latest' -- the latest version; used in the file name 'latest_net_%s.pth'
        """
        assert False, 'Not implemented yet'

    def load_networks(self):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        assert False, 'Not implemented yet'

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        self.logger.info('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if self.verbose:
                    print(net)
                self.logger.info(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                self.logger.info('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')
        self.logger.info('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    # @abstractmethod
    # def post_process(self):
    #     pass