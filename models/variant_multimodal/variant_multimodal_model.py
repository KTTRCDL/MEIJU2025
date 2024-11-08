from models.base_model import BaseModel
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.interact_model import InteractModule
from models.networks.classifier import FcClassifier
from models.networks import tools
import torch.nn as nn
import os
import json
from models.utils.config import OptConfig
import torch
import argparse
import torch.nn.functional as F

NETWORK = {
    'LSTMEncoder': LSTMEncoder,
    'TextCNN': TextCNN,
}
class VariantMultimodalModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--embed_methods', type=str, nargs='+',
                             default=['maxpool', 'NULL', 'maxpool'],
                            help='modal embedding method, last, maxpool, mean or attention')
        parser.add_argument('--emotion_network_names', type=str, nargs='+',
                            default=['LSTMEncoder', 'TextCNN', 'LSTMEncoder'],
                            help='emotion network names')
        parser.add_argument('--intent_network_names', type=str, nargs='+',
                            default=['LSTMEncoder', 'TextCNN', 'LSTMEncoder'],
                            help='intent network names')
        parser.add_argument('--input_feature_dims', type=int, nargs='+', default=[512, 1024, 768],
                            help='input feature dimensions in order using list, e.g. --input_feature_dims 512 1024 768')
        parser.add_argument('--embed_dims', type=int, nargs='+', default=[128, 128, 128],
                            help='embedding dimensions in order using list, e.g. --embed_dims 128 128 128')
        parser.add_argument('--embed_dropout', type=float, default=0.5, help='dropout rate of embedding')
        parser.add_argument('--hidden_size',type=int, default=128,
                            help='hidden size of model (int), e.g. --hidden_size 128')
        parser.add_argument('--Transformer_head', type=int, default=2, help='head of fusion Transformer')
        parser.add_argument('--Transformer_layers', type=int, default=1, help='layer of fusion Transformer')
        parser.add_argument('--emotion_cls_layers', type=str, default='128,64',
                    help='128,64 for 2 layers with 128,64 nodes respectively')
        parser.add_argument('--intent_cls_layers', type=str, default='128,64',
                    help='128,64 for 2 layers with 128,64 nodes respectively')
        parser.add_argument('--ablation', type=str, default='normal', help='which module should be ablate in interaction module')
        parser.add_argument('--cls_dropout', type=float, default=0.2, help='dropout rate of classifier')
        parser.add_argument('--emotion_output_dim', type=int, default=7, help='output dimension of emotion')
        parser.add_argument('--intent_output_dim', type=int, default=8, help='output dimension of intent')
        parser.add_argument('--use_ICL', default=False, action=argparse.BooleanOptionalAction, 
                            help='add imbalance classify loss or not, if not, --no-use_ICL,use cross entropy loss')
        parser.add_argument('--focal_weight', type=float, default=1.0, help='weight of loss criterion_focal (focal loss or cross entropy loss)')

        return parser
    
    def __init__(self, opt, logger=None):
        super().__init__(opt, logger)
        self.model_names = []
        self.loss_names = []
        self.optimizers = []
        self.opt = opt
        self.device = opt.device
        self.logger = logger
        self.is_train = opt.is_train

        for i, (emotion_network_name, intent_network_name) in enumerate(zip(opt.emotion_network_names, opt.intent_network_names)):
            setattr(self, f'netEmoEmbed{i}', NETWORK[emotion_network_name](opt.input_feature_dims[i], 
                                                                      opt.embed_dims[i],
                                                                      embd_method=opt.embed_methods[i],
                                                                      droupout=opt.embed_dropout))
            self.model_names.append(f'EmoEmbed{i}')
            setattr(self, f'netIntEmbed{i}', NETWORK[intent_network_name](opt.input_feature_dims[i], 
                                                                      opt.embed_dims[i], 
                                                                      embd_method=opt.embed_methods[i],
                                                                      droupout=opt.embed_dropout))
            self.model_names.append(f'IntEmbed{i}')

        # embed to hidden
        # for i in range(len(opt.input_feature_dims)):
        #     setattr(self, f'netEmoHidden{i}', nn.Linear(opt.embed_dims[i], opt.hidden_size))
        #     self.model_names.append(f'EmoHidden{i}')
        #     setattr(self, f'netIntHidden{i}', nn.Linear(opt.embed_dims[i], opt.hidden_size))
        #     self.model_names.append(f'IntHidden{i}')

        # Transformer Fusion model
        emo_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=opt.Transformer_head)
        self.netEmoFusion = nn.TransformerEncoder(emo_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('EmoFusion')
        int_encoder_layer = nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=opt.Transformer_head)
        self.netIntFusion = nn.TransformerEncoder(int_encoder_layer, num_layers=opt.Transformer_layers)
        self.model_names.append('IntFusion')

        # Modality Interaction
        self.netEmo_Int_interaction = InteractModule(opt)
        self.model_names.append('Emo_Int_interaction')
        self.netInt_Emo_interaction = InteractModule(opt)
        self.model_names.append('Int_Emo_interaction')

        # Classifier
        emo_cls_layers = list(map(lambda x: int(x), opt.emotion_cls_layers.split(',')))
        int_cls_layers = list(map(lambda x: int(x), opt.intent_cls_layers.split(',')))
        cls_input_size = opt.hidden_size * len(opt.input_feature_dims)

        # TODO: may FcClassifier use_bn
        self.netEmoCF = FcClassifier(cls_input_size, emo_cls_layers, output_dim=opt.emotion_output_dim, dropout=opt.cls_dropout)
        self.model_names.append('EmoCF')
        self.loss_names.append('EmoF_CE')

        self.netIntCF = FcClassifier(cls_input_size, int_cls_layers, output_dim=opt.intent_output_dim, dropout=opt.cls_dropout)
        self.model_names.append('IntCF')
        self.loss_names.append('IntF_CE')

        if self.opt.load_model_path != 'None':
            self.netEmoC = FcClassifier(cls_input_size, emo_cls_layers, output_dim=opt.emotion_output_dim, dropout=opt.cls_dropout)
            self.model_names.append('EmoC')
            self.loss_names.append('Emo_CE')

            self.netIntC = FcClassifier(cls_input_size, int_cls_layers, output_dim=opt.intent_output_dim, dropout=opt.cls_dropout)
            self.model_names.append('IntC')
            self.loss_names.append('Int_CE')


    def setup(self):
        """Load and print networks; create schedulers"""
        if self.is_train:
            self.schedulers = [tools.get_scheduler(optimizer, self.opt) for optimizer in self.optimizers]
            # init net
            for name in self.model_names:
                net = getattr(self, 'net' + name)
                net = tools.init_net(net, init_type=self.opt.init_type, 
                                    init_gain=self.opt.init_gain, device=self.opt.device, logger=self.logger)
                setattr(self, 'net' + name, net)
            # load pretrained model
            self.load_pretrained_encoder()
            if self.opt.use_ICL:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                # TODO: may change high parameter
                self.criterion_focal = Focal_Loss()
            else:
                self.criterion_ce = torch.nn.CrossEntropyLoss()
                self.criterion_focal = torch.nn.CrossEntropyLoss()
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.focal_weight = self.opt.focal_weight
            # TODO: modify save_dir and create save_dir
        else:
            self.load_networks()
            self.cuda()
            self.eval()
        
        self.print_networks()

    def load_pretrained_encoder(self):
        # judge pretrain or not
        if self.opt.load_model_path == 'None':
            self.logger.info('No pretrained model to load, training from scratch to get the pretrained model')
            return
        
        self.logger.info(f'Loading pretrained model (encoder) from {self.opt.load_model_path}, file prefix: {self.opt.load_model_prefix}')
        all_checkpoint_files = list(filter(lambda x: x.endswith('.pth'), os.listdir(self.opt.load_model_path)))
        for name in self.model_names:
            # !!! TODO: may change the condition?
            if isinstance(name, str) and 'interaction' not in name and 'EmoC' not in name and 'IntC' not in name:
            # if isinstance(name, str) and 'interaction' not in name:
            # if isinstance(name, str):
                load_filename = list(filter(lambda x: x.split('.')[0].endswith(self.opt.load_model_prefix + '_' + name), 
                                            all_checkpoint_files))
                if len(load_filename) == 1:
                    load_filename = load_filename[0]
                    self.logger.info(f'Loading {name} from {load_filename}')
                    getattr(self, 'net' + name).load_state_dict(torch.load(os.path.join(self.opt.load_model_path, load_filename)))
                    self.logger.info(f'load pretrain model, name: {name} load_filename: {load_filename}')
                else:
                    self.logger.info(f'!!! NOT LOAD PRETRAIN MODEL, BECAUSE name: {name} load_filename: {load_filename}')

    def load_networks(self):
        self.logger.info(f'Loading whole model (encoder) from {self.opt.load_model_path}, file prefix: {self.opt.load_model_prefix}')
        all_checkpoint_files = list(filter(lambda x: x.endswith('.pth'), os.listdir(self.opt.load_model_path)))
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = list(filter(lambda x: x.split('.')[0].endswith(self.opt.load_model_prefix + '_' + name), 
                                            all_checkpoint_files))
                if len(load_filename) == 1:
                    load_filename = load_filename[0]
                    self.logger.info(f'Loading {name} from {load_filename}')
                    getattr(self, 'net' + name).load_state_dict(torch.load(os.path.join(self.opt.load_model_path, load_filename)))
                    self.logger.info(f'load pretrain model, name: {name} load_filename: {load_filename}')
                else:
                    self.logger.info(f'!!! NOT LOAD PRETRAIN MODEL, BECAUSE name: {name} load_filename: {load_filename}')
 
    # def load_from_opt_record(self, file_path):
    #     opt_content = json.load(open(file_path, 'r'))
    #     opt = OptConfig()
    #     opt.load(opt_content)
    #     return opt

    def save_networks(self, prefix):
        # judge pretrain or not
        if self.opt.load_model_path == 'None':
            self.logger.info('save the pretrained model')
            for name in self.model_names:
                if isinstance(name, str):
                    save_filename = f'pretrained_{prefix}_{name}.pth'
                    self.logger.info(f'Saving {name} to {save_filename}')
                    torch.save(getattr(self, 'net' + name).cpu().state_dict(), os.path.join(self.opt.checkpoints_dir, save_filename))
        else:
            self.logger.info('save the trained model')
            for name in self.model_names:
                if isinstance(name, str):
                    save_filename = f'{prefix}_{name}.pth'
                    self.logger.info(f'Saving {name} to {save_filename}')
                    torch.save(getattr(self, 'net' + name).cpu().state_dict(), os.path.join(self.opt.checkpoints_dir, save_filename))
        self.cuda()

    def set_input(self, input):
        if self.is_train:
            self.emo_label = input['emotion_label'].to(self.device)
            self.int_label = input['intent_label'].to(self.device)
            
        features = input['features']
        for i in range(len(features)):
            features[i] = features[i].to(self.device)
        self.features = features

    def forward(self):
        emo_feats = []
        int_feats = []

        for i in range(len(self.features)):
            emo_embed = getattr(self, f'netEmoEmbed{i}')(self.features[i])
            # emo_hidden = getattr(self, f'netEmoHidden{i}')(emo_embed)
            emo_hidden = emo_embed
            emo_feats.append(emo_hidden)

            int_embed = getattr(self, f'netIntEmbed{i}')(self.features[i])
            # int_hidden = getattr(self, f'netIntHidden{i}')(int_embed)
            int_hidden = int_embed
            int_feats.append(int_hidden)

        self.emo_logits_fusion, _ = self.netEmoCF(torch.cat(emo_feats, dim=-1))
        self.int_logits_fusion, _ = self.netIntCF(torch.cat(int_feats, dim=-1))

        # emotion prediction
        if self.opt.load_model_path == 'None':
            self.emo_pred = F.softmax(self.emo_logits_fusion, dim=-1)
            self.int_pred = F.softmax(self.int_logits_fusion, dim=-1)

        elif self.opt.load_model_path != 'None':
            emo_fusion_feats = torch.stack([emo_feats[i] for i in range(len(emo_feats))], dim=0)
            emo_fusion_feat = self.netEmoFusion(emo_fusion_feats)

            int_fusion_feats = torch.stack([int_feats[i] for i in range(len(int_feats))], dim=0)
            int_fusion_feat = self.netIntFusion(int_fusion_feats)
            emo_int_interaction = self.netEmo_Int_interaction(emo_fusion_feat, int_fusion_feat, int_fusion_feat)
            int_emo_interaction = self.netInt_Emo_interaction(int_fusion_feat, emo_fusion_feat, emo_fusion_feat)
            self.emo_logits, _ = self.netEmoC(emo_int_interaction)
            self.int_logits, _ = self.netIntC(int_emo_interaction)

            self.emo_pred = F.softmax(self.emo_logits, dim=-1)
            self.int_pred = F.softmax(self.int_logits, dim=-1)
            

    def backward(self):
        """Calculate the loss for back propagation"""
        self.loss_EmoF_CE = self.focal_weight * self.criterion_focal(self.emo_logits_fusion, self.emo_label)
        self.loss_IntF_CE = self.focal_weight * self.criterion_focal(self.int_logits_fusion, self.int_label)

        loss = self.loss_EmoF_CE + self.loss_IntF_CE

        if self.opt.load_model_path != 'None':
            self.loss_Emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
            self.loss_Int_CE = self.criterion_ce(self.int_logits, self.int_label)

            loss += self.loss_Emo_CE + self.loss_Int_CE

        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 1.0)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.optimizer.zero_grad()
        self.forward()
        self.backward()
        self.optimizer.step()

    # def post_process(self):
    #     return super().post_process()

class Focal_Loss(torch.nn.Module):
    def __init__(self, weight=0.5, gamma=3, reduction='mean'):
        super(Focal_Loss, self).__init__()
        self.gamma = gamma
        self.alpha = weight
        self.reduction = reduction

    def forward(self, preds, targets):
        ce_loss = F.cross_entropy(preds, targets, reduction='mean')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            raise NotImplementedError("Invalid reduction mode. Please choose 'none', 'mean', or 'sum'.")