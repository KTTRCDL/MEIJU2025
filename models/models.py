import torch
import torch.nn as nn
from models.networks import tools
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier
from models.networks.interact_model import InteractModule
import torch.nn.functional as F

NETWORK = {
    'LSTMEncoder': LSTMEncoder,
    'TextCNN': TextCNN,
}

class MLP_Linear_Attention_Model(nn.Module):
    def __init__(self, opt):
        super(MLP_Linear_Attention_Model, self).__init__()
        self.opt = opt
        self.gpu_id = opt.gpu_id
        self.device = torch.device(f'cuda:{self.gpu_id}') if int(self.gpu_id) >= 0 else torch.device('cpu')
        self.save_dir = opt.checkpoint_dir

        self.emo_output_dim = 7
        self.int_output_dim = 8


class BaselineModel(nn.Module):
    def __init__(self, opt):
        super(BaselineModel, self).__init__()
        self.opt = opt
        self.gpu_id = opt.gpu_id
        self.device = torch.device(f'cuda:{self.gpu_id}') if int(self.gpu_id) >= 0 else torch.device('cpu')
        self.save_dir = opt.checkpoint_dir
        
        self.emo_output_dim = 7
        self.int_output_dim = 8

        self.emotion_networks = []
        for i, emotion_network_name in enumerate(opt.emotion_network_names):
            emotion_network = NETWORK[emotion_network_name](opt.input_feature_dims[i],
                                                            opt.emotion_network_embed_dims[i]).to(self.device)
            # self.networks.append(emotion_network)
            self.emotion_networks.append(emotion_network)
        
        self.intent_networks = []
        for i, intent_network_name in enumerate(opt.intent_network_names):
            intent_network = NETWORK[intent_network_name](opt.input_feature_dims[i],
                                                        opt.intent_network_embed_dims[i]).to(self.device)
            # self.networks.append(intent_network)
            self.intent_networks.append(intent_network)

        # Transformer Fusion model
        emo_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=2)
        self.EmoFusion = torch.nn.TransformerEncoder(emo_encoder_layer, num_layers=1)
        # self.networks.append(self.EmoFusion)

        int_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=opt.hidden_size, nhead=2)
        self.IntFusion = torch.nn.TransformerEncoder(int_encoder_layer, num_layers=1)
        # self.networks.append(self.IntFusion)

        # Modality Interaction
        self.netEmo_Int_interaction = InteractModule(opt)
        # self.networks.append(self.netEmo_Int_interaction)
        self.netInt_Emo_interaction = InteractModule(opt)
        # self.networks.append(self.netInt_Emo_interaction)

        # Classifier
        cls_layers = [128, 64]
        cls_input_size = opt.hidden_size * len(opt.input_feature_dims)

        self.netEmoC = FcClassifier(cls_input_size, cls_layers, output_dim=self.emo_output_dim, dropout=0.2)
        # self.networks.append(self.netEmoC)
        self.netEmoCF = FcClassifier(cls_input_size, cls_layers, output_dim=self.emo_output_dim, dropout=0.2)
        # self.networks.append(self.netEmoCF)

        self.netIntC = FcClassifier(cls_input_size, cls_layers, output_dim=self.int_output_dim, dropout=0.2)
        # self.networks.append(self.netIntC)
        self.netIntCF = FcClassifier(cls_input_size, cls_layers, output_dim=self.int_output_dim, dropout=0.2)
        # self.networks.append(self.netIntCF)


        if opt.pretrained_path != 'None':
            # torch.load(opt.pretrained_path)
            pretrained_dict = torch.load(opt.pretrained_path)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

    def forward(self, data):
        emo_feats = []
        int_feats = []
        for i, emotion_network in enumerate(self.emotion_networks):
            emo_feat = emotion_network(data[i])
            emo_feats.append(emo_feat)
        
        for i, intent_network in enumerate(self.intent_networks):
            int_feat = intent_network(data[i])
            int_feats.append(int_feat)
        
        # emo_fusion_feats = torch.cat(emo_feats, dim=0)
        emo_fusion_feats = torch.stack([emo_feats[i] for i in range(len(emo_feats))], dim=0)
        emo_fusion_feat = self.EmoFusion(emo_fusion_feats)
        emo_logits_fusion, _ = self.netEmoCF(torch.cat(emo_feats, dim=-1))

        # int_fusion_feats = torch.cat(int_feats, dim=0)
        int_fusion_feats = torch.stack([int_feats[i] for i in range(len(int_feats))], dim=0)
        int_fusion_feat = self.IntFusion(int_fusion_feats)
        int_logits_fusion, _ = self.netIntCF(torch.cat(int_feats, dim=-1))

        emo_final_feat = self.netEmo_Int_interaction(emo_fusion_feat, int_fusion_feat, int_fusion_feat)
        int_final_feat = self.netInt_Emo_interaction(int_fusion_feat, emo_fusion_feat, emo_fusion_feat)

        # emotion classifier
        emo_logits, _ = self.netEmoC(emo_final_feat)

        # intent classifier
        int_logits, _ = self.netIntC(int_final_feat)

        return emo_logits, int_logits, emo_logits_fusion, int_logits_fusion

    def predict(self, data):
        self.eval()
        emo_feats = []
        int_feats = []
        for i, emotion_network in enumerate(self.emotion_networks):
            emo_feat = emotion_network(data[i])
            emo_feats.append(emo_feat)
        
        for i, intent_network in enumerate(self.intent_networks):
            int_feat = intent_network(data[i])
            int_feats.append(int_feat)
        
        # emo_fusion_feats = torch.cat(emo_feats, dim=0)
        emo_fusion_feats = torch.stack([emo_feats[i] for i in range(len(emo_feats))], dim=0)
        emo_fusion_feat = self.EmoFusion(emo_fusion_feats)
        # emo_logits_fusion, _ = self.netEmoCF(torch.cat(emo_feats, dim=-1))

        # int_fusion_feats = torch.cat(int_feats, dim=0)
        int_fusion_feats = torch.stack([int_feats[i] for i in range(len(int_feats))], dim=0)
        int_fusion_feat = self.IntFusion(int_fusion_feats)
        # int_logits_fusion, _ = self.netIntCF(torch.cat(int_feats, dim=-1))

        emo_final_feat = self.netEmo_Int_interaction(emo_fusion_feat, int_fusion_feat, int_fusion_feat)
        int_final_feat = self.netInt_Emo_interaction(int_fusion_feat, emo_fusion_feat, emo_fusion_feat)

        # emotion classifier
        emo_logits, _ = self.netEmoC(emo_final_feat)

        # intent classifier
        int_logits, _ = self.netIntC(int_final_feat)

        emo_pred = F.softmax(emo_logits, dim=-1)
        int_pred = F.softmax(int_logits, dim=-1)

        return emo_pred, int_pred