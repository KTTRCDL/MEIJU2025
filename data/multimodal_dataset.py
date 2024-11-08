import torch.utils.data.dataset
import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def create_dataloader_with_args(opt, set_name='train'):
    dataset = MultimodalDataset(opt, set_name=set_name)
    if dataset.manual_collate_fn:
        return DataLoader(dataset, 
                          batch_size=opt.batch_size, 
                          shuffle=not opt.serial_batches, 
                          num_workers=int(opt.num_threads),
                          drop_last=opt.drop_last,
                          collate_fn=dataset.collate_fn)
    else:
        return DataLoader(dataset, 
                          batch_size=opt.batch_size, 
                          shuffle=not opt.serial_batches, 
                          num_workers=int(opt.num_threads),
                          drop_last=opt.drop_last)

class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, opt, set_name='train'):
        super(MultimodalDataset, self).__init__()
        self.set_name = set_name
        self.filelist_dir = opt.dataset_filelist_dir
        self.feature_dirs = opt.feature_dirs
        self.emotion_dict = {'happy': 0, 'surprise': 1, 'sad': 2, 'disgust': 3, 'anger': 4, 'fear': 5, 'neutral': 6}
        self.intent_dict = {'questioning': 0, 'agreeing': 1, 'acknowledging': 2, 'encouraging': 3, 'consoling': 4,
                       'suggesting': 5, 'wishing': 6, 'neutral': 7}
        filelist = os.path.join(self.filelist_dir, f'{set_name}_filelist.txt')
        self.sample = []

        with open(filelist, 'r') as f:
            for line in f.readlines():
                cut = line.strip().split()
                filename = cut[0]
                if set_name == 'train' or set_name == 'valid':
                    emotion = self.emotion_dict[cut[1]]
                    intent = self.intent_dict[cut[2]]
                    self.sample.append((filename, emotion, intent))
                else:
                    self.sample.append((filename))
        
        self.manual_collate_fn = True
        
    def __getitem__(self, index):
        if self.set_name == 'train' or self.set_name == 'valid':
            filename, emotion_label, intent_label = self.sample[index]
        elif self.set_name == 'test':
            filename = self.sample[index]
        else:
            raise ValueError(f'Invalid set name: {self.set_name}')

        features = []
        for feature_dir in self.feature_dirs:
            feature_path = f'{feature_dir}/{filename}.npy'
            feature = torch.from_numpy(np.load(feature_path)).float()

            # TODO: variable padding
            if feature.shape[0] < 12:
                times = 12 // feature.shape[0] + 1
                feature = torch.cat([feature] * times, 0)

            features.append(feature)
        
        if self.set_name == 'train' or self.set_name == 'valid':
            return {
                'filename': filename,
                'features': features,
                'emotion_label': emotion_label,
                'intent_label': intent_label,
            }
        else:
            return {
                'filename': filename,
                'features': features,
            }

    def __len__(self):
        return len(self.sample)
    
    def collate_fn(self, batch):
        filenames = [item['filename'] for item in batch]
        if self.set_name == 'train' or self.set_name == 'valid':
            emotion_labels = torch.tensor([item['emotion_label'] for item in batch]).long()
            intent_labels = torch.tensor([item['intent_label'] for item in batch]).long()

        features = [item['features'] for item in batch]
        features = [[feature[i] for feature in features] for i in range(len(self.feature_dirs))]
        features_lengths = [[len(feature) for feature in features[i]] for i in range(len(self.feature_dirs))]
        # padding
        features = [pad_sequence(features[i], batch_first=True, padding_value=0) for i in range(len(self.feature_dirs))]

        if self.set_name == 'train' or self.set_name == 'valid':
            return {
                'filename': filenames,
                'features': features,
                'features_lengths': features_lengths,
                'emotion_label': emotion_labels,
                'intent_label': intent_labels,
            }
        else:
            return {
                'filename': filenames,
                'features': features,
                'features_lengths': features_lengths,
            }
        