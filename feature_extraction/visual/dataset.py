# *_*coding:utf-8 *_*
import os
import glob
from PIL import Image
from skimage import io
import torch.utils.data as data
import cv2


class FaceDataset(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FaceDataset, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, f'{vid}_aligned')
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name


class FaceDataset2(data.Dataset):
    def __init__(self, vid, face_dir, transform=None):
        super(FaceDataset2, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, f'{vid}_aligned')
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        sorted_frames = sorted(frames, key=lambda x: os.path.basename(x).split('.')[0])
        return sorted_frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = cv2.imread(path)
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name


class FaceDatasetForEmoNet(data.Dataset):
    def __init__(self, vid, face_dir, transform=None, augmentor=None):
        super(FaceDatasetForEmoNet, self).__init__()
        self.vid = vid
        self.path = os.path.join(face_dir, vid)
        self.augmentor = augmentor
        self.transform = transform
        self.frames = self.get_frames()

    def get_frames(self):
        frames = glob.glob(os.path.join(self.path, '*'))
        return frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        path = self.frames[index]
        img = io.imread(path)
        if self.augmentor is not None:
            img = self.augmentor(img)[0]
        if self.transform is not None:
            img = self.transform(img)
        name = os.path.basename(path)[:-4]
        return img, name