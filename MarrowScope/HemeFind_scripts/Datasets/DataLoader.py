import albumentations

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils import data
## Simple augumentation to improtve the data generalibility
transform_shape = albumentations.Compose([
    albumentations.ShiftScaleRotate(p=0.8),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Affine(shear=(-10, 10), p=0.3),
    albumentations.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.01), always_apply=False, p=0.2)
])
transform_color = albumentations.Compose([
    albumentations.RandomBrightnessContrast(contrast_limit=0.4, brightness_by_max=0.4, p=0.5),
    albumentations.CLAHE(p=0.3),
    albumentations.ColorJitter(p=0.2),
    albumentations.RandomGamma(p=0.2),

])

class Img_DataLoader(data.Dataset):
    def __init__(self, img_list='', in_dim=3, split='train', transform=False, in_size=96, df=None, encoder=None,
                 if_augumented=False,
                 if_external=False, df_features=None):
        super(Img_DataLoader, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.filelist = img_list
        self.in_size = in_size
        self.file_paths = img_list
        self.transform = transform
        self.df = df
        self.encoder = encoder
        self.if_external = if_external
        self.df_features = df_features
        self.if_augumented = if_augumented

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        # prepare image
        orig_img = cv2.imread(img_path)
        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        if self.if_augumented:
            augmented_image = transform_shape(image=image)['image']

            # for _channel in range(3):
            #    augmented_image[:,:,_channel] = transform_color(image=augmented_image[:,:,_channel])['image']
            image = transform_color(image=augmented_image)['image']

        # hed_lighter_aug.randomize()
        # augmented_image =hed_lighter_aug.transform(augmented_image)

        if self.transform is not None:
            try:
                img = self.transform(image=image)["image"]
            except:
                assert 1 == 2, 'something wrong'
                print(image)

        label = img_path.split('/')[-2]
        # print(img.shape)
        # if self.if_external:
        
        img = cv2.resize(img, (512,512))

        
        img = np.einsum('ijk->kij', img)

        high_level_name = label
        if self.split != "compute":  # Use compute if you only want the prediction results. if you do this, make sure you don't shuffle the data
            if img_path.split('/')[2] == 'BM_cytomorphology_data':
                label = img_path.split('/')[-3]
                high_level_name = mapping_dic[label]

            mask = self.df[self.df['Types'] == high_level_name].iloc[:, 2:].to_numpy()
            length = mask.shape[1]

            sample["label"] = torch.from_numpy(mask.reshape(1, length)).float()  # one hot encoder

            if self.df_features is not None:
                features = self.df_features[self.df_features.index == high_level_name].to_numpy()
                length = features.shape[1]
                sample['features'] = torch.from_numpy(features.reshape(1, length)).float()
        sample["image"] = torch.from_numpy(img).float()  # self.encoder(torch.from_numpy(img).float())
        sample["ID"] = img_path
        return sample