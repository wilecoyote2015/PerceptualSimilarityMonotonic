import os.path
import torchvision.transforms as transforms
from data.dataset.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
# from IPython import embed

class TwoAFCDataset(BaseDataset):
    def initialize(self, dataroots, load_size=64):
        if(not isinstance(dataroots,list)):
            dataroots = [dataroots,]
        self.roots = dataroots
        self.load_size = load_size

        # image directory
        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.ref_paths = make_dataset(self.dir_ref)
        self.ref_paths = sorted(self.ref_paths)

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list.append(transforms.Resize(load_size))
        transform_list += [transforms.ToTensor()]

        self.individual_transform = transforms.Compose(transform_list)
        
        group_transforms_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # random gamma to simulate anscombe transform and linearization
            transforms.Lambda(lambda x: x.pow((1/3) + torch.rand(1).item()*(1.-1/3 + 1))),
            # transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.4)
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        ]
        self.group_transform = transforms.Compose(group_transforms_list)
        

        # judgement directory
        self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        self.judge_paths = make_dataset(self.dir_J,mode='np')
        self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        # TODO: is dtype float? must be! Check! Otherwise, bounds are wrong!
        p0_path = self.p0_paths[index]
        p0_img_ = Image.open(p0_path).convert('RGB')
        p0_img = self.individual_transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = Image.open(p1_path).convert('RGB')
        p1_img = self.individual_transform(p1_img_)

        ref_path = self.ref_paths[index]
        ref_img_ = Image.open(ref_path).convert('RGB')
        ref_img: torch.Tensor = self.individual_transform(ref_img_)
        
        imgs = torch.stack([p0_img, p1_img, ref_img])
        
        
        
        # print(imgs.dtype)
        
        # Random jittering etc.
        imgs = self.group_transform(imgs)
        # print(imgs.min(), imgs.max())
        
        # Normalize as in default training
        ref_img_jittered = imgs[-1]
        ref_flat = ref_img_jittered.reshape([3, -1])
        std_imgs = torch.std(ref_flat, dim=-1)[None, :, None, None] + 1e-6
        mean_imgs = torch.mean(ref_flat, dim=-1)[None, :, None, None]
        
        
        # normalize channels
        imgs_normalized = (imgs - mean_imgs) / std_imgs
        
        
        # scale to [0,1] and clamp
        min_ref = ref_img_jittered.min()
        max_ref = ref_img_jittered.max()
        
        scale = (max_ref - min_ref) if max_ref > min_ref else 1.
        imgs_normalized = (imgs - min_ref) / scale
        imgs_normalized = torch.clamp(imgs_normalized, 0., 1.)
        
        
        p0_img, p1_img, ref_img = imgs_normalized        

        judge_path = self.judge_paths[index]
        # judge_img = (np.load(judge_path)*2.-1.).reshape((1,1,1,)) # [-1,1]
        judge_img = np.load(judge_path).reshape((1,1,1,)) # [0,1]

        judge_img = torch.FloatTensor(judge_img)

        return {'p0': p0_img, 'p1': p1_img, 'ref': ref_img, 'judge': judge_img,
            'p0_path': p0_path, 'p1_path': p1_path, 'ref_path': ref_path, 'judge_path': judge_path}

    def __len__(self):
        return len(self.p0_paths)
