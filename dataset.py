import os
import cv2
import torch
import re
import json
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from torchvision import transforms
import sys

class StorySalonDataset(Dataset): 
    def __init__(self, root, dataset_name):
        self.root = root
        self.dataset_name = dataset_name

        self.train_image_seg = []
        self.train_mask_seg = []
        self.train_text_seg = []
        self.test_image_seg = []
        self.test_mask_seg = []
        self.test_text_seg = []

        self.train_image_list = []
        self.train_mask_list = []
        self.train_text_list = []
        self.test_image_list = []
        self.test_mask_list = []
        self.test_text_list = []

        self.PDF_test_set = []
        self.video_test_set = []
        for line in open(os.path.join(self.root, 'PDF_test_set.txt')).readlines():
            self.PDF_test_set.append(line[:-1])

        keys = ['African', 'Bloom', 'Book', 'Digital', 'Literacy', 'StoryWeaver']

        for key in keys:
            self.PDF_image_dir = os.path.join(self.root, 'Image_inpainted', key)
            # self.PDF_image_dir = os.path.join(self.root, 'Image',key)
            self.PDF_mask_dir = os.path.join(self.root, 'Mask_ins',key)
            self.PDF_text_dir = os.path.join(self.root, 'Text','Caption',key)
            # self.PDF_text_dir = os.path.join(self.root, 'Text','Caption_minigpt',key) #推理的时候的text
            PDF_folders = sorted(os.listdir(self.PDF_image_dir)) 
            #打开data文件夹
            self.train_image_folders = [os.path.join(self.PDF_image_dir, folder) for folder in PDF_folders if folder not in self.PDF_test_set]
            self.train_mask_folders = [os.path.join(self.PDF_mask_dir, folder) for folder in PDF_folders if folder not in self.PDF_test_set]
            self.train_text_folders = [os.path.join(self.PDF_text_dir, folder) for folder in PDF_folders if folder not in self.PDF_test_set] #推理的时候注释掉
            self.test_image_folders = [os.path.join(self.PDF_image_dir, folder) for folder in PDF_folders if folder in self.PDF_test_set]
            self.test_mask_folders = [os.path.join(self.PDF_mask_dir, folder) for folder in PDF_folders if folder in self.PDF_test_set]
            self.test_text_folders = [os.path.join(self.PDF_text_dir, folder) for folder in PDF_folders if folder in self.PDF_test_set]

            for folder in self.train_image_folders:
                image_files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.jpg')])
                if len(image_files) < 2:
                    continue
                for i in range(len(image_files) - 1):
                    image_group = []
                    ref_image_file = image_files[i]
                    ref_image_path = os.path.join(folder, ref_image_file)
                    image_group.append(ref_image_path)
                    
                    ref_corresponding_folder = os.path.join(folder, ref_image_file.split('.')[0])
                    if os.path.exists(ref_corresponding_folder) and len(os.listdir(ref_corresponding_folder)) > 0:
                        sub_images = sorted([f for f in os.listdir(ref_corresponding_folder) if f.endswith('.jpg')])
                        for sub_image in sub_images:
                            sub_image_path = os.path.join(ref_corresponding_folder, sub_image)
                            image_group.append(sub_image_path)
                    
                    target_image_file = image_files[i+1]
                    target_image_path = os.path.join(folder, target_image_file)
                    image_group.append(target_image_path)

                    self.train_image_seg.append(image_group)
            
            for folder in self.train_mask_folders:
                mask_dirs = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
                if len(mask_dirs) < 2:
                    continue
                
                for i in range(len(mask_dirs) - 1):
                    target_mask_dir_name = mask_dirs[i+1]
                    target_mask_dir_path = os.path.join(folder, target_mask_dir_name)
                    
                    mask_group = []
                    if os.path.exists(target_mask_dir_path):
                        sub_masks = sorted([f for f in os.listdir(target_mask_dir_path) if f.endswith('.png')])
                        for sub_mask in sub_masks:
                            sub_mask_path = os.path.join(target_mask_dir_path, sub_mask)
                            mask_group.append(sub_mask_path)
                    
                    self.train_mask_seg.append(mask_group)
            
            #推理的时候注释掉
            for folder in self.train_text_folders:
                text_files = sorted([f for f in os.listdir(folder) if f.endswith('.txt') and os.path.isfile(os.path.join(folder, f))])
                if len(text_files) < 2:
                    continue
                
                for i in range(len(text_files) - 1):
                    ref_text_path = os.path.join(folder, text_files[i])
                    target_text_path = os.path.join(folder, text_files[i+1])
                    self.train_text_seg.append([ref_text_path, target_text_path])


            for video in self.test_image_folders: 
                images = sorted(os.listdir(video))
                images = [img for img in images if os.path.isfile(os.path.join(video, img))]
                if len(images) <= 1:
                    print(video)
                    continue
                else:
                    for i in range(len(images) - 1):
                        self.test_image_list.append([os.path.join(video, images[i]), os.path.join(video, images[i + 1])])

            for video in self.test_text_folders:  
                texts = sorted(os.listdir(video))
                if len(texts) <= 1:
                    continue
                else:
                    for i in range(len(texts) - 1):
                        self.test_text_list.append([os.path.join(video, texts[i]), os.path.join(video, texts[i + 1])])
            
        
        if self.dataset_name == 'train':
            self.image_list = self.train_image_seg
            self.mask_list = self.train_mask_seg
            self.text_list = self.train_text_seg
        elif self.dataset_name == 'test':
            self.image_list = self.test_image_list
            self.mask_list = self.test_mask_list
            self.text_list = self.test_text_list
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
    
        def apply_dropout_and_mosaic(masks, prob=0.3):
            """
            对 masks 应用随机 dropout。
            """
            if random.uniform(0, 1) < prob:
                return torch.zeros_like(masks)
            return masks

        
        if self.dataset_name=='train':
            image_group = self.image_list[index]
            ref_image_ids = image_group[:-1] #前面几张
            image = image_group[-1]
            image_id = image
            
            mask_paths = self.mask_list[index]

            ref_texts = self.text_list[index][0] 
            text = self.text_list[index][1]
            #append
            ref_images_0 = []
            for id in ref_image_ids:
                if id != '':
                    ref_images_0.append(Image.open(id).convert('RGB')) #转成RGB size:1280*720
            image = Image.open(image).convert('RGB') #转换目标图像 1280*720
            
            masks_0 = []
            for m_path in mask_paths:
                masks_0.append(Image.open(m_path).convert('RGB'))

            ref_images_1 = [] #将所有的参考图像、目标图像和掩码图像的大小调整为 512x512 像素。
            for ref_image in ref_images_0:
                if ref_image != '':
                    ref_images_1.append(ref_image.resize((512, 512)))

            image = image.resize((512, 512))
            
            masks_1 = []
            for m in masks_0:
                masks_1.append(m.resize((512, 512)))

            ref_images_2 = [] 
            for ref_image in ref_images_1:
                if ref_image != '':
                    ref_images_2.append(np.ascontiguousarray(transforms.ToTensor()(ref_image)))

            image = transforms.ToTensor()(image)

            masks_2 = []
            for m in masks_1:
                masks_2.append(np.ascontiguousarray(transforms.ToTensor()(m)))

            if len(ref_images_2) > 0:
                ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()
            else:
                ref_images = torch.zeros(1, 3, 512, 512).float()  
            
            if len(masks_2) > 0:
                mask = torch.from_numpy(np.ascontiguousarray(masks_2)).float()
            else:
                mask = torch.zeros(1, 3, 512, 512).float()

            mask = apply_dropout_and_mosaic(mask, prob=0.3)
            image = torch.from_numpy(np.ascontiguousarray(image)).float()

            with open(ref_texts, "r") as f:
                ref_prompts = f.read()
            with open(text, "r") as f:
                prompt = f.read()

            # Unconditional generation for classifier-free guidance
            if self.dataset_name == 'train':
                p = random.uniform(0, 1)
                if p < 0.05:
                    prompt = '' #p小于 0.05，那么 prompt 被设置为空字符串 ''，表示该样本将进行无条件生成
                p = random.uniform(0, 1)
                if p < 0.1:
                    ref_prompts = ''
                    ref_images = ref_images * 0.

            # normalize
            for ref_image in ref_images: #将图像像素值从 [0, 1] 映射到 [-1, 1] 范围。
                ref_image = ref_image * 2. - 1.
            image = image * 2. - 1.
            
            return {"ref_image": ref_images, "image": image, "mask": mask, "ref_prompt": ref_prompts, "prompt": prompt, 'ref_image_ids':ref_image_ids, 'image_id':image_id}

        elif self.dataset_name == 'test':

            ref_image_ids = self.image_list[index][0:1]
            image = self.image_list[index][1]

            ref_texts = self.text_list[index][0:1]  
            text = self.text_list[index][1]  
  
            ref_images_0 = []
            for id in ref_image_ids:
                ref_images_0.append(Image.open(id).convert('RGB'))  
            image = Image.open(image).convert('RGB')  
           

            ref_images_1 = []  
            for ref_image in ref_images_0:
                ref_images_1.append(ref_image.resize((512, 512)))
            image = image.resize((512, 512))
   

            ref_images_2 = []  
            for ref_image in ref_images_1:
                ref_images_2.append(np.ascontiguousarray(transforms.ToTensor()(ref_image)))
            image = transforms.ToTensor()(image)
        

            ref_images = torch.from_numpy(np.ascontiguousarray(ref_images_2)).float()  
            image = torch.from_numpy(np.ascontiguousarray(image)).float()


            ref_prompts = []  
            for ref_text in ref_texts:
                with open(ref_text, "r") as f:
                    ref_prompts.append(f.read())
            with open(text, "r") as f:
                prompt = f.read()

            # normalize
            for ref_image in ref_images:  
                ref_image = ref_image * 2. - 1.
            image = image * 2. - 1.

            return {"ref_image": ref_images, "image": image, "ref_prompt": ref_prompts, "prompt": prompt ,'ref_image_ids':ref_image_ids}
