import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk
import yaml
import math
import random
import pdb
from training import augmentation
import os
import yaml
import time
import sys
import pandas as pd
import shutil

def extract_id(image_path):
    """
    Extract the base ID from an image file path by removing the directory,
    extension, and a trailing '_gt' if present.
    """
    file_name = os.path.basename(image_path)
    base = os.path.splitext(file_name)[0]  # removes .npy or .npz
    if base.endswith('_gt'):
        base = base[:-3]  # remove the '_gt'
    return base

def balance_ct_scans(df, healthy_cases, tumor_type_col='tumor_type', case_name_col='case_name', random_state=None):
    """
    Given a dataframe where each row represents a tumor and CT scans are identified by case_name,
    along with a list of healthy case_names (i.e. scans with none of the lesions), this function:
      1) Counts the number of unique CT scans for each tumor type (cyst, pdac, pnet) and healthy cases.
      2) Balances the dataset by randomly repeating (with replacement) case names for the
         groups with fewer scans, so that each group ends up with the same count.
    
    Parameters:
      df (pd.DataFrame): The input dataframe (for tumor cases).
      healthy_cases (list): A list of healthy case_names (scans with no lesions).
      tumor_type_col (str): The column name representing the tumor type.
      case_name_col (str): The column name identifying the CT scans.
      random_state (int, optional): Seed for reproducible random sampling.
    
    Returns:
      balanced_list (list): A list containing the balanced case_names across all groups.
      counts (dict): Original counts (number of unique CT scans per group).
      balanced_counts (dict): New counts after balancing per group.
    """
    # Group unique CT scans by tumor type.
    groups = {}
    for t in ['cyst', 'pdac', 'pnet']:
        # get unique case_names for tumor type t from the dataframe
        groups[t] = df.loc[df[tumor_type_col] == t, case_name_col].unique().tolist()
    
    # Add the healthy cases as another group.
    groups['healthy'] = list(set(healthy_cases))  # Ensure they are unique.
    
    # Count the number of unique CT scans per group.
    counts = {t: len(groups[t]) for t in groups}
    print("Original counts:", counts)
    
    # Identify the maximum count across the groups.
    max_count = max(counts.values())
    
    # Initialize a random number generator.
    rng = np.random.default_rng(random_state)
    
    # Balance groups: if a group has fewer unique CT scans than max_count,
    # sample additional names with replacement.
    balanced_groups = {}
    balanced_counts = {}
    for t, names in groups.items():
        current_count = len(names)
        if current_count < max_count:
            additional = max_count - current_count
            # Randomly select additional names with replacement.
            sampled = rng.choice(names, size=additional, replace=True).tolist()
            balanced_names = names + sampled
        else:
            balanced_names = names
        balanced_groups[t] = balanced_names
        balanced_counts[t] = len(balanced_names)
        print(f"Balanced count for {t}: {len(balanced_names)}")
    
    # Combine the balanced case names across all groups into one list.
    balanced_list = []
    for t in balanced_groups:
        balanced_list.extend(balanced_groups[t])
    
    return balanced_list

#python dataset_abdomenatlas.py --dataset abdomenatlas --model medformer --dimension 3d --batch_size 2 --crop_on_tumor --save_destination /fastwork/psalvador/JHU/data/atlas_300_medformer_augmented_npy_augmented_multich_crop_on_tumor/ --crop_on_tumor --multi_ch_tumor --workers_overwrite 10

def balance_classes(class1, class2):
    """
    Balances two lists of strings by repeating the smaller one until its length
    matches the larger one and then shuffling both lists.
    
    Parameters:
        class1 (list of str): The first class.
        class2 (list of str): The second class.
        
    Returns:
        tuple: A tuple (balanced_class1, balanced_class2) with both lists balanced.
    """
    # Determine which list is smaller
    if len(class1) < len(class2):
        # Compute how many times to repeat class1 to match class2's size
        times = len(class2) // len(class1)
        remainder = len(class2) % len(class1)
        balanced_class1 = class1 * times + class1[:remainder]
        balanced_class2 = class2[:]  # Make a copy of class2
    elif len(class2) < len(class1):
        times = len(class1) // len(class2)
        remainder = len(class1) % len(class2)
        balanced_class2 = class2 * times + class2[:remainder]
        balanced_class1 = class1[:]  # Make a copy of class1
    else:
        # If they are already equal in size, just copy them
        balanced_class1, balanced_class2 = class1[:], class2[:]
    
    # Shuffle both lists in place
    random.shuffle(balanced_class1)
    random.shuffle(balanced_class2)
    
    return balanced_class1, balanced_class2


class AbdomenAtlasDataset(Dataset):
    def __init__(self, args, mode='train', seed=0, all_train=False,
                crop_on_tumor=False,
                 save_destination=None,  
                 load_augmented=False,
                 gigantic_length=False,
                 save_augmented=False,
                 id_list=None,
                 balance_lesion_types = True):    
        
        self.mode = mode
        self.args = args
        self.load_augmented = load_augmented   
        self.save_counter = 0 
        self.save_destination = save_destination
        self.load_augmented = load_augmented
        self.gigantic_length=gigantic_length
        self.save_augmented = save_augmented
        assert mode in ['train', 'test']

        with open(os.path.join(args.jhh_root, 'list', 'label_names.yaml'), 'r') as f:
            classes_jhh = yaml.load(f, Loader=yaml.SafeLoader)
            if 'pancreatic_pdac' not in classes_jhh:
                raise ValueError(f'The label names in JHH do not contain pancreatic_pdac, please check your JHH {os.path.join(args.jhh_root, "list", "label_names.yaml")} file.')
            #sort--we sorted when saving in nii2npy.py
            classes_jhh = sorted(classes_jhh)
            
        self.img_name_list_jhh = pd.read_csv(args.jhh_train)['BDMAP ID'].tolist()
        img_name_list = self.img_name_list_jhh #here we can use jhh only

        random.Random(seed).shuffle(img_name_list)

        if id_list is not None and mode == 'train':
            subset=pd.read_csv(id_list,header=None)[0].tolist()
            img_name_list = [i for i in img_name_list if i in subset]
            print('Using id_list:',id_list)
            print('Length of img_name_list:',len(img_name_list))
            #raise ValueError('Not implemented yet')


        if not all_train:
            length = len(img_name_list)
            test_name_list = img_name_list[:min(200, length//10)]#limited size, or validation will take too long
            train_name_list = list(set(img_name_list) - set(test_name_list))
        else:
            train_name_list = img_name_list
            test_name_list = None
        
        if mode == 'train':
            img_name_list = train_name_list
        else:
            img_name_list = test_name_list

        #print(img_name_list)
        print('Start loading %s data'%self.mode)
        
        

        self.img_list = []
        self.lab_list = []
        self.stage_1 = []
        self.spacing_list = []

        for name in img_name_list:
                
            img_name = name + '.npy'
            lab_name = name + '_gt.npy'
            path = args.jhh_root
            #check is the file exists, if not, try .npz
            if not os.path.exists(os.path.join(path, img_name)):
                img_name = name + '.npz'
                lab_name = name + '_gt.npz'
            if not os.path.exists(os.path.join(path, img_name)):
                raise ValueError('File does not exist, neither as npy nor as npz:', os.path.join(path, img_name))
            

            stage_1_name = name + '_gt.npz'
            
            img_path = os.path.join(path, img_name)
            lab_path = os.path.join(path, lab_name)
            stage_1_name = os.path.join(args.stage_1_path, stage_1_name)

            spacing = np.array((1.0, 1.0, 1.0)).tolist()
            self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order

            self.img_list.append(img_path)
            self.lab_list.append(lab_path)
            self.stage_1.append(stage_1_name)

        self.classes_jhh = classes_jhh

        self.saved_count = 0  # Reset the saved count on instantiation
        print('Load done, length of dataset:', len(self.img_list))
        self.num_classes = len(self.classes_jhh)  # Number of classes for JHH
        
        self.label_order = ['background', 'pancreatic_cyst', 'pancreatic_pdac', 'pancreatic_pnet']
        self.classes=self.label_order
        
        if balance_lesion_types and self.mode == 'train':
            #this part oversamples less common lesion types, for balancing
            jhh_healthy = pd.read_csv(args.jhh_train)
            jhh_healthy = jhh_healthy[jhh_healthy['Diagnosis']=='normal']['BDMAP ID'].tolist()
            jhh_meta = pd.read_csv(args.jhh_meta)
            #filter out: case_names only in IDs
            ids = [extract_id(i) for i in self.img_list]
            jhh_meta = jhh_meta[jhh_meta['case_name'].isin(ids)]
            jhh_healthy = [i for i in jhh_healthy if i in ids]
            ids = balance_ct_scans(jhh_meta,jhh_healthy)
            img_id_map = {extract_id(i): i for i in self.img_list}
            self.img_list = [img_id_map[i] for i in ids]
            self.lab_list = [img_id_map[i].replace('.npy', '_gt.npy').replace('.npz', '_gt.npz') for i in ids]
            self.stage_1 = [os.path.join(args.stage_1_path, i+'.npz/predictions_raw/pancreatic_lesion.npz') for i in ids]
            spacing = np.array((1.0, 1.0, 1.0)).tolist()
            self.spacing_list = []
            for i in range(len(self.img_list)):
                self.spacing_list.append(spacing[::-1])  # itk axis order is inverse of numpy axis order
            print('Number of cases in balanced dataset:', len(self.img_list))

    def __len__(self):
        if self.mode == 'train':
            if self.gigantic_length:
                return len(self.img_list) * 100000
            else:
                return len(self.img_list)
        else:
            return len(self.img_list)
        

    def random_crop(self, tensor_img, tensor_lab, d, h, w):
        tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d+40, h+40, w+40], mode='random')
        if self.args.aug_device == 'gpu':
            tensor_img = tensor_img.cuda(self.args.proc_idx).float()
            tensor_lab = tensor_lab.cuda(self.args.proc_idx).float()
        if np.random.random() < 0.4:
            tensor_img, tensor_lab = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate)
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, self.args.training_size, mode='center')
        else:
            tensor_img, tensor_lab = augmentation.crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
        return tensor_img, tensor_lab
    
    

    def __getitem__(self, idx):
        start = time.time()
        
        
        idx = idx % len(self.img_list)
        #print('Loading:', self.img_list[idx], self.lab_list[idx])
        if self.load_augmented:
            #return self.load_augmented_data(idx)
            try:
                return self.load_augmented_data(idx)#loads and returns data already augmented and pre-saved
            except:
                #change index to another one at random
                idx = np.random.randint(len(self.img_list))
                try:
                    return self.load_augmented_data(idx)
                except:
                    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx], flush=True, file=sys.stderr)
            #    print('FAILED TO LOAD AUGMENTED DATA:', self.img_list[idx], self.lab_list[idx])
            #    pass
            
            
            
        try:
            np_img = np.load(self.img_list[idx], mmap_mode='r', allow_pickle=False)
            if 'npz' in self.img_list[idx]:
                np_img = np_img['arr_0']
        except:
            print('Error loading:', self.img_list[idx])
            try:
                np_img = np.load(self.img_list[idx])
                if 'npz' in self.img_list[idx]:
                    np_img = np_img['arr_0']
            except:
                raise ValueError('Error loading:', self.img_list[idx])
        try:
            np_lab = np.load(self.lab_list[idx], mmap_mode='r', allow_pickle=False)
            if 'npz' in self.lab_list[idx]:
                np_lab = np_lab['arr_0']
        except:
            print('Error loading:', self.lab_list[idx])
            try:
                np_lab = np.load(self.lab_list[idx])
                if 'npz' in self.lab_list[idx]:
                    np_lab = np_lab['arr_0']
            except:
                raise ValueError('Error loading:', self.lab_list[idx])
        
        try:
            stage_1 = np.load(self.stage_1[idx], mmap_mode='r', allow_pickle=False)['arr_0']#this is the output of the lesion detector
        except:
            print('Error loading:', self.stage_1[idx], 'trying another index')
            idx = np.random.randint(len(self.img_list))
            return self.__getitem__(idx)
        #print('Label shape:', np_lab.shape, flush=True, file=sys.stderr)

        sample_name = os.path.basename(self.img_list[idx])
        sample_name = sample_name[:sample_name.rfind('.')]  # remove extension
        
        #chek if jhh
        num_classes = len(self.classes_jhh)
        

        if np_lab.shape[0] != num_classes:
            #print('Label path is:', self.lab_list[idx], flush=True, file=sys.stderr)
            
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            #print('Data type:', np_lab.dtype, flush=True, file=sys.stderr)
            #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
            #start_unpack = time.time()
            # 4. Unpack the bits along the same axis.
            np_lab = np.unpackbits(np_lab, axis=0)
            assert np_lab.shape[0] < (num_classes+10)
            assert np_lab.shape[0] >= (num_classes)
            np_lab = np_lab[:num_classes]
            #print('Label unpacked:', np_lab.shape, flush=True, file=sys.stderr)
            #print('Time to unpack:', time.time() - start_unpack, flush=True, file=sys.stderr)
        if len(stage_1.shape) == 3:
            stage_1 = np.expand_dims(stage_1, axis=0)
        if stage_1.shape[0] != 1:
            if stage_1.shape[0] != num_classes:
                #print('Data type:', np_lab.dtype, flush=True, file=sys.stderr)
                #print('Shape before unpack:', np_lab.shape, flush=True, file=sys.stderr)
                #start_unpack = time.time()
                # 4. Unpack the bits along the same axis.
                stage_1 = np.unpackbits(stage_1, axis=0)
                assert stage_1.shape[0] < (num_classes+10)
                assert stage_1.shape[0] >= (num_classes)
                stage_1 = stage_1[:num_classes]
                
            stage_1 = stage_1[self.classes_jhh.index('pancreatic_lesion')]
            

            
        #get pancreas mask
        pancreas = np_lab[self.classes_jhh.index('pancreas')]
            
        #get only pdac, pnet, cyst and background
        new_label = {}
        new_label['pancreatic_pdac'] = np_lab[self.classes_jhh.index('pancreatic_pdac')]
        new_label['pancreatic_pnet'] = np_lab[self.classes_jhh.index('pancreatic_pnet')]
        new_label['pancreatic_cyst'] = np_lab[self.classes_jhh.index('pancreatic_cyst')]
        lesion = new_label['pancreatic_pdac'] + new_label['pancreatic_pnet'] + new_label['pancreatic_cyst']
        #threshold
        lesion[lesion > 0] = 1
        new_label['background'] = 1 - lesion
        
        tmp = []
        for name in self.label_order:
            tmp.append(new_label[name])
            
        #to make code easier, let's append the stage_1 pancreatic_lesion output to the label
        tmp.append(stage_1.squeeze(0))
        
        new_label = np.stack(tmp, axis=0)
        np_lab = new_label
        
        
        #print(os.path.basename(self.img_list[idx]),self.img_name_list_jhh[:10])
        if self.mode == 'train':
            d, h, w = self.args.training_size
            #np_img, np_lab = augmentation.np_crop_3d(np_img, np_lab, [d+20, h+40, w+40], mode='random')

            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
            pancreas = torch.from_numpy(pancreas).unsqueeze(0).unsqueeze(0)
            stage_1 = torch.from_numpy(stage_1).unsqueeze(0)
            assert len(stage_1.shape) == len(tensor_lab.shape), f'Stage 1 shape {stage_1.shape} does not match label shape {tensor_lab.shape}'
            assert len(pancreas.shape) == len(tensor_lab.shape), f'Pancreas shape {pancreas.shape} does not match label shape {tensor_lab.shape}'
            if len(pancreas.shape) < len(tensor_lab.shape):
                pancreas = pancreas.unsqueeze(0)
            #print('Time to load data:', time.time() - start, flush=True, file=sys.stderr)
            aug_start = time.time()

            del np_img, np_lab

            #pad with zeros if the image is smaller than the training patch size + a little margin
            tensor_img, tensor_lab = augmentation.pad_volume_pair(tensor_img, tensor_lab, d+20, h+40, w+40)
            #np_crop_around_coordinate_3d
            #check if there is a tumor in the image
            stage_1_binary = (stage_1 > 0.5).float()
            tumor_presence = stage_1_binary.sum() > 0
            if not tumor_presence:
                foreground_mask = pancreas
                if foreground_mask.sum() == 0:
                    # Fallback to random crop
                    tensor_img, tensor_lab = self.random_crop(tensor_img, tensor_lab, d, h, w)
                else:
                    #random crop on pancreas with high probability, background with low probability
                    if random.random() < 0.1 or foreground_mask.sum()==0:#random crop on background
                        # Crop around a random background voxel
                        tensor_img, tensor_lab = self.crop_on_coordinate(tensor_img, tensor_lab, 1-foreground_mask, d, h, w)
                    else: #90% chance, crop on pancreas
                        assert pancreas.sum()>0, 'Pancreas mask is empty, this should be impossible'
                        assert torch.equal((pancreas>0.5).float(),pancreas.float()), 'Pancreas mask is not binary'
                        tensor_img, tensor_lab = self.crop_on_coordinate(tensor_img, tensor_lab, foreground_mask, d, h, w)
            else:
                #try to apply erosion to the lesion, if if becomes empty, ignore the eroded mask. The idea is to avoid cropping centered on borders
                eroded_mask = augmentation.denoise_mask(stage_1_binary,iterations=4, connected_component=False)
                if eroded_mask.sum() > 0:
                    foreground_mask = eroded_mask
                else:
                    foreground_mask = stage_1_binary
                assert foreground_mask.sum() > 0, 'Foreground mask is empty, this should be impossible'
                tensor_img, tensor_lab = self.crop_on_coordinate(tensor_img, tensor_lab, foreground_mask, d, h, w)


            tensor_img, tensor_lab = tensor_img.contiguous(), tensor_lab.contiguous()

            if not self.save_augmented:
                #this augmentation is online.
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
                if np.random.random() < 0.3:
                    tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
                if np.random.random() < 0.3:
                    std = np.random.random() * 0.2 
                    tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
        
        else:
            tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0)#.float()
            tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
            assert tensor_img.dtype == torch.float32
            del np_img, np_lab

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        assert tensor_img.shape[1:] == tensor_lab.shape[1:]
        #print('Shapes:',tensor_img.shape, tensor_lab.shape)

        # Save for sanity check
        self.save_sanity_check(tensor_img, tensor_lab, idx)

        # If a save_destination is given, store the augmented sample there as .npy
        if self.save_augmented:
            #print('Shape:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)
            self.save(tensor_img, tensor_lab, idx)
        #print('Time to augment data:', time.time() - aug_start, flush=True, file=sys.stderr)


        
        if self.mode == 'train':
            return tensor_img.float(), tensor_lab.float()
        else:
            return tensor_img.float(), tensor_lab.float(), np.array(self.spacing_list[idx])
        
    def crop_on_coordinate(self, tensor_img, tensor_lab, foreground, d, h, w):
        foreground = foreground.squeeze()  # Ensure foreground is 3D
        #print('Shape of foreground:', foreground.shape, flush=True, file=sys.stderr)
        foreg_voxels = torch.nonzero(foreground == 1, as_tuple=False)    
        # Crop around a random tumor voxel
        center = foreg_voxels[torch.randint(0, len(foreg_voxels), (1,))][0]
        backup_tensor_img = tensor_img.clone()
        backup_tensor_lab = tensor_lab.clone()
        backup_center = center.clone()
        if np.random.random() < 0.4:
            #crop large, then rotate and crop small
            assert len(tensor_lab.shape) == 5
            #crop large on segment
            out = augmentation.crop_around_coordinate_3d(tensor_img, tensor_lab, [d+40, h+40, w+40], coordinate=center, mode='center',foreground=foreground)
            if isinstance(out, tuple):
                tensor_img, tensor_lab, foreground = out
            else:
                return out
            if self.args.aug_device == 'gpu':
                tensor_img = tensor_img.cuda(self.args.proc_idx).float()
                tensor_lab = tensor_lab.cuda(self.args.proc_idx).float()
                foreground = foreground.cuda(self.args.proc_idx).float()
            foreground = foreground.squeeze()  # Ensure foreground is 3D
            tensor_img, tensor_lab, foreground = augmentation.random_scale_rotate_translate_3d(tensor_img, tensor_lab, self.args.scale, self.args.rotate, self.args.translate, foreground=foreground)
            
            #re calculate center    
            try:
                foreg_voxels = torch.nonzero(foreground == 1, as_tuple=False)
                center = foreg_voxels[torch.randint(0, len(foreg_voxels), (1,))][0]
                #print('Center is:', center, flush=True, file=sys.stderr)
                out =  augmentation.crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='center')
            except:
                print('Foreground is empty after rotation, using backup tensor')
                out = augmentation.crop_around_coordinate_3d(backup_tensor_img, backup_tensor_lab, [d, h, w], 
                                                             coordinate=center, mode='center')
                
            
        else:
            #print('Center is:', center, flush=True, file=sys.stderr)
            out = augmentation.crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='center')
        if isinstance(out, tuple):
            tensor_img, tensor_lab = out
            return tensor_img, tensor_lab
        else:
            return out
        
    def save(self, tensor_img, tensor_lab, idx):
        """
        Saves the augmented image/label pair to disk if a destination was specified.
        Uses numpy .npy/.npz format and keeps the original naming scheme.
        """
        os.makedirs(self.save_destination, exist_ok=True)

        # Keep the same filenames as the original
        base_img_name = os.path.basename(self.img_list[idx])   # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])   # e.g. "xxx_gt.npy"

        img_filename = os.path.join(self.save_destination, base_img_name)
        lab_filename = os.path.join(self.save_destination, base_lab_name)

        np_img = tensor_img.cpu().numpy()
        np_lab = tensor_lab.cpu().numpy()
        #print('Is stage 1 binary?', np.unique(np_lab[-1]), flush=True, file=sys.stderr)
        
        print('Sample:', base_img_name, 'Number of classes:', np_lab.shape[0], flush=True, file=sys.stderr)

        # Save as .npy---we can save the crops as npy, they are small, and we can load them faster. But we save the full images as npz.
        np.savez(img_filename.replace('.npy','.npz'), np_img)
        np.savez(lab_filename.replace('.npy','.npz'), np_lab)

        self.save_counter += 1

    def load_augmented_data(self, idx):
        #print('Loading augmented data:', self.img_list[idx], self.lab_list[idx], flush=True, file=sys.stderr)
        # We'll assume the user has already run the dataset once to save the augmented data.
        if self.save_destination is None:
            raise ValueError("load_augmented=True but save_destination=None. Cannot load augmented data.")
        
        

        start = time.time()

        # Derive the filenames from the original naming scheme
        base_img_name = os.path.basename(self.img_list[idx])    # e.g. "xxx.npy"
        base_lab_name = os.path.basename(self.lab_list[idx])    # e.g. "xxx_gt.npy"

        # Replace npz by npy
        base_img_name = base_img_name.replace('.npy','.npz')
        base_lab_name = base_lab_name.replace('.npy','.npz')

        aug_img_path = os.path.join(self.save_destination, base_img_name)
        aug_lab_path = os.path.join(self.save_destination, base_lab_name)
        
        # Load the augmented data
        np_img = np.load(aug_img_path, mmap_mode='r', allow_pickle=False)['arr_0']  # shape as saved
        tensor_img = torch.from_numpy(np_img).unsqueeze(0).unsqueeze(0).float()
        
        #print('Time to load augmented image:', time.time() - start, flush=True, file=sys.stderr)
        start = time.time()
        #print shapes
        #print('Shape:', np_img.shape, np_lab.shape)

        # Convert to torch
        # The code expects image to be float32 and label int8 (for checking).
        np_lab = np.load(aug_lab_path, mmap_mode='r', allow_pickle=False)['arr_0']

        tensor_lab = torch.from_numpy(np_lab).unsqueeze(0)
        #print('Shape of tensor_lab:', tensor_lab.shape, flush=True, file=sys.stderr)

        #print('Time to load augmented label:', time.time() - start, flush=True, file=sys.stderr)
        aug_start = time.time()

        tensor_img = tensor_img.squeeze(0)
        tensor_lab = tensor_lab.squeeze(0)

        
        if self.mode == 'train':
            #this augmentation is online.
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_multiply(tensor_img, multiply_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.brightness_additive(tensor_img, std=0.1)
            if np.random.random() < 0.3:
                tensor_img = augmentation.gamma(tensor_img, gamma_range=[0.7, 1.5])
            if np.random.random() < 0.3:
                tensor_img = augmentation.contrast(tensor_img, contrast_range=[0.7, 1.3])
            if np.random.random() < 0.3:
                tensor_img = augmentation.gaussian_blur(tensor_img, sigma_range=[0.5, 1.5])
            if np.random.random() < 0.3:
                std = np.random.random() * 0.2 
                tensor_img = augmentation.gaussian_noise(tensor_img, std=std)
            #print('Applied augmentation online!')
        
        #print('Augmentation deactivated!')

        # You can still call save_sanity_check if desired
        self.save_sanity_check(tensor_img, tensor_lab, idx)

        #print('Time augmenting data:', time.time() - aug_start, flush=True, file=sys.stderr)

        tensor_img = tensor_img.squeeze(0)

        #print('Shapes:', tensor_img.shape, tensor_lab.shape, flush=True, file=sys.stderr)

        #print('', flush=True, file=sys.stderr)
        #print('Loaded augmented data:', self.img_list[idx], self.lab_list[idx], 'Shape:', tensor_lab.shape, flush=True, file=sys.stderr)
        #print('', flush=True, file=sys.stderr)
        stage1 = tensor_lab[-1]
        #print('Is stage 1 binary in aug data:', torch.unique(stage1), flush=True, file=sys.stderr)
        
        if self.mode == 'train':
            return tensor_img.float(), tensor_lab.float()
        else:
            return tensor_img, tensor_lab, np.array(self.spacing_list[idx])

    def save_sanity_check(self, img, lab, idx):
        """Save the image and labels to NIfTI format for sanity checking."""
        #if self.saved_count == 0 and os.path.exists('./SanityCheckEPAIStage2'):
        #    print('Sanity check folder already exists, removing it.')
        #    shutil.rmtree('./SanityCheckEPAIStage2')
        
        if self.saved_count < 10:
            save_dir = './SanityCheckEPAIStage2'
            os.makedirs(save_dir, exist_ok=True)
            
            name = os.path.basename(self.img_list[idx])

            img_folder = os.path.join(save_dir, name)
            img_folder = img_folder.replace('.nii.gz','').replace('.npz','')
            os.makedirs(img_folder, exist_ok=True)

            # Save the image
            img_nifti = sitk.GetImageFromArray(img.squeeze().cpu().numpy())
            #print shape
            #print('Shape:', img.squeeze().cpu().numpy().shape)
            img_nifti.SetSpacing(self.spacing_list[idx])
            sitk.WriteImage(img_nifti, os.path.join(img_folder, 'CT.nii.gz'))
            
            # Save the labels
            for i, cls in enumerate(self.label_order+['stage_1']):
                label_array = (lab[i].squeeze().cpu().numpy())
                #print(f"  → {cls}: dtype={label_array.dtype}, min={label_array.min()}, max={label_array.max()}")
                if cls != 'stage_1':
                    label_array = label_array.astype(np.int8)
                else:
                    # convert to float32 so viewers and SimpleITK play nicely
                    label_array = label_array.astype(np.float32)
                label_nifti = sitk.GetImageFromArray(label_array)
                label_nifti.SetSpacing(self.spacing_list[idx])
                sitk.WriteImage(label_nifti, os.path.join(img_folder, f'{cls}.nii.gz'))

            self.saved_count += 1

def npy_to_nii(npy_path, nii_path, spacing=(1.0, 1.0, 1.0)):
    """
    Reads a .npy file, converts it to a SimpleITK image, 
    sets spacing, and saves as .nii.gz.

    :param npy_path:    Path to the input .npy file.
    :param nii_path:    Path to the output .nii.gz file.
    :param spacing:     Tuple or list specifying the (z, y, x) spacing. 
                        Default is (1.0, 1.0, 1.0).
    """
    # Load the NumPy array
    array = np.load(npy_path)

    # Convert NumPy array to SimpleITK image
    sitk_image = sitk.GetImageFromArray(array)

    # Optionally set image spacing (if known)
    sitk_image.SetSpacing(spacing)

    # Write to .nii.gz
    sitk.WriteImage(sitk_image, nii_path)


