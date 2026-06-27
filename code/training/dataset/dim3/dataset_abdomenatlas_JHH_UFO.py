import torch
from torch.utils.data import Dataset

from . import dataset_abdomenatlas_JHH_types as atlas_jhh
from . import dataset_abdomenatlas_UFO as atlas_ufo

class AbdomenAtlasDataset(Dataset):
    def __init__(self, args, mode='train', seed=0, all_train=False,
                crop_on_tumor=True,
                 save_destination=None,  
                 load_augmented=False,
                 gigantic_length=False,
                 save_augmented=False,
                 tumor_classes=['pancreas'],
                 balance_supervision=False,
                 balance_datasets=False,
                 atlas_subsample=0.2):    
        super(AbdomenAtlasDataset, self).__init__()
        
        args.UFO_root = '/mnt/bodymaps/temp_data_to_delete_very_soon/psalvad2/UFO_27k_medformerNpz/'
        args.ucsf_ids = 'UCSF_train_set_multi_tumor_no_patient_leakage.csv'
        args.jhh_root = '/mnt/bodymaps/temp_data_to_delete_very_soon/psalvad2/JHH_lesion_types_medformer_npz/'
        args.data_root = '/mnt/bodymaps/temp_data_to_delete_very_soon/psalvad2/AbdomenAtlas3.0MedformerNpz/'
        
        
        self.atlas_dataset = atlas_ufo.AbdomenAtlasDataset(args=args, mode=mode, seed=seed, all_train=all_train,
                crop_on_tumor=crop_on_tumor,
                 save_destination='/home/psalvad2/data/AbdomenAtlas3_UFO27k_MedformerNpzAugmented/',  
                 load_augmented=load_augmented,
                 gigantic_length=False,
                 save_augmented=save_augmented,
                 tumor_classes=tumor_classes,
                 balance_supervision=False, 
                 UFO_only=False,
                 Atlas_only=True)
        
        self.UFO_dataset = atlas_ufo.AbdomenAtlasDataset(args=args, mode=mode, seed=seed, all_train=all_train,
                crop_on_tumor=crop_on_tumor,
                 save_destination='/home/psalvad2/data/AbdomenAtlas3_UFO27k_MedformerNpzAugmented/',  
                 load_augmented=load_augmented,
                 gigantic_length=False,
                 save_augmented=save_augmented,
                 tumor_classes=tumor_classes,
                 balance_supervision=False, 
                 UFO_only=True,
                 Atlas_only=False)
        
        self.JHH_dataset = atlas_jhh.AbdomenAtlasDataset(args=args, mode=mode, seed=seed, all_train=all_train,
                crop_on_tumor=crop_on_tumor,
                 save_destination='/home/psalvad2/data/jhh_augmented_npz/',  
                 load_augmented=load_augmented,
                 gigantic_length=False,
                 save_augmented=save_augmented,
                 id_list=None,
                 JHH_only=True)
        
        # print sizes
        self.length = self.atlas_dataset.__len__() + self.UFO_dataset.__len__() + self.JHH_dataset.__len__()
        print(f'Atlas length: {self.atlas_dataset.__len__()}')
        print(f'UFO length: {self.UFO_dataset.__len__()}')
        print(f'JHH length: {self.JHH_dataset.__len__()}')
        print(f'Total length: {self.length}')
        
        self.ids = list(range(self.length))
        
        if balance_datasets:
            self.length =  3*max(self.atlas_dataset.__len__(), self.UFO_dataset.__len__(), self.JHH_dataset.__len__())
            print(f'Balanced length: {self.length}')
        
        self.balance_datasets = balance_datasets
        
        self.gigantic_length = gigantic_length
        self.mode = mode
        self.classes_JHH = self.JHH_dataset.classes
        self.classes = self.UFO_dataset.classes
        self.num_classes = len(self.classes)
        
        #assert self.num_classes == self.UFO_dataset.num_classes, "UFO and JHH datasets should have the same number of classes."
        #assert self.num_classes == self.atlas_dataset.num_classes, "Atlas and JHH datasets should have the same number of classes."
        
        if balance_supervision:
            raise ValueError("Balance supervision is not implemented.")
        
        self.atlas_subsample = atlas_subsample
        if balance_datasets and self.atlas_subsample < 1:
            raise ValueError("Atlas subsample should be 1 when balancing datasets.")
        
        if self.atlas_subsample < 1:
            self.shuffle_atlas()
            self.length = int(self.atlas_subsample * self.atlas_dataset.__len__()) + self.UFO_dataset.__len__() + self.JHH_dataset.__len__()
        
    def shuffle_atlas(self):
        atlas_length = self.atlas_dataset.__len__()
        atlas_ids = list(range(atlas_length))
        atlas_ids = torch.randperm(atlas_length).tolist()
        self.atlas_ids_map = atlas_ids[:int(self.atlas_subsample*atlas_length)]
        
    def __len__(self):
        if self.mode == 'train':
            if self.gigantic_length:
                return self.length * 100000
            else:
                return self.length
        else:
            return self.length
    
    def __getitem__(self, idx):
        idx = idx % self.length
            
        
        #outputs: tensor_img, tensor_lab, unk_channels_tensor,tumor_volumes_in_crop,chosen_segment_mask,tumor_diameters
        if not self.balance_datasets:
            if idx < int(self.atlas_subsample * self.atlas_dataset.__len__()):
                if self.atlas_subsample < 1:
                    idx = self.atlas_ids_map[idx]
                out = self.atlas_dataset.__getitem__(idx)
                chosen='atlas'
            elif idx < int(self.atlas_subsample * self.atlas_dataset.__len__()) + self.UFO_dataset.__len__():
                out = self.UFO_dataset.__getitem__(idx - int(self.atlas_subsample * self.atlas_dataset.__len__()))
                chosen='UFO'
            else:
                out = self.JHH_dataset.__getitem__(idx - int(self.atlas_subsample * self.atlas_dataset.__len__()) - self.UFO_dataset.__len__())
                chosen='JHH'
        else:
            if idx < int(self.length/3):
                idx = idx % self.atlas_dataset.__len__()
                out = self.atlas_dataset.__getitem__(idx)
                chosen='atlas'
            elif idx < int(2*self.length/3):
                idx = idx - int(self.length/3)
                idx = idx % self.UFO_dataset.__len__()
                out = self.UFO_dataset.__getitem__(idx)
                chosen='UFO'
            elif idx < int(self.length):
                idx = idx - int(2*self.length/3)
                idx = idx % self.JHH_dataset.__len__()
                out = self.JHH_dataset.__getitem__(idx)
                chosen='JHH'
            else:
                raise ValueError("Index out of range.")
        #check how many samples are in out:
        if chosen=='UFO' or chosen=='atlas':
            tensor_img, tensor_lab, unk_channels_tensor,tumor_volumes_in_crop,chosen_segment_mask,tumor_diameters = out
            sample_weights = torch.ones_like(tensor_lab)
        else:
            tensor_img, tensor_lab, unk_channels_tensor = out
            #select only classes in UFO---remove lesion types
            print(f'Shape of tensor_img: {tensor_img.shape}',flush=True)
            assert len(tensor_lab.shape) == 4, f'Expected 4 dimensions, but got {tensor_lab.shape}'
            assert tensor_img.shape[0]==1,f'Batch size should be 1, but got {tensor_img.shape[0]}'
            tensor_lab_tmp, unk_channels_tensor_tmp = [], []
            for class_ufo in sorted(self.classes):
                for i,class_jhh in enumerate(sorted(self.classes_JHH),0):
                    if class_ufo not in self.classes_JHH:
                        raise ValueError(f"Class {class_ufo} not in JHH dataset.")
                    if class_jhh == class_ufo:
                        tensor_lab_tmp.append(tensor_lab[i])
                        unk_channels_tensor_tmp.append(unk_channels_tensor[i])
            tensor_lab = torch.stack(tensor_lab_tmp, dim=0)
            unk_channels_tensor = torch.stack(unk_channels_tensor_tmp, dim=0)
            tumor_volumes_in_crop=[0,0,0,0,0,0,0,0,0,0]
            tumor_volumes_in_crop=torch.tensor(tumor_volumes_in_crop).float()
            chosen_segment_mask = torch.zeros(tensor_lab.shape).type_as(tensor_lab).float()
            tumor_diameters=torch.zeros((10,3)).float()
            sample_weights = torch.ones_like(tensor_lab)
              
        retur =    {"image":      tensor_img,
                    "label":           tensor_lab,
                    "unk_channels":    unk_channels_tensor,
                    "volumes":         tumor_volumes_in_crop,
                    "mask":            chosen_segment_mask,
                    "diameters":       tumor_diameters,
                    "weights":         sample_weights
                    }  
        return retur