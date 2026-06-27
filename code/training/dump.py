
class choose_organ_class_match_tumor_reports_deprecated():
    def __init__(self,lesion_classes,class_names,
             alpha=0.1,epsilon=1e-4):
        """
        Unlike the previous class, this one is used for report superivsion. So, we crop on organ segments with tumor, not the tumor themselves.
        Initialize the state for the class. This uses a P controller to make the organ crop distribution match the tumor crop distribution.
        The P controller is useful especially if not all organs appear every time. Say an organ is rare, the P controller will get high error in the
        samples it does not appear, and it will compensate in the samples it appear (giving it strong preference in those samples).
        """
        
        #get organs with tumors from lesion_classes
        organs_with_tumors = []
        for clss in lesion_classes:
            organ_name = clss.replace('_lesion', '').replace('_pdac','').replace('_pnet','').replace('_cyst','').replace('pancreatic','pancreas')
            if 'kidney' in clss or 'lung' in clss or 'adrenal_gland' in clss or 'femur' in clss:
                organs_with_tumors.append(organ_name+'_right')
                organs_with_tumors.append(organ_name+'_left')
            elif 'pancreas' in clss:
                organs_with_tumors.append('pancreas')
                organs_with_tumors.append('pancreas_head')
                organs_with_tumors.append('pancreas_body')
                organs_with_tumors.append('pancreas_tail')
            elif 'liver' in clss:
                organs_with_tumors.append('liver')
                for i in range(1, 9):
                    organs_with_tumors.append('segment_'+str(i))
            elif clss == 'gallbladder_lesion':
                organs_with_tumors.append('gall_bladder')
            elif clss == 'adrenal_lesion':
                organs_with_tumors.append('adrenal_gland_right')
                organs_with_tumors.append('adrenal_gland_left')
            elif clss == 'uterus_lesion':
                organs_with_tumors.append('prostate')
            else:
                organs_with_tumors.append(organ_name)
        print('organs_with_tumors:', organs_with_tumors)
                
        
        tumor_proportions = {}
        for clss in organs_with_tumors:
            tumor_proportions[clss] = 1/len(organs_with_tumors) # initialize with 1.0 for all classes, this will be used for sampling
        self.organ_proportions={}
        for key, value in tumor_proportions.items():
            organ_name = self.tumor_to_organ(key)
            if isinstance(organ_name, str):
                self.organ_proportions[organ_name] = value
            else:
                for o in organ_name:
                    self.organ_proportions[o] = value / len(organ_name)
                    
                    
        self.p_sample = copy.deepcopy(self.organ_proportions)    
        self.tumor_proportions = tumor_proportions
        
        self.lesion_classes = lesion_classes
        self.lesion_class_indices = []
        for i,clss in enumerate(class_names):
            if clss in self.lesion_classes:
                self.lesion_class_indices.append(i)
        print('Lesion class indices UFO (should be empty, as UFO_classes have no lesion):', self.lesion_class_indices)
                
        #raise ValueError('Lesion class indices:', self.lesion_class_indices, 'Classes:', class_names)
    
            
        self.class_names = class_names
        self.alpha = alpha 
        self.epsilon = epsilon
        


    def choose_organ_class(self,
                    possibilities,
                    update_EMA=True,
                ):
        """
        This is similar to a P controller. The idea is to keep track of the organ_proportions and tumor_proportions,
        and adjust the organ sampling probabilities (p_sample) according to the differences between the two.
        p_sample should make organ_proportions match tumor_proportions over time.
        """
        
        #both organ and tumor proportions use the same keys: organs
        p_target = self.tumor_proportions
        
        #now we have our target organ proportions in p_target
        
        # 1) For each organ in p_target, measure difference vs. the current organ proportions
        for organ in p_target.keys():
            tp = p_target.get(organ, 0.0)
            op = self.organ_proportions.get(organ, 0.0)
            diff = tp - op
            # Update p_sample, diff 
            self.p_sample[organ] = self.p_sample.get(organ, 0.0) + self.alpha*diff

        # 2) Ensure all p_sample values are at least epsilon, so none is fully zero.
        #    This also keeps it stable if the difference is negative enough to cross zero.
        for organ in self.p_sample:
            self.p_sample[organ] = max(self.p_sample[organ], self.epsilon)

        # 3) Re-normalize p_sample so it sums to 1
        sum_p = sum(self.p_sample.values())
        if sum_p < self.epsilon:
            # fallback: uniform
            n_orgs = len(self.p_sample)
            for organ in self.p_sample:
                self.p_sample[organ] = 1.0 / n_orgs
        else:
            for organ in self.p_sample:
                self.p_sample[organ] /= sum_p

        # 4) Now we figure out the sampling probability among the 'possibilities' we have.
        #    For each possibility, find the organ name, then read p_sample[organ_name].
        weights = []
        orgnames = []
        for idx in possibilities:
            organ_name = self.class_names[idx]
            orgnames.append(organ_name)
            prob = self.p_sample.get(organ_name, self.epsilon)  # if missing, fallback to epsilon
            weights.append(prob)

        # 5) Use torch.multinomial to choose one possibility
        weights_tensor = torch.tensor(weights, dtype=torch.float)
        total = weights_tensor.sum()
        if total <= self.epsilon:
            # fallback: uniform among possibilities
            weights_tensor = torch.ones_like(weights_tensor) / len(weights_tensor)
        else:
            weights_tensor = weights_tensor / total

        chosen_idx_in_possibilities = torch.multinomial(weights_tensor, 1).item()
        chosen_idx = possibilities[chosen_idx_in_possibilities]
        
        if update_EMA:
            #update the organ_proportions
            self.organ_proportions = self.update_crop_proportions_EMA(self.organ_proportions, self.class_names[chosen_idx])
            print('Organ proportions:', self.organ_proportions)

        return chosen_idx
    
    
    def choose_tumor_class(self, possibilities, epsilon=1e-6, update_EMA=False):
        """
        Selects one tumor class (by index) from possibilities, favoring those with smaller crop proportions.
        
        Parameters:
        possibilities (list of str): Candidate tumor classes that are present.
        class_names (list of str): Mapping from index to tumor class name.
        epsilon (float): A small constant to avoid division by zero.
        
        Returns:
        int: The chosen tumor class index (from possibilities).
                You can then look up its name using class_names[chosen_index].
        """
        weights = []
        tumors = []
        for cls_name in possibilities:
            proportion = self.tumor_proportions.get(cls_name, 0.0)
            weight = 1.0 / (proportion + epsilon)
            weights.append(weight)
            tumors.append(cls_name)
            
        # Convert weights to tensor and normalize them.
        weights = torch.tensor(weights, dtype=torch.float)
        weights = weights / weights.sum()
        
        # Use torch.multinomial to sample one index according to the computed weights.
        chosen_idx = torch.multinomial(weights, num_samples=1).item()
        chosen_tumor = tumors[chosen_idx]
        if update_EMA:
            self.tumor_proportions = self.update_crop_proportions_EMA(self.tumor_proportions, chosen_tumor)
            print('Tumor proportions:', self.tumor_proportions)
            return possibilities[chosen_idx]
    
    # Suppose crop_proportions is a dict mapping class id to its current moving average.
    # And for the current sample, the chosen_class gets new_value = 1, and for others new_value = 0.
    def update_crop_proportions_EMA(self,crop_proportions, chosen_class, alpha=0.01):
        def update_moving_average(proportion, new_value, alpha=0.01):
            return alpha * new_value + (1 - alpha) * proportion
        
        print('EMA chosen class:', chosen_class)
        
        for cls in crop_proportions.keys():
            new_val = 1.0 if cls == chosen_class else 0.0
            crop_proportions[cls] = update_moving_average(crop_proportions[cls], new_val, alpha)
            if new_val==1.0:
                print('EMA updated class:', chosen_class)
            
        return crop_proportions

    def tumor_to_organ(self,tumor_name):
        """
        Convert a tumor class name to an organ class name:
        - Remove '_lesion'
        - Substitutes some known patterns (like 'pancreatic' -> 'pancreas')
        - Randomly chooses a sided version for kidney, adrenal_gland, lung, or femur.
        """
        base = tumor_name.replace('_lesion', '')
        lower_base = base.lower()
        if lower_base == 'pancreatic':
            return 'pancreas'
        elif lower_base == 'kidney':
            return ['kidney_right', 'kidney_left']
        elif lower_base == 'adrenal_gland':
            return ['adrenal_gland_right', 'adrenal_gland_left']
        elif lower_base == 'lung':
            return ['lung_right', 'lung_left']
        elif lower_base == 'femur':
            return ['femur_right', 'femur_left']
        else:
            return base
        
    def __call__(self, tensor_img, tensor_lab, d, h, w, tumor_case, tumor_prob=None, foreground_prob=None, background_prob=None):
        """
        The idea here is: we want similar crops for tumor and halthy patients. Then, if a dataset has balanced positive/negative cases,
        we will get similar numbers of crops on pancreas with and without tumor, liver with and without tumor,...
        tumor_proportions is a dict, which stores the percentages of each type of tumor crop in the positive patients, as moving averages.
        
        As a bonus, in tumor crops we use tumor_proportions to favor crops on rarer tumors (e.g., in a CT with a liver and a prostate tumor, 
        crop on prostate).
        """
        rnd = np.random.random()
        if (tumor_prob is None) or (foreground_prob is None) or (background_prob is None):
            #standard probs
            if tumor_case:
                tumor_prob=0.8
                foreground_prob=0.1
                background_prob=0.1
                print('Tumor case')
            else:
                tumor_prob=0
                foreground_prob=0.9
                background_prob=0.1
                print('Non-tumor case')

        if rnd < tumor_prob:
            raise ValueError('This should not be called, use this funcrtion for negative cases, not to crop on tumors')
        elif rnd < (tumor_prob + background_prob):
            # Negative crop
            foreground_mask = tensor_lab[0].sum(0, keepdim=False)
            back_voxels = torch.nonzero(foreground_mask == 0, as_tuple=False)
            #add also organs without tumor
            other_organ_voxels = torch.zeros_like(foreground_mask)
            for i,name in enumerate(self.class_names,0):
                if name not in self.p_sample.keys():
                    other_organ_voxels += tensor_lab[0][i]
            other_organ_voxels = torch.nonzero(other_organ_voxels, as_tuple=False)
            #50% chance of cropping on other_organ_voxels
            if np.random.random() < 0.5:
                back_voxels = other_organ_voxels
                
            if len(back_voxels) == 0:
                # Fallback to random crop
                tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
            else:
                # Crop around a random background voxel
                center = back_voxels[torch.randint(0, len(back_voxels), (1,))][0]
                tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
        else:
            # Random label crop - here, we try to crop on organs in the same proportion as the tumor crops - if on unhealthy patients 80% of crops 
            # are on pancreas, I want that, on healthy patients, 80% of crops are on pancreas too.
            foreground_mask = tensor_lab[0]#.sum(0, keepdim=False)  --- do not sum, it will favor large organs! This makes little sense.
            #get organ classes
            possibilities = []
            for i in list(range(tensor_lab.shape[1])):
                if i not in self.lesion_class_indices and (torch.sum(tensor_lab[0][i])>0) and (self.class_names[i] in self.p_sample.keys()):
                    possibilities.append(i)
            if len(possibilities) == 0:
                # Fallback to random crop
                tensor_img, tensor_lab = crop_3d(tensor_img, tensor_lab, [d, h, w], mode='random')
            else:
                #randomly choose one of the organ classes
                chosen_class = self.choose_organ_class(possibilities)
                foreground_mask = foreground_mask[chosen_class]
                foreground_voxels = torch.nonzero(foreground_mask)
                center = foreground_voxels[torch.randint(0, len(foreground_voxels), (1,))][0]
                tensor_img, tensor_lab = crop_around_coordinate_3d(tensor_img, tensor_lab, [d, h, w], coordinate=center, mode='small_rnd_shift')
                
        return tensor_img, tensor_lab