import torch
from torch.utils.data import Dataset, DataLoader

from pytorch_metric_learning import miners
from pytorch_metric_learning.distances import CosineSimilarity

import random


class ColorText_dataset(Dataset):
    
    def __init__(self, data_dict, image_features, shuffle=False, random_seed=1234):

        order = list(range(len(data_dict)))
        if shuffle:
            random.seed(random_seed)
            random.shuffle(order)

        self.data_dict = []
        for o in order:
            self.data_dict.append(data_dict[o])
        
        self.image_features = image_features[order]
        
        self.unique_classes = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "BLACK"]
        self.class2id = {c:i for i, c in enumerate(self.unique_classes)}
        self.classes = torch.tensor([self.class2id[sample['font_color']] for sample in data_dict], dtype=int)

        self.unique_attrs = ["RED", "GREEN", "BLUE", "YELLOW", "PURPLE", "TEXT"]
        self.attr2id = {a:i for i, a in enumerate(self.unique_attrs)}
        self.attrs = torch.tensor([self.attr2id[sample['text']] for sample in data_dict], dtype=int)

        self.label_tuples = torch.stack([self.classes, self.attrs], dim=1)

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        return self.image_features[idx], self.classes[idx], self.attrs[idx]



class Miner():
    def __init__(self, distance='Cosine', default_miner='TripletMarginMiner', triplet_margin=0.2):
        # return tensors of anchor_ids, positive_ids, negative_ids
        if distance == 'Cosine':
            self.distance = CosineSimilarity()
        else:
            raise NotImplementedError

        self.default_miner = default_miner
        if self.default_miner == 'TripletMarginMiner':
            self.triplet_miner = miners.TripletMarginMiner( distance = self.distance, margin=triplet_margin )
        elif self.default_miner == 'BatchHardMiner':
            self.triplet_miner = miners.BatchHardMiner( distance = self.distance )

    def _mine_with_TripletMiner(self, target_embeds, sa_embeds, classes, attributes):

        class_triplets = torch.stack(self.triplet_miner(target_embeds, classes)).T # [n_triplets, 3]
        attr_triplets = torch.stack(self.triplet_miner(sa_embeds, attributes)).T # [n_triplets, 3]

        target_anc_embeds = target_embeds[class_triplets[:, 0]] 
        target_pos_embeds = target_embeds[class_triplets[:, 1]]
        target_neg_embeds = target_embeds[class_triplets[:, 2]] 
        
        sa_anc_embeds = sa_embeds[attr_triplets[:, 0]]
        sa_pos_embeds = sa_embeds[attr_triplets[:, 1]]d
        sa_neg_embeds = sa_embeds[attr_triplets[:, 2]]


        return target_anc_embeds, target_pos_embeds, target_neg_embeds,\
                sa_anc_embeds, sa_pos_embeds, sa_neg_embeds

    def _mine_with_TupleMiner(self,):
        # when using default miner as tuple miner
        # returning (pos_anc, pos, neg_anc, neg)
        raise NotImplementedError

    def mine(self, target_embeds, sa_embeds, classes, attributes):
        # target_embeds / sa_embeds = [batch_size, embed_dim]
        # classes / attributes = [batch_size]
        
        if self.default_miner == 'TripletMarginMiner' or self.default_miner == 'BatchHardMiner':
            return self._mine_with_TripletMiner(target_embeds, sa_embeds, classes, attributes)

