import os, re
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AlbertTokenizer

class HIVDataset(Dataset):
    def __init__(self, dataset_name, tokenizer_name, split=None, dataset_dir='newHIV-1_data'):
        """
        dataset_name: name of dataset. Should be one of ['1625', '746', 'imens', 'schillings']
        split: split mode. Should be one of ['train', 'test', None]. If None, do not split.
        """
        filename = dataset_name + 'Data.txt'
        with open(os.path.join(dataset_dir, filename), 'r') as f:
            lines = f.readlines()
            
        self.octamers = [] # sequence of 8 amino acids
        self.labels = [] # -1: uncleaved, 1: cleaved
        for line in lines:
            octamer, label = line.strip().split(',')
            self.octamers.append(octamer)
            self.labels.append(int((int(label) + 1) / 2))

        self.octamers, self.labels = np.array(self.octamers), np.array(self.labels)

        if split is not None:
            pos_idx = np.where(self.labels == 1)[0]
            neg_idx = np.where(self.labels == 0)[0]
            if split == 'train': 
                self.octamers = np.concatenate([
                    self.octamers[pos_idx[:int(len(pos_idx)*0.8)]],
                    self.octamers[neg_idx[:int(len(neg_idx)*0.8)]]
                ])
                self.labels = np.concatenate([
                    self.labels[pos_idx[:int(len(pos_idx)*0.8)]],
                    self.labels[neg_idx[:int(len(neg_idx)*0.8)]]
                ])
            elif split == 'test':
                self.octamers = np.concatenate([
                    self.octamers[pos_idx[int(len(pos_idx)*0.8):]],
                    self.octamers[neg_idx[int(len(neg_idx)*0.8):]]
                ])
                self.labels = np.concatenate([
                    self.labels[pos_idx[int(len(pos_idx)*0.8):]],
                    self.labels[neg_idx[int(len(neg_idx)*0.8):]]
                ])

        self.tokenizer = self.build_tokenizer(tokenizer_name)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        octamer, label = self.octamers[idx], self.labels[idx]

        octamer = " ".join(octamer)
        octamer = re.sub(r"[UZOB]", "X", octamer) # U, Z, O, B -> X
        octamer_ids = self.tokenizer(octamer)

        sample = {key: torch.tensor(val) for key, val in octamer_ids.items()}
        sample['labels'] = torch.tensor(label) # -1, 1 -> 0, 1
        
        return sample

    def build_tokenizer(self, tokenizer_name):
        return AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)

class HIVDataset_Albert(HIVDataset):
    def build_tokenizer(self, tokenizer_name):
        return AlbertTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)