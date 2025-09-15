import torch
from torch.utils.data import Dataset

import os
from itertools import islice
from math import inf

import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling # for ChemBERTa
from transformers import OpenAIGPTForSequenceClassification, DataCollatorWithPadding

import logging

class ProcessedDataset(Dataset):
    """
    Data structure for a pre-processed cormorant dataset.  Extends PyTorch Dataset.

    Parameters
    ----------
    data : dict
        Dictionary of arrays containing molecular properties.
        {num_atoms: torch.tensor([2,3,4,5,2,3,2, ..]),
         ...,
         smiles: np.array(['C', 'CCNH3', ..]), 
         }
         
    included_species : tensor of scalars, optional
        Atomic species to include in ?????.  If None, uses all species.
    num_pts : int, optional
        Desired number of points to include in the dataset.
        Default value, -1, uses all of the datapoints.
    normalize : bool, optional
        ????? IS THIS USED?
    shuffle : bool, optional
        If true, shuffle the points in the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.

    """
    def __init__(self, data, included_species=None, num_pts=-1, normalize=True, shuffle=True, subtract_thermo=True, tokenizer=None, cls_rep_3d='virtual_atom'):

        self.data = data
        # print('inside processeddataset data: ', data)
        # for mol in self.data['charges'].tolist():
        #     print(mol)

        # need to add atom x_0 if we take the CLS virtual atom representation in the model
        if cls_rep_3d=='virtual_atom':
            # add charge '-1' to every molecule charge list for atom x_0
            print('charges before virtual: ', self.data['charges'], self.data['charges'].shape)
            nr_mols = self.data['charges'].shape[0]
            virtual_atom_charges = torch.ones(nr_mols, 1)*-1
            self.data['charges'] = torch.cat([virtual_atom_charges, self.data['charges']], dim=-1)
            print('charges before virtual: ', self.data['charges'], self.data['charges'].shape)

            # add 1 to every 'num_atoms' value
            self.data['num_atoms'] += 1

            # add center of each molecule as coordinates[0] to each molecule
            bs, n_atoms, coord_nf = self.data['positions'].shape
            virtual_atom_coords = self.data['positions'].mean(dim=1, keepdim=True)
            self.data['positions'] = torch.cat([virtual_atom_coords, self.data['positions']], dim=1)

            # check if coordinates were modified correctly
            bs_n,n_atoms_n,coord_nf_n = self.data['positions'].shape
            assert [bs_n,n_atoms_n,coord_nf_n] == [bs, n_atoms+1, coord_nf]


        ## added
        # defining LM tokenizer as passed from utils.py
        self.tokenizer = tokenizer
        
        # defining full smiles list
        self.smiles_list = self.data['smiles']

        # we extracted the smiles, now delete the string smiles version
        self.dict_1d = {'smiles': self.smiles_list}

        del self.data['smiles']
        ##


        if num_pts < 0:
            self.num_pts = len(data['charges'])
        else:
            if num_pts > len(data['charges']):
                logging.warning('Desired number of points ({}) is greater than the number of data points ({}) available in the dataset!'.format(num_pts, len(data['charges'])))
                self.num_pts = len(data['charges'])
            else:
                self.num_pts = num_pts

        # If included species is not specified
        if included_species is None:
            included_species = torch.unique(self.data['charges'], sorted=True)
            if included_species[0] == 0:
                included_species = included_species[1:]

        if subtract_thermo:
            thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
            if len(thermo_targets) == 0:
                logging.warning('No thermochemical targets included! Try reprocessing dataset with --force-download!')
            else:
                logging.info('Removing thermochemical energy from targets {}'.format(' '.join(thermo_targets)))
            for key in thermo_targets:
                data[key] -= data[key + '_thermo'].to(data[key].dtype)

        self.included_species = included_species

        self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)

        self.num_species = len(included_species)
        self.max_charge = max(included_species)

        self.parameters = {'num_species': self.num_species, 'max_charge': self.max_charge}

        # Get a dictionary of statistics for all properties that are one-dimensional tensors.
        self.calc_stats()

        if shuffle:
            self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
        else:
            self.perm = None

    ## added
    def preprocess(self, feature_dict):
        # print('feature dict: ', feature_dict)
        batch_encoding = self.tokenizer(
            feature_dict["smiles"],
            padding='max_length',
            add_special_tokens=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            # return_attention_mask=True -> we do not need to pass this, as it seems to be part of the tokenizer's output by default!
        )
        return {'input_ids': torch.tensor(batch_encoding["input_ids"]), 'attention_mask': torch.tensor(batch_encoding["attention_mask"])}
    ##

    def calc_stats(self):
        self.stats = {key: (val.mean(), val.std()) for key, val in self.data.items() if type(val) is torch.Tensor and val.dim() == 1 and val.is_floating_point()}

    def convert_units(self, units_dict):
        for key in self.data.keys():
            if key in units_dict:
                self.data[key] *= units_dict[key]

        self.calc_stats()

    def __len__(self):
        return self.num_pts

    def __getitem__(self, idx):
        if self.perm is not None:
            idx = self.perm[idx]

        ## added
        smiles_dict = {key: val[idx] for key, val in self.dict_1d.items()}
        # contains input_ids and attention_mask
        tokenized_1d_dict = self.preprocess(smiles_dict)
        
        # simply took dict3d out of return statement
        dict_3d = {key: val[idx] for key, val in self.data.items()}

        dict_combined = {'1d': tokenized_1d_dict, '3d': dict_3d}
        ##

        # dict_3d['input_ids'] = tokenized_1d_dict['input_ids']
        # dict_3d['attention_mask'] = tokenized_1d_dict['attention_mask']
        return dict_combined
