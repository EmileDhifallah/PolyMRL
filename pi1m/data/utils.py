import torch
import numpy as np

import logging
import os

from torch.utils.data import DataLoader
from pi1m.data.dataset_class import ProcessedDataset
from pi1m.data.prepare import prepare_dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling # for ChemBERTa
from transformers import OpenAIGPTForSequenceClassification, DataCollatorWithPadding


def initialize_datasets(args, datadir, dataset, subset=None, splits=None,
                        force_download=True, subtract_thermo=False,
                        remove_h=False, tokenizer=None, cls_rep_3d='virtual_atom',
                        train_mode='pretrain'):
    """
    Initialize datasets.

    Parameters
    ----------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datadir : str
        Path to the directory where the data and calculations and is, or will be, stored.
    dataset : str
        String specification of the dataset.  If it is not already downloaded, must currently by "qm9" or "md17".
    subset : str, optional
        Which subset of a dataset to use.  Action is dependent on the dataset given.
        Must be specified if the dataset has subsets (i.e. MD17).  Otherwise ignored (i.e. GDB9).
    splits : str, optional
        TODO: DELETE THIS ENTRY
    force_download : bool, optional
        If true, forces a fresh download of the dataset.
    subtract_thermo : bool, optional
        If True, subtracts the thermochemical energy of the atoms from each molecule in GDB9.
        Does nothing for other datasets.
    remove_h: bool, optional
        If True, remove hydrogens from the dataset
    Returns
    -------
    args : dict
        Dictionary of input arguments detailing the cormorant calculation.
    datasets : dict
        Dictionary of processed dataset objects (see ????? for more information).
        Valid keys are "train", "test", and "valid"[ate].  Each associated value
    num_species : int
        Number of unique atomic species in the dataset.
    max_charge : pytorch.Tensor
        Largest atomic number for the dataset.

    Notes
    -----
    TODO: Delete the splits argument.
    """
    print('initialize_datasets called')
    # Set the number of points based upon the arguments
    num_pts = {'train': args.num_train,
               'test': args.num_test, 'valid': args.num_valid}

    # Download and process dataset. Returns datafiles.
    # note: --> the prepare_dataset function returns dict {train: train.npz filepath, ...} i.e. the train/valid/test npz file names
    
    # note 2: returns ONLY datafile paths! not the data sets/objects themselves
    # note: prepare_dataset calls download func. which calls process_xyz_files()
    datafiles = prepare_dataset(
        datadir, 'qm9', subset, splits, force_download=force_download)

    # Load downloaded/processed datasets
    # note: loading all .npz files in here, same data as printed in .out logs as 'datasets as input: ...'
    datasets = {}
    for split, datafile in datafiles.items():
        with np.load(datafile) as f:
            # turn train.npz into a numpy array, into a dict, with the feature name as the key and the feature values as a numpy array (numpy version of a list);
            # then, the numpy array is turned into a torch array (=a tensor), so we get for train.npz a dict with {atom_numbers: torch.tensor([2,4,6,3,4,2,3, ...]), ... };
            datasets[split] = {key: torch.from_numpy(
                val) if not isinstance(val[0], np.str_) else val for key, val in f.items()}
    #note: intervene here, ensure that our property and PI1M datasets are in agreement w.r.t. the features/keys in the datasets (w the qm9 dataset);
    print('datasets as input: ', datasets)

    if dataset not in ["qm9", "Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat", "PI1M"]:
        np.random.seed(42)
        fixed_perm = np.random.permutation(len(datasets['train']['num_atoms']))
        if dataset == 'qm9_second_half':
            sliced_perm = fixed_perm[len(datasets['train']['num_atoms'])//2:]
        elif dataset == 'qm9_first_half':
            sliced_perm = fixed_perm[0:len(datasets['train']['num_atoms']) // 2]
        else:
            raise Exception('Wrong dataset name')
        for key in datasets['train']:
            datasets['train'][key] = datasets['train'][key][sliced_perm]

    # Basic error checking: Check the training/test/validation splits have the same set of keys.
    keys = [list(data.keys()) for data in datasets.values()]
    assert all([key == keys[0] for key in keys]
               ), 'Datasets must have same set of keys!'

    # TODO: remove hydrogens here if needed
    if remove_h:
        for key, dataset in datasets.items():

            pos = dataset['positions']
            charges = dataset['charges']
            num_atoms = dataset['num_atoms']
            smiles = dataset['smiles'] # this is a numpy array with smiles strings in np.str_ data format

            # Check that charges corresponds to real atoms -> assert that there are 0 molecule entries where the nr. of atoms is unequal to the number of positive charges in dataset[molecule]['charges'];
            assert torch.sum(num_atoms != torch.sum(charges > 0, dim=1)) == 0

            mask = dataset['charges'] > 1
            new_positions = torch.zeros_like(pos)
            new_charges = torch.zeros_like(charges)
            for i in range(new_positions.shape[0]):
                m = mask[i]
                p = pos[i][m]   # positions to keep
                p = p - torch.mean(p, dim=0)    # Center the new positions
                c = charges[i][m]   # Charges to keep
                n = torch.sum(m)
                new_positions[i, :n, :] = p
                new_charges[i, :n] = c

            dataset['positions'] = new_positions # without hydrogen atoms
            dataset['charges'] = new_charges # without hydrogen atoms
            dataset['num_atoms'] = torch.sum(dataset['charges'] > 0, dim=1)
            dataset['smiles'] = smiles

    # Get a list of all species across the entire dataset
    all_species = _get_species(datasets, ignore_check=False, cls_rep_3d=cls_rep_3d, train_mode=train_mode) # the species are simply all unique values of the 'charges' feature
    print('all_species: ', all_species)


    ## added
    tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1") # (must match model vocab and tokenization style)
    # tokenizer_path = "seyonec/PubChem10M_SMILES_BPE_450k"
    # tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path) # **tokenizer_kwargs, local_files_only=True)

    # get max length by sampling 100 longest smiles strings from current dataset
    # first get single list with all dataset's smiles strings:
    smiles_list = []
    for data in datasets.values():
        # print('smiles: ', data['smiles'], type(data['smiles']))
        smiles_list += data['smiles'].tolist()

    smiles_biggest_100_idxs = np.argsort([len(smiles) for smiles in smiles_list])[-100:] # indices from smallest to longest
    smiles_biggest_100 = [smiles_list[smiles_idx] for smiles_idx in smiles_biggest_100_idxs]
    # this should give the longest tokenized sequence length of the dataset
    tokenizer_max_length = max([len(tokenizer(smiles)["input_ids"]) for smiles in smiles_biggest_100])

    # tokenizer.model_max_length = tokenizer_max_length
    print('tokenizer cls: ', tokenizer.cls_token)
    print('tokenizer special: ', tokenizer.special_tokens_map)
    ##

    for name, token in tokenizer.special_tokens_map.items():
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"{name}: {token} --> ID {token_id}")

    # Now initialize MolecularDataset based upon loaded data --> simply a {train: 'ProcessedDataset' obj; valid: ...} dictionary?
    # note: --> this (ProcessedDataset initial. and get_item function) should work with our SMILES column added to the datasets/dicts;
    # we can now use datasets, which is {train: ProcessedDataset, valid: ...}, to sample data instances using a specific index with; this way,
    # our torch DataLoader object, and our Data Collator function, can create batches of multiple data idxs by sampling from the datasets objects,
    # sending the batch into the forward function.
    datasets = {split: ProcessedDataset(data, num_pts=num_pts.get(
        split, -1), included_species=all_species, subtract_thermo=subtract_thermo, tokenizer=tokenizer, cls_rep_3d=cls_rep_3d) for split, data in datasets.items()}

    # Now initialize MolecularDataset based upon loaded data

    # Check that all datasets have the same included species:
    assert(len(set(tuple(data.included_species.tolist()) for data in datasets.values())) ==
           1), 'All datasets must have same included_species! {}'.format({key: data.included_species for key, data in datasets.items()})

    # These parameters are necessary to initialize the network
    num_species = datasets['train'].num_species # for context length, or ...?
    max_charge = datasets['train'].max_charge

    # Now, update the number of training/test/validation sets in args --> does this mean data points per set?
    args.num_train = datasets['train'].num_pts
    args.num_valid = datasets['valid'].num_pts
    args.num_test = datasets['test'].num_pts

    # return dataset dictionary, the used args, the #unique values in charges of the train dataset, and the maximum value in charges of the train dataset
    return args, datasets, num_species, max_charge, tokenizer


def _get_species(datasets, ignore_check=False, cls_rep_3d='naive', train_mode='pretrain'):
    """
    Generate a list of all species.

    Includes a check that each split contains examples of every species in the
    entire dataset.

    Parameters
    ----------
    datasets : dict
        Dictionary of datasets.  Each dataset is a dict of arrays containing molecular properties.
    ignore_check : bool
        Ignores/overrides checks to make sure every split includes every species included in the entire dataset

    Returns
    -------
    all_species : Pytorch tensor
        List of all species present in the data.  Species labels shoguld be integers.

    """
    PI1M_species = torch.tensor([-2,  1,  5,  6,  7,  8,  9, 11, 14, 15, 16, 17, 26, 32, 33, 34, 35, 50, 53])

    # Get a list of all species in the dataset across all splits
    all_species = torch.cat([dataset['charges'].unique()
                             for dataset in datasets.values()]).unique(sorted=True)

    # Find the unique list of species in each dataset.
    split_species = {split: species['charges'].unique(
        sorted=True) for split, species in datasets.items()}


    # If zero charges (padded, non-existent atoms) are included, remove them
    if 0 in all_species:
        index_0 = (all_species==0).nonzero(as_tuple=True)[0].item()
        all_species = torch.cat([all_species[:index_0], all_species[index_0+1:]])

    # if all_species[0] == 0:
    #     all_species = all_species[1:]
    print('all_species:', all_species)

    # Remove zeros if zero-padded charges exst for each split
    for split, species in split_species.items():
        if 0 not in species:
            split_species[split] = species
        else:
            index_0 = (species==0).nonzero(as_tuple=True)[0].item()
            species = torch.cat([species[:index_0], species[index_0+1:]])
            split_species[split] = species

    # split_species = {split: species[1:] if species[0] ==
    #                  0 else species }

    # use the union of the pretrained -PI1M- atom dictionary and the property prediction dataset's atom dict if we are in property_prediction
    print(train_mode)
    if train_mode=='property_prediction':
        all_species = torch.cat([all_species, PI1M_species]).unique().sort()[0]
        split_species = {split: torch.cat([species, PI1M_species]).unique().sort()[0] for split, species in split_species.items()}


    print('split species: ', split_species)


    # if we add virtual atom x_0, add -1 to species list
    if cls_rep_3d=='virtual_atom':
        print('virtual atom yes')
        for split, species in split_species.items():

            species_split = torch.cat([torch.tensor([-1]), split_species[split]])
            split_species[split] = species_split.sort()[0]
        
        # taking uniques to be sure
        for split, species in split_species.items():
            split_species[split] = split_species[split].unique(sorted=True)

        all_species = torch.cat([torch.tensor([-1]), all_species])
        all_species = all_species.sort()[0]


    print('split_species: ', split_species)
    print('all_species: ', all_species)
    # Now check that each split has at least one example of every atomic spcies from the entire dataset.
    if not all([split.tolist() == all_species.tolist() for split in split_species.values()]):
        print('split species check: ', [split.tolist() == all_species.tolist() for split in split_species.values()])
        # Allows one to override this check if they really want to. Not recommended as the answers become non-sensical.
        if ignore_check:
            logging.error(
                'The number of species is not the same in all datasets!')
        else:
            raise ValueError(
                'Not all datasets have the same number of species!')

    # Finally, return a list of all species
    return all_species
