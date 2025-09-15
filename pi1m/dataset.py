from torch.utils.data import DataLoader
from pi1m.data.args import init_argparse
from pi1m.data.collate import PreprocessQM9
from pi1m.data.utils import initialize_datasets
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling # for ChemBERTa
from transformers import OpenAIGPTForSequenceClassification, DataCollatorWithPadding


def retrieve_dataloaders(cfg):
    '''
    Returns dataloaders dict, with train, valid and test loaders;

    input: all args passed to the main training script;
    -> note: this cfg are all the args passed on in the main script, main_pi1m.py
    output: dataloaders dictionary;
    '''

    if cfg.dataset in ["Egc", "Egb", "Eea", "Ei", "Xc", "EPS", "Nc", "Eat", "PI1M"]: # check if selected dataset is qm9
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers # the number of workers, given in args by user, indicates how many processes should in parallel work on loading data on cpu (if anything multi-process, i.e. >1, is desired)
        filter_n_atoms = cfg.filter_n_atoms # if we subset QM9 to only contain x number of atoms in the molecules (e.g. for very specific distribution learning/generation)

        # Initialize dataloader
        args = init_argparse('qm9')
        print('qm9 args: ', args)
        # data_dir = cfg.data_root_dir

        # retrieve dataset object --> this function gives us a dictionary of shape {train: dataset object, valid: ...}
        # note: -> when indexing e.g. datasets['train'], e.g. train_dataset[9], we get {'num_atoms': torch.tensor([5]), 'smiles': np.array(['CCCNC']), ...}, for example (I believe);
        args, datasets, num_species, charge_scale, tokenizer = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h, cls_rep_3d=cfg.cls_rep_3d,
                                                                        train_mode=cfg.train_mode)



        # convert the basic/constant unit of energy for molecular quantum properties in qm9 dataset to eV instead of hartree
        # eV = kinetic energy gained/lost by 1 electron accelerating from rest through an electric potential difference of one volt;
        # used often for describing energy levels in atoms/bonds/particle energy due to its small scale (1eV = 1.602176634 × 10^-19J).
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        # convert all the data in the train-, valid- and test-dataset objects to be in the eV unit
        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        # convert the datasets dictionary to have every set of train/valid/test only contain those molecule of size n
        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        # --> the ProcessedDataset object from datasets dict goes into the DataLoader object, 
        # and a dataloader object per split comes out in a dict of the shape: {train: DataLoader obj; valid: ...};
        # over these dataloader objects we can iterate, and the batches of data for the specified split are spit out
        preprocess = PreprocessQM9(load_charges=cfg.include_charges, tokenizer=tokenizer) # what does the preprocess function do exactly?

        # now, running get next iter(dataloader) should return a batch with the smiles string {..., 'smiles': 'CCNCCC'}, as well; then we can pass it to our model (using custom collator/dataset!)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}
        
    elif 'qm9' in cfg.dataset: # check if selected dataset is qm9
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers # the number of workers, given in args by user, indicates how many processes should in parallel work on loading data on cpu (if anything multi-process, i.e. >1, is desired)
        filter_n_atoms = cfg.filter_n_atoms # if we subset QM9 to only contain x number of atoms in the molecules (e.g. for very specific distribution learning/generation)

        # Initialize dataloader
        args = init_argparse('qm9')
        print('qm9 args: ', args)
        # data_dir = cfg.data_root_dir

        # retrieve dataset object --> this function gives us a dictionary of shape {train: dataset object, valid: ...}
        # note: -> when indexing e.g. datasets['train'], e.g. train_dataset[9], we get {'num_atoms': torch.tensor([5]), 'smiles': np.array(['CCCNC']), ...}, for example (I believe);
        args, datasets, num_species, charge_scale, tokenizer = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h, cls_rep_3d=cfg.cls_rep_3d,
                                                                        train_mode=cfg.train_mode)



        # convert the basic/constant unit of energy for molecular quantum properties in qm9 dataset to eV instead of hartree
        # eV = kinetic energy gained/lost by 1 electron accelerating from rest through an electric potential difference of one volt;
        # used often for describing energy levels in atoms/bonds/particle energy due to its small scale (1eV = 1.602176634 × 10^-19J).
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        # convert all the data in the train-, valid- and test-dataset objects to be in the eV unit
        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        # convert the datasets dictionary to have every set of train/valid/test only contain those molecule of size n
        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        # --> the ProcessedDataset object from datasets dict goes into the DataLoader object, 
        # and a dataloader object per split comes out in a dict of the shape: {train: DataLoader obj; valid: ...};
        # over these dataloader objects we can iterate, and the batches of data for the specified split are spit out
        preprocess = PreprocessQM9(load_charges=cfg.include_charges, tokenizer=tokenizer) # what does the preprocess function do exactly?
        
        # now, running get next iter(dataloader) should return a batch with the smiles string {..., 'smiles': 'CCNCCC'}, as well; then we can pass it to our model (using custom collator/dataset!)
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         shuffle=args.shuffle if (split == 'train') else False,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn)
                             for split, dataset in datasets.items()}

    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        data_file = './data/geom/geom_drugs_30.npy'
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform)
            shuffle = (key == 'train') and not cfg.sequential

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle)
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    # return either qm9 or geomdrugs dataloaders and the charge scale (the latter is ...?)
    return dataloaders, charge_scale, tokenizer


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets