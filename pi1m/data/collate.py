import torch
from transformers import RobertaConfig, RobertaTokenizerFast, RobertaForMaskedLM, DataCollatorForLanguageModeling # for ChemBERTa

charge_dict_full = {'*': -2, 'CLS': -1, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Si': 14, 'P':15, 'S': 16, 'Cl': 17, 'K': 19, 'Ca': 20, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Zn': 30, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Cd': 48, 'Sn': 50, 'Te': 52, 'I': 53, 'Pb': 82} # added 2-5 & 10
charge_dict = {'*': -2, 'CLS': -1, 'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Si': 14, 'P':15, 'S': 16, 'Cl': 17, 'Fe': 26, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Sn': 50, 'I': 53} # added 2-5 & 10

def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        # print('props dropzeros: ', props, to_keep)
        # print('subselected: ', props[:, to_keep, ...])
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True, tokenizer=None, cls_rep_3d='virtual_atom'):
        self.load_charges = load_charges
        self.mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True) # , mlm_probability=0.15
        self.cls_rep_3d = cls_rep_3d

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn_2(self, batch):
        """
        Second, wrapping data collator function for 1D and 3D data, collectivel.
        Sends batch containing both 1D (SMILES) and 3D (QM9) data to model's forward;
        """
        # q: what does our MLM data collator function need as input ..?



    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        # print('collate batch: ', batch)
        # smiles_input = {'smiles': batch['smiles']}
        # smiles_list = [molecule['smiles']for molecule in batch]

        # shape of batch = [{1d: {mol1_smiles}, 3d: {mol1_3d}}, {1d: {mol2_smiles}, 3d: {mol2_3d}}, {...}, ...]
        batch_1d = [molecule['1d'] for molecule in batch]
        batch_3d = [molecule['3d'] for molecule in batch]
        
        # print('collate batch 3d: ', batch_3d)
        # print('collate batch 1d: ', batch_1d)


        # note: we do not stack 1d inputs, as collator for MLM expects list of molecule smile dicts as input
        # batch_1d = {prop: torch.stack([molecule[prop] for molecule in batch_1d]) for prop in batch_1d[0].keys()}
        
        ## 1d
        # collate smiles input, which pads smiles batch and masks tokens randomly for Masked L.M.
        batch_1d = self.mlm_collator(batch_1d)
        # print('collated batch 1d: ', batch_1d)
        
        # print('batch: ', batch)
        # remove non-tensor type input (the smiles field of the batch dict.)

        # for molecule in batch:
        #     del molecule['smiles']

        # input_1d = 
        ##3d
        batch_3d = {prop: batch_stack([mol[prop] for mol in batch_3d]) for prop in batch_3d[0].keys()}
        # print('batch 3d stacked: ', batch_3d)

        to_keep = (batch_3d['charges'].sum(0) > 0) # sum up over batch dim, so this gives (True, True, ..., False, .. False), where False is the first atom without a charge in any of the batch's molecules
        # print('to_keep: ', to_keep)
        
        nr_atoms=batch_3d['charges'].shape[-1]
        batch_3d['positions']=batch_3d['positions'][:,:nr_atoms,:]
        # NOTE: intervene and palce x0 as charegdict['CLS'] and as the mean coord in posiiton 0 of the coordinates;
        print('to keep: ', to_keep)
        print('batch_3d: ', batch_3d)
        print('batch_3d shapes: ', [batch_3d.keys()],[el.shape for el in batch_3d.values()])
        batch_3d = {key: drop_zeros(prop, to_keep) for key, prop in batch_3d.items()}
        # print('batch_3d zeros dropped: ', batch_3d)

        atom_mask = batch_3d['charges'] != 0 # changed > into != for incl. of cls and *
        # print('atom_mask: ', atom_mask)
        batch_3d['atom_mask'] = atom_mask

        #Obtain edges mask of every possible edge (i,j)
        batch_size, n_nodes = atom_mask.size()
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

        #mask diagonal
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask *= diag_mask

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch_3d['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

        if self.load_charges:
            batch_3d['charges'] = batch_3d['charges'].unsqueeze(2)
        else:
            batch_3d['charges'] = torch.zeros(0)
        # print('batch 3d charges: ', batch_3d['charges'])

        # print('shape: ', batch['charges'].shape)
        # batch['smiles'] = smiles_list
        # print('inside collate_fn batch: ', batch)

        return {'1d': batch_1d, '3d': batch_3d}
