import logging
import pickle
import os
import lmdb
import torch
import tarfile
from torch.nn.utils.rnn import pad_sequence
import numpy as np

charge_dict_full = {'*': -2, 'CLS': -1, 'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Si': 14, 'P':15, 'S': 16, 'Cl': 17, 'K': 19, 'Ca': 20, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Zn': 30, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Cd': 48, 'Sn': 50, 'Te': 52, 'I': 53, 'Pb': 82} # added 2-5 & 10
charge_dict = {'*': -2, 'CLS': -1, 'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'Si': 14, 'P':15, 'S': 16, 'Cl': 17, 'Fe': 26, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Sn': 50, 'I': 53} # added 2-5 & 10

def split_dataset(data, split_idxs):
    """
    Splits a dataset according to the indices given.

    Parameters
    ----------
    data : dict
        Dictionary to split.
    split_idxs :  dict
        Dictionary defining the split.  Keys are the name of the split, and
        values are the keys for the items in data that go into the split.

    Returns
    -------
    split_dataset : dict
        The split dataset.
    """
    split_data = {}
    for set, split in split_idxs.items():
        split_data[set] = {key: val[split] for key, val in data.items()}

    return split_data

# def save_database()


def process_xyz_files(data, process_file_fn, file_ext=None, file_idx_list=None, stack=True):
    """
    Take a set of datafiles and apply a predefined data processing script to each
    one. Data can be stored in a directory, tarfile, or zipfile. An optional
    file extension can be added.

    Parameters
    ----------
    data : str
        Complete path to datafiles. Files must be in a directory, tarball, or zip archive.
        --> note: or, right now, a list containing complete paths to three datafiles, namely train, valid and test.lmdb files;

    process_file_fn : callable
        Function to process files. Can be defined externally.
        Must input a file, and output a dictionary of properties, each of which
        is a torch.tensor. Dictionary must contain at least three properties:
        {'num_elements', 'charges', 'positions'}
    file_ext : str, optional
        Optionally add a file extension if multiple types of files exist.
    file_idx_list : ?????, optional
        Optionally add a file filter to check a file index is in a
        predefined list, for example, when constructing a train/valid/test split.
    stack : bool, optional
        ?????
    """
    print('process_xyz_files called')
    print('data: ', data)
    logging.info('Processing data file: {}'.format(data))
    # print('data: ', data)
    print('tar datafile: ', data)
    if tarfile.is_tarfile(data):

        tardata = tarfile.open(data, 'r')
        files = tardata.getmembers()

        readfile = lambda data_pt: tardata.extractfile(data_pt)


        # note: -v moved rest of this if statement from below the if/else for extending to lmdb files
        # Use only files that end with specified extension.
        if file_ext is not None:
            files = [file for file in files if file.endswith(file_ext)]

        # Use only files that match desired filter.
        if file_idx_list is not None:
            files = [file for idx, file in enumerate(files) if idx in file_idx_list]

        # Now loop over files using readfile function defined above
        # Process each file accordingly using process_file_fn

        molecules = []

        for file in files:
            with readfile(file) as openfile:
                molecules.append(process_file_fn(openfile))

    elif os.path.isdir(data):
        files = os.listdir(data)
        files = [os.path.join(data, file) for file in files]

        readfile = lambda data_pt: open(data_pt, 'r')


        # note: -v moved rest of this if statement from below the if/else for extending to lmdb files
        # Use only files that end with specified extension.
        if file_ext is not None:
            files = [file for file in files if file.endswith(file_ext)]

        # Use only files that match desired filter.
        if file_idx_list is not None:
            files = [file for idx, file in enumerate(files) if idx in file_idx_list]

        # Now loop over files using readfile function defined above
        # Process each file accordingly using process_file_fn

        molecules = []

        for file in files:
            with readfile(file) as openfile:
                molecules.append(process_file_fn(openfile))
    
    # check if file is .lmdb file
    elif data.endswith('.lmdb'):
        
        # note: modify this
        # train_file, valid_file, test_file = data[0],data[1],data[2] # unpacking for readability
        # lmdb_split_files = {} = {'train':train_file, 'valid':valid_file, 'test':test_file}
        # for split, file in lmdb_split_files.items():
        # replacing the file path with the actual file/data list in the dictionary we initialize above
        # lmdb_split_files[split] = data_list
        # append all dicts to one big molecules list for the check below the if/else statement
        # molecules=[]
        # for split_list in lmdb_split_files.values():
        #     molecules+=split_list

        split_file = data
        env = lmdb.open(
            split_file,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )

        data_list=[] # data_list will have the structure [{mol_1_num_atoms:list(..), ..., attention_mask:tensor(..)}, {mol_2..}, ...], i.e. a list of dicts (with the smiles part pre-tokenized, i.e. they are already of the type tensor.)
        with env.begin() as iter_env:
            with iter_env.cursor() as curs:
                for key, value in curs:
                    datapoint = pickle.loads(value) # this is a dict like {num_atoms:list(..), star_indices:[..], ..., attention_mask:tensor(..)}

                    # filter out atoms that do not appear in charge dict, which means they are atoms that occur in <=250 molecules; this is to keep embedding dimensionality low(er)
                    infrequent_atom_check=False
                    for atom in datapoint['atoms']:
                        if atom not in charge_dict.keys():
                            infrequent_atom_check=True

                    if not infrequent_atom_check:
                        data_list.append(process_file_fn(datapoint))

        # our list of dicts to use in this function
        molecules = data_list

        # add dataset index similar to qm9's
        for idx, molecule in enumerate(molecules):
            molecule['index'] = torch.tensor(idx+1)

    else:
        raise ValueError('Can only read from directory or tarball archive!')


    # molecules 1/here: [{num_atoms_molecule_1: ..., ..}, {num_atoms_molecule_2: ..., ...}, ...]
    # print('molecules 1: ', molecules)

    # Check that all molecules have the same set of items in their dictionary:
    props = molecules[0].keys()
    assert all(props == mol.keys() for mol in molecules), 'All molecules must have same set of properties/keys!'

    # SMILES EDIT
    # print('molecule:', molecules[0])
    # Convert list-of-dicts to dict-of-lists
    molecules = {prop: [mol[prop] for mol in molecules] for prop in props} # into format {'property_name': [property_mol_1, property_mol_2, ...], '...': ...}

    # molecules 2/here: {num_atoms: [num_atoms_mol_1, num_atoms_mol_2, ...], positions: [...], ..., smiles_relaxed: [...]}
    # print('molecules 2: ', molecules)

    # print('all molecules: ', molecules)

    molecules_smiles = molecules['smiles'] # : molecule['smiles'] for molecule in molecules}, 'smiles_relaxed': molecules[0]['smiles_relaxed']}]
    del molecules['smiles']
    if 'smiles_relaxed' in molecules.keys():
        del molecules['smiles_relaxed']

    # print('molecules before stacking: (3: ) ', [(key,prop[0]) for (key, prop) in molecules.items()])
    # If stacking is desireable, pad and then stack.
    if stack: # fix here
        molecules = {key: pad_sequence(val, batch_first=True) if val[0].dim() > 0 else torch.stack(val) for key, val in molecules.items()}
    
    # print('molecules after stacking: (4: ) ', [(key,prop[0]) for (key, prop) in molecules.items()])

    molecules['smiles'] = molecules_smiles

    # molecules['smiles_relaxed'] = molecules_smiles['smiles_relaxed']
    print('molecules smiles 0-3: (5: ) ', molecules['smiles'][0:3])

    # molecules  5/here: {num_atoms: tensor([num_atoms_mol_1, ...]), positions: tensor([...]), ..., smiles: [smiles_mol_1, smiles_mol_2, ...]}, but then with zero-padding on charges and positions entries!
    # print('molecules: (5: ) ', molecules)

    return molecules

'''
for key_name in ['mol', 'origin_smi', 'star_pair', 'scaffold', 'input_ids', 'attention_mask']:
        # note: removing 'origin_smi' means that we are using the substituted smiles for now (atoms ipv asterisks '*')
        if key_name in datafile.keys():
            del datafile[key_name]

    assert 'atoms' in datafile.keys()
    
    for atom in datafile['atoms']:
        if atom not in charge_dict.keys():
            charge_dict[atom] = 50
            print(f'WARNING: made an exception for atom {atom}, set to charge = 50')

    datafile['charges'] = [charge_dict[atom] for atom in datafile['atoms']] # charges becomes e.g. [1, 1, 1, 6, 6, 6, 1, 1, 1]

    del datafile['atoms']

    datafile['num_atoms'] = len(datafile['charges'])

    # choose random rdkit conformer from list of conformers and set it as datafile['coordinates']
    assert 'coordinates' in datafile.keys()
    nr_conformers = len(datafile['coordinates'])
    found_conformer=False
    # sample a random conformer at most nr_conformers times, until we get one where the z-axis has non-zero positions (i.e. 3d and not 2d rdkit conformer)
    while not found_conformer:
        if nr_conformers>1:
            conformer_idx = np.random.randint(nr_conformers)
            conformer = datafile['coordinates'][conformer_idx] # np array of size (batch, seq_len, 3)
            # if the conformer has an empty z axis, remove it from list and sample again
            if not np.any(conformer[:,-1]):
                del datafile['coordinates'][conformer_idx]
                nr_conformers -= 1
            elif np.any(conformer[:,-1]):
                found_conformer=True
        elif nr_conformers <= 1:
            conformer = datafile['coordinates'][-1]
            found_conformer=True
    datafile['coordinates'] = conformer



    # normalization of every coordinate x,y,z by subtracting mean of the entire molecule's x, y and z coordinates.
    datafile['coordinates'] = datafile['coordinates'] - datafile['coordinates'].mean(axis=0)
    # rename coordinates to positions
    datafile['positions'] = datafile['coordinates']
    del datafile['coordinates']
'''



def process_mm_lmdb_files(datafile):
    """
    Read existing lmdb file from the MMPolymer repo, and turn it into a dict that our process_xyz_files can exactly
    take and feed to our e3/polyMRL module; the output molecular dict contains number of atoms, charges, coordinates, 
    index, and target (if applicable i.e. prop. pred.) coming out of the e.g. Eat property prediction dataset (or another one).

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.
        -> in our case, a dict containing the raw data as it is present in the mmpolymer lmdb files.

    Returns
    -------
    molecule : dict
        Dictionary containing the correct molecular properties of the associated dict/file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    ## removed unnecessary columns from molecules
    ## added normalization -> also for target or not ..?
    ## molecule's smiles is added as 'smiles'
    ## .. also make sure to add the column for the target property -if present-
    
    # delete all unnecessary columns from datafile
    # from here
    for key_name in ['mol', 'origin_smi', 'star_pair', 'scaffold', 'input_ids', 'attention_mask']:
        if key_name in datafile.keys():
            del datafile[key_name]

    assert 'atoms' in datafile.keys()

    # assign charges to atoms
    for atom in datafile['atoms']:
        if atom not in charge_dict.keys():
            charge_dict[atom] = 50
            print(f'WARNING: made an exception for atom {atom}, set to charge = 50')

    datafile['charges'] = [charge_dict[atom] for atom in datafile['atoms']]
    del datafile['atoms']
    datafile['num_atoms'] = len(datafile['charges'])

    # choose random rdkit conformer from list of conformers and set it as datafile['coordinates']
    assert 'coordinates' in datafile.keys()
    nr_conformers = len(datafile['coordinates'])
    found_conformer = False

    while not found_conformer:
        if nr_conformers > 1:
            conformer_idx = np.random.randint(nr_conformers)
            conformer = datafile['coordinates'][conformer_idx]  # np array of size (num_atoms, 3)
            if not np.any(conformer[:, -1]):  # remove if 2D (empty z-axis)
                del datafile['coordinates'][conformer_idx]
                nr_conformers -= 1
            else:
                found_conformer = True
        elif nr_conformers <= 1:
            conformer = datafile['coordinates'][-1]
            found_conformer = True

    # Remove hydrogen coordinates
    atom_symbols = datafile.get('atom_symbols', None)
    if atom_symbols is None and 'mol' in datafile:
        # fallback: try to get atom symbols from RDKit mol
        atom_symbols = [atom.GetSymbol() for atom in datafile['mol'].GetAtoms()]

    if atom_symbols is not None:
        heavy_atom_indices = [i for i, a in enumerate(atom_symbols) if a != 'H']
        conformer = conformer[heavy_atom_indices]

    datafile['coordinates'] = conformer

    # normalize coordinates
    datafile['coordinates'] = datafile['coordinates'] - datafile['coordinates'].mean(axis=0)
    # rename coordinates to positions
    datafile['positions'] = datafile['coordinates']
    del datafile['coordinates']
    # until here

    # turn target float into list so we can convert it to a tensor of shape (1) after this function
    if 'target' in datafile.keys():
        datafile['target'] = datafile['target']

    # rename smi to smiles
    smiles = datafile['smi']
    del datafile['smi']
    # check that no smiles are left in molecule dict
    assert all([(not isinstance(el, str) for el in datafile.values())])

    # rename and turn molecule values into tensors
    molecule = datafile
    print('molecule', molecule)
    if isinstance(molecule['target'],str):
        molecule['target'] = float(molecule['target'])

    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    molecule['smiles'] = smiles    

    return molecule


    # after this func.

    # {num_atoms:list(..), star_indices:[..], ..., attention_mask:tensor(..)}
    
    # print('datafile: ', datafile.readlines())
    # xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]
    
    # for el in range(len(xyz_lines)):
        # print(el, ': ', xyz_lines[el])

        # print('xyz_lines: ', xyz_lines)
    
    # num_atoms = int(xyz_lines[0])
    # mol_props = xyz_lines[1].split()
    # mol_xyz = xyz_lines[2:num_atoms+2]
    # mol_freq = xyz_lines[num_atoms+2]
    # mol_smiles, mol_smiles_relaxed = xyz_lines[num_atoms+3].split()

    # atom_charges, atom_positions = [], []
    # for line in mol_xyz:
    #     atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
    #     atom_charges.append(charge_dict[atom])
    #     atom_positions.append([float(posx), float(posy), float(posz)])

    # note: add index
    # prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    # prop_strings = prop_strings[1:]
    # mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    # mol_props = dict(zip(prop_strings, mol_props))
    # mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    # molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    # molecule.update(mol_props)
    # turn into tensors
    # molecule = {key: torch.tensor(val) for key, val in molecule.items()}
    # molecule['smiles'] = mol_smiles
    # molecule['smiles_relaxed'] = mol_smiles_relaxed

    # return molecule

def process_xyz_md17(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the MD-17 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.
    """
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]

    line_counter = 0
    atom_positions = []
    atom_types = []
    for line in xyz_lines:
        if line[0] == '#':
            continue
        if line_counter == 0:
            num_atoms = int(line)
        elif line_counter == 1:
            split = line.split(';')
            assert (len(split) == 1 or len(split) == 2), 'Improperly formatted energy/force line.'
            if (len(split) == 1):
                e = split[0]
                f = None
            elif (len(split) == 2):
                e, f = split
                f = f.split('],[')
                atom_energy = float(e)
                atom_forces = [[float(x.strip('[]\n')) for x in force.split(',')] for force in f]
        else:
            split = line.split()
            if len(split) == 4:
                type, x, y, z = split
                atom_types.append(split[0])
                atom_positions.append([float(x) for x in split[1:]])
            else:
                logging.debug(line)
        line_counter += 1

    atom_charges = [charge_dict[type] for type in atom_types]

    molecule = {'num_atoms': num_atoms, 'energy': atom_energy, 'charges': atom_charges,
                'forces': atom_forces, 'positions': atom_positions}

    molecule = {key: torch.tensor(val) for key, val in molecule.items()}

    return molecule

def process_xyz_gdb9(datafile):
    """
    Read xyz file and return a molecular dict with number of atoms, energy, forces, coordinates and atom-type for the gdb9 dataset.

    Parameters
    ----------
    datafile : python file object
        File object containing the molecular data in the MD17 dataset.

    Returns
    -------
    molecule : dict
        Dictionary containing the molecular properties of the associated file object.

    Notes
    -----
    TODO : Replace breakpoint with a more informative failure?
    """
    # print('datafile: ', datafile.readlines())
    xyz_lines = [line.decode('UTF-8') for line in datafile.readlines()]
    
    # for el in range(len(xyz_lines)):
        # print(el, ': ', xyz_lines[el])

        # print('xyz_lines: ', xyz_lines)
    
    num_atoms = int(xyz_lines[0])
    mol_props = xyz_lines[1].split()
    mol_xyz = xyz_lines[2:num_atoms+2]
    mol_freq = xyz_lines[num_atoms+2]
    mol_smiles, mol_smiles_relaxed = xyz_lines[num_atoms+3].split()

    atom_charges, atom_positions = [], []
    for line in mol_xyz:
        atom, posx, posy, posz, _ = line.replace('*^', 'e').split()
        atom_charges.append(charge_dict[atom])
        atom_positions.append([float(posx), float(posy), float(posz)])

    prop_strings = ['tag', 'index', 'A', 'B', 'C', 'mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv']
    prop_strings = prop_strings[1:]
    mol_props = [int(mol_props[1])] + [float(x) for x in mol_props[2:]]
    mol_props = dict(zip(prop_strings, mol_props))
    mol_props['omega1'] = max(float(omega) for omega in mol_freq.split())

    molecule = {'num_atoms': num_atoms, 'charges': atom_charges, 'positions': atom_positions}
    molecule.update(mol_props)
    molecule = {key: torch.tensor(val) for key, val in molecule.items()}
    molecule['smiles'] = mol_smiles
    molecule['smiles_relaxed'] = mol_smiles_relaxed

    return molecule
