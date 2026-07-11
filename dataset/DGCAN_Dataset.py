# -*- coding: utf-8 -*-  # Specify UTF-8 file encoding

from collections import defaultdict  # Import defaultdict from collections to create dictionaries
import numpy as np  # Import the NumPy library
from rdkit import Chem  # Import Chem from RDKit for cheminformatics
import torch  # Import the PyTorch library

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set the device to GPU if available

# Create default dictionaries for atoms, bonds, fingerprints, and edges
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
radius = 1  # Set the fingerprint extraction radius


def create_atoms(mol, atom_dict):
    """Convert atom types in a molecule, such as H, C, and O, to indices, such as H=0, C=1, and O=2.
    Each atom index accounts for aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]  # Get each atom symbol
    for a in mol.GetAromaticAtoms():  # Iterate over aromatic atoms
        i = a.GetIdx()  # Get the atom index
        atoms[i] = (atoms[i], 'aromatic')  # Mark aromatic atoms as aromatic
    atoms = [atom_dict[a] for a in atoms]  # Convert atom symbols to indices
    return np.array(atoms)  # Return a NumPy array of atom indices


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary whose keys are node IDs and whose values are tuples of neighbor nodes and chemical bond IDs, such as single and double bonds.
    """
    i_jbond_dict = defaultdict(lambda: [])  # Create a defaultdict to store key-value pairs
    for b in mol.GetBonds():  # Iterate over each chemical bond in the molecule
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()  # Get the start and end atom indices
        bond = bond_dict[str(b.GetBondType())]  # Get the chemical bond type index
        i_jbond_dict[i].append((j, bond))  # Add the neighbor node and bond type to the dictionary
        i_jbond_dict[j].append((i, bond))  # Add the reverse direction
    return i_jbond_dict  # Return the dictionary of atoms, neighbors, and bonds


def extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict):
    """Extract fingerprints from the molecular graph using the Weisfeiler-Lehman algorithm.
    """
    if (len(atoms) == 1) or (radius == 0):  # If there is only one atom or the radius is 0
        nodes = [fingerprint_dict[a] for a in atoms]  # Get fingerprints directly
    else:
        nodes = atoms  # Initialize nodes as atom indices
        i_jedge_dict = i_jbond_dict  # Initialize the edge dictionary

        for _ in range(radius):  # Iteratively update each node fingerprint
            nodes_ = []  # Initialize the new node list
            for i, j_edge in i_jedge_dict.items():  # Iterate over each node and its edges
                neighbors = [(nodes[j], edge) for j, edge in j_edge]  # Get neighbor nodes and edge information
                fingerprint = (nodes[i], tuple(sorted(neighbors)))  # Create the new fingerprint
                nodes_.append(fingerprint_dict[fingerprint])  # Update the fingerprint

            # Update each edge ID based on the nodes at both ends
            i_jedge_dict_ = defaultdict(lambda: [])  # Initialize the new edge dictionary
            for i, j_edge in i_jedge_dict.items():  # Iterate over the old edge dictionary
                for j, edge in j_edge:  # Iterate over each edge
                    both_side = tuple(sorted((nodes[i], nodes[j])))  # Create a sorted node tuple
                    edge = edge_dict[(both_side, edge)]  # Get the edge index
                    i_jedge_dict_[i].append((j, edge))  # Update the edge dictionary

            nodes = nodes_  # Update nodes
            i_jedge_dict = i_jedge_dict_  # Update the edge dictionary

    return np.array(nodes)  # Return the updated fingerprint array


def split_dataset(dataset, ratio):
    """Shuffle and split the dataset."""
    np.random.seed(1234)  # Fix the random seed for reproducibility
    # np.random.shuffle(dataset)  # Optional: shuffle the dataset
    n = int(ratio * len(dataset))  # Calculate the split point from the ratio
    return dataset[:n], dataset[n:]  # Return the split dataset



def create_testdataset(data_original,device):
    # with open(filepath, 'r') as f:  # Open the file
    #     # smiles_property = f.readline().strip().split()  # Optional: read the first line
    #     data_original = f.read().strip().split()  # Read all data
    data_original = [data for data in data_original if '.' not in data.split()[0]]  # Filter out data containing '.'
    dataset = []  # Initialize the dataset list
    for data in data_original:  # Iterate over each data item
        smiles = data  # Read the SMILES string
        try:
            """Create each data item using the functions defined above."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Generate a molecule from SMILES and add hydrogens
            atoms = create_atoms(mol, atom_dict)  # Create atom indices
            molecular_size = len(atoms)  # Get the molecular size
            i_jbond_dict = create_ijbonddict(mol, bond_dict)  # Create the atom-bond dictionary
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # Extract fingerprints
            adjacency = Chem.GetAdjacencyMatrix(mol)  # Get the adjacency matrix
            """Convert each item above from NumPy to PyTorch tensors on the device, either CPU or GPU."""
            fingerprints = torch.LongTensor(fingerprints).to(device)  # Convert fingerprints to tensors
            adjacency = torch.FloatTensor(adjacency).to(device)  # Convert the adjacency matrix to a tensor
            proper = torch.LongTensor([int(0)]).to(device)  # Create the label tensor
            dataset.append((smiles, fingerprints, adjacency, molecular_size, proper))  # Add the data item to the dataset
        except:
            print(smiles)  # Print the invalid SMILES string
    return dataset  # Return the created dataset

 
def create_dataset(filename, path, dataname,device):
    dir_dataset = path + dataname  # Build the dataset directory
    print(filename)  # Print the filename
    """Load the dataset."""
    try:
        with open(dir_dataset + filename, 'r') as f:  # Try to open the file
            smiles_property = f.readline().strip().split()  # Read the first line
            data_original = f.read().strip().split('\n')  # Read all data
    except:
        with open(dir_dataset + filename, 'r') as f:  # If an exception occurs, try opening the file again
            smiles_property = f.readline().strip().split()  # Read the first line
            data_original = f.read().strip().split('\n')  # Read all data

    # Exclude data containing '.'
    data_original = [data for data in data_original if '.' not in data.split()[0]]
    dataset = []  # Initialize the dataset list
    for data in data_original:  # Iterate over each data item
        # print(data)
        smiles, property = data.strip().split()  # Read SMILES and properties
        try:
            """Create each data item using the functions defined above."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Generate a molecule from SMILES and add hydrogens
            atoms = create_atoms(mol, atom_dict)  # Create atom indices
            molecular_size = len(atoms)  # Get the molecular size
            i_jbond_dict = create_ijbonddict(mol, bond_dict)  # Create the atom-bond dictionary
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # Extract fingerprints
            adjacency = Chem.GetAdjacencyMatrix(mol)  # Get the adjacency matrix
            """
            Convert each item above from NumPy to PyTorch tensors on the device, either CPU or GPU.
            """
            fingerprints = torch.LongTensor(fingerprints).to(device)  # Convert fingerprints to tensors
            adjacency = torch.FloatTensor(adjacency).to(device)  # Convert the adjacency matrix to a tensor
            property = torch.LongTensor([int(property)]).to(device)  # Create the label tensor
            dataset.append((smiles, fingerprints, adjacency, molecular_size, property))  # Add the data item to the dataset
        except:
            print(smiles)  # Print the invalid SMILES string
    return dataset  # Return the created dataset

def get_dcan_dataset(pos_smiles,neg_smiles):# Create a dataset specifically for the voting model
    # try:
    #     with open(dir_dataset + filename, 'r') as f:  # Try to open the file
    #         smiles_property = f.readline().strip().split()  # Read the first line
    #         data_original = f.read().strip().split('\n')  # Read all data
    # except:
    #     with open(dir_dataset + filename, 'r') as f:  # If an exception occurs, try opening the file again
    #         smiles_property = f.readline().strip().split()  # Read the first line
    #         data_original = f.read().strip().split('\n')  # Read all data

    # Exclude data containing '.'
    pos_data=[i+' 1' for i in pos_smiles]
    neg_data=[i+' 0' for i in neg_smiles]
    data_original=pos_data+neg_data
    data_original = [data for data in data_original if '.' not in data.split()[0]]
    dataset = []  # Initialize the dataset list
    for data in data_original:  # Iterate over each data item
        smiles, property = data.strip().split()  # Read SMILES and properties
        try:
            """Create each data item using the functions defined above."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Generate a molecule from SMILES and add hydrogens
            atoms = create_atoms(mol, atom_dict)  # Create atom indices
            molecular_size = len(atoms)  # Get the molecular size
            i_jbond_dict = create_ijbonddict(mol, bond_dict)  # Create the atom-bond dictionary
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # Extract fingerprints
            adjacency = Chem.GetAdjacencyMatrix(mol)  # Get the adjacency matrix
            """
            Convert each item above from NumPy to PyTorch tensors on the device, either CPU or GPU.
            """
            fingerprints = torch.LongTensor(fingerprints)  # Convert fingerprints to tensors
            adjacency = torch.FloatTensor(adjacency) # Convert the adjacency matrix to a tensor
            property = torch.LongTensor([int(property)])  # Create the label tensor
            dataset.append((smiles, fingerprints, adjacency, molecular_size, property))  # Add the data item to the dataset
        except:
            print("DCAN SMILES conversion failed")
            print(smiles)  # Print the invalid SMILES string
    return dataset  # Return the created dataset

def get_dacan_test_dataset(smiles1,device):
    if isinstance(smiles1, str):
        data_original=[smiles1]
        dataset = []  # Initialize the dataset list
        for data in data_original:  # Iterate over each data item
            smiles = data  # Read the SMILES string
            try:
                """Create each data item using the functions defined above."""
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))  # Generate a molecule from SMILES and add hydrogens
                atoms = create_atoms(mol, atom_dict)  # Create atom indices
                molecular_size = len(atoms)  # Get the molecular size
                i_jbond_dict = create_ijbonddict(mol, bond_dict)  # Create the atom-bond dictionary
                fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict, fingerprint_dict, edge_dict)  # Extract fingerprints
                adjacency = Chem.GetAdjacencyMatrix(mol)  # Get the adjacency matrix
                """Convert each item above from NumPy to PyTorch tensors on the device, either CPU or GPU."""
                fingerprints = torch.LongTensor(fingerprints).to(device)  # Convert fingerprints to tensors
                adjacency = torch.FloatTensor(adjacency).to(device)  # Convert the adjacency matrix to a tensor
                proper = torch.LongTensor([int(0)]).to(device)  # Create the label tensor
                dataset.append((smiles, fingerprints, adjacency, molecular_size, proper))  # Add the data item to the dataset
            except:
                print(smiles)  # Print the invalid SMILES string
    return dataset  # Return the created dataset

def to_gpu(dataset,device):
    new_data=[]
    for i in dataset:
        new_data.append((i[0],torch.tensor(np.array(i[1])).to(device),torch.tensor(np.array(i[2])).to(device),i[3],torch.tensor(np.array(i[4])).to(device)))
    return new_data