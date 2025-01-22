import pathlib
import pandas as pd
import numpy as np
import torch
from gprofiler import GProfiler
#extract all coordinates with ATOM and CA

import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from scipy.sparse import csr_matrix, save_npz, load_npz
from torch_geometric.data import Data


def load_structure(pdb_file):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)   
    return structure

def calc_distance_matrix(structure):    
    model = structure[0]
    atoms = [residue['CA'] for residue in model.get_residues() if 'CA' in residue]
    num_atoms = len(atoms)
    distance_matrix = np.zeros((num_atoms, num_atoms))

    for i, atom_i in enumerate(atoms):
        for j, atom_j in enumerate(atoms):
            distance_matrix[i, j] = atom_i - atom_j

    return distance_matrix

#threshold of 10 Angstorms
def create_sparse_contact_map(distance_matrix, threshold=10.0):
    contact_map = distance_matrix < threshold
    sparse_contact_map = csr_matrix(contact_map)
    return sparse_contact_map

def create_adjacency_matrix(distance_matrix, threshold=10.0):
    adjacency_matrix = (distance_matrix < threshold).astype(int)
    return adjacency_matrix     

def save_sparse_matrix(filename, matrix):
    sparse_matrix = csr_matrix(matrix)
    save_npz(filename, sparse_matrix)

def load_sparse_matrix(filename):
    return load_npz(filename)

def plot_sparse_matrix(sparse_matrix):
    plt.imshow(sparse_matrix.toarray(), cmap='Greys', interpolation='none')
    plt.title('Protein Adjacency Matrix')
    plt.xlabel('Residue Index')
    plt.ylabel('Residue Index')
    plt.show()

def convert_to_pytorch_sparse(sparse_matrix):
    # Convert to PyTorch sparse COO tensor
    sparse_matrix_coo = sparse_matrix.tocoo()
    values = torch.tensor(sparse_matrix_coo.data, dtype=torch.float32)
    indices = torch.tensor([sparse_matrix_coo.row, sparse_matrix_coo.col], dtype=torch.int64)
    sparse_tensor = torch.sparse_coo_tensor(indices, values, sparse_matrix.shape)
    return sparse_tensor 

#all file paths 
pbdfiles = ['2c0k.pdb', '8d3u.pdb', '5xix.pdb', '7dmp.pdb', '2da6.pdb', '5cqr.pdb', '7r63.pdb', '5edp.pdb', '4pr3.pdb', '3qbp.pdb']
matricies = []
go_terms = ['oxivitygen carrier acty', 'P-type sodium:potassium-exchanging transporter activity']


for pdbfile in pbdfiles:
    structure = load_structure(pdbfile)
    distance_matrix = calc_distance_matrix(structure)
    adjacency_matrix = create_adjacency_matrix(distance_matrix)
    
    # Save adjacency matrix
    save_sparse_matrix('adjacency_matrix.npz', adjacency_matrix)

    # Load and plot adjacency matrix
    loaded_adjacency_matrix = load_sparse_matrix('adjacency_matrix.npz')
    #matricies += loaded_adjacency_matrix
    #plot_sparse_matrix(loaded_adjacency_matrix)
    sparse_tensor = convert_to_pytorch_sparse(loaded_adjacency_matrix)
    matricies += sparse_tensor
    
#amino acids encoded array
x = torch.eye(20)[[0, 1, 2, 3]] 

#adjacency matrix
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

# Example one-hot encoded GO term vector (y[0] = 1 is a go term)
output_dim = 128  
y = torch.zeros(output_dim)

# Create the graph object
data = Data(x=x, edge_index=edge_index, y=y)

class ProteinGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProteinGNN, self).__init__()
        self.conv1 =    (input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 2 Graph Convolutional layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global mean pooling (summarize graph features)
        x = torch.mean(x, dim=0)

        # MLP for classification
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x
    
# test parameters
num_amino_acids = 10
input_dim = num_amino_acids
hidden_dim = 64
output_dim = 128  # Number of GO terms

model = ProteinGNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

output = model(data)
print("Model Output:", output)

# Convert output to binary vector
pred = torch.argmax(output, dim=0)
print("Predicted Labels:", pred)

# Define the loss function
criterion = torch.nn.BCELoss()

# Example target (y)
target = data.y

# Compute the loss
loss = criterion(output, target)
print('Loss:', loss.item())