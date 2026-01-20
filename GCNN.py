import torch
import numpy as np
import pandas as pd
from pathlib import Path
from Bio.PDB import PDBParser
from scipy.sparse import csr_matrix, save_npz, load_npz
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn as nn
import torch.nn.functional as F

AA_MAPPING = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19
}

def parse_go_annotations(gene_list, gaf_df):
    delete_indices = []
    valid_genes = []

    for g, gene in enumerate(gene_list):
        if pd.isna(gene) or gene == "nan":
            delete_indices.append(g)
        else:
            go_terms = gaf_df[gaf_df['Gene'] == gene]['GO']
            if not go_terms.empty:
                valid_genes.append(gene)

    return valid_genes, delete_indices

def map_go_terms(gene, gaf_df, go_to_index):
    go_vector = np.zeros(len(go_to_index))
    go_terms = gaf_df[gaf_df['Gene'] == gene]['GO'].tolist()
    for go_term in go_terms:
        if go_term in go_to_index:
            go_vector[go_to_index[go_term]] = 1
    return go_vector

def load_structure(pdb_file):
    parser = PDBParser()
    return parser.get_structure('protein', pdb_file)

def calc_distance_matrix(structure):
    model = structure[0]
    residues = [residue for residue in model.get_residues() if 'CA' in residue]
    residue_names = [residue.resname for residue in residues]
    num_residues = len(residues)
    distance_matrix = np.zeros((num_residues, num_residues))

    for i, res_i in enumerate(residues):
        for j, res_j in enumerate(residues):
            distance_matrix[i, j] = res_i['CA'] - res_j['CA']

    return distance_matrix, residue_names

def create_adjacency_matrix(distance_matrix, threshold=10.0):
    return csr_matrix((distance_matrix < threshold).astype(int))

class ProteinGCN(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=128, output_dim=30954):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = global_mean_pool(x, batch)
        return self.classifier(x)

def main():
    # Load data
    go_data_path = '/content/drive/MyDrive/ProteinClassificationData/GO_data'
    gene_cond_df = pd.read_csv(f'{go_data_path}/gene_condition_source_id.txt', sep='\t')
    gaf_df = pd.read_table(f'{go_data_path}/goa_human.gaf', comment='!', header=None,
                          usecols=[2,9], names=["Gene", "GO"])

    # Process genes
    subsample = gene_cond_df.sample(n=10)
    column_values = subsample['AssociatedGenes'].tolist()
    valid_genes, delete_idx = parse_go_annotations(column_values, gaf_df)

    # Create GO term mapping
    unique_go = gaf_df['GO'].unique()
    go_to_index = {go: i for i, go in enumerate(unique_go)}
    gene_go_vectors = [map_go_terms(gene, gaf_df, go_to_index) for gene in valid_genes]

    # Process PDB files
    pdb_files = ['/content/drive/MyDrive/ProteinClassificationData/PBD_data/2c0k.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/8d3u.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/5xix.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/7dmp.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/2da6.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/5cqr.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/7r63.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/5edp.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/4pr3.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/3qbp.pdb']


    adjacency_matrices = []
    for idx, pdb_file in enumerate(pdb_files):
        if idx in delete_idx:
            continue

        try:
            structure = load_structure(pdb_file)
            dist_matrix, _ = calc_distance_matrix(structure)
            adj_matrix = create_adjacency_matrix(dist_matrix)
            adjacency_matrices.append(adj_matrix)
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            continue

    # Create PyG Dataset
    dataset = []
    for idx, (adj_matrix, go_vector) in enumerate(zip(adjacency_matrices, gene_go_vectors)):
        try:
            # Get node features
            structure = load_structure(pdb_files[idx])
            _, residue_names = calc_distance_matrix(structure)
            indices = torch.tensor([AA_MAPPING.get(name.upper(), -1) for name in residue_names])
            valid_indices = indices[indices != -1]
            x = F.one_hot(valid_indices, num_classes=20).float()

            # Process edges
            edge_index, edge_attr = torch_geometric.utils.from_scipy_sparse_matrix(adj_matrix)

            # Validate edges
            num_nodes = x.size(0)
            mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]

            if edge_index.size(1) == 0:
                edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                edge_attr = torch.tensor([1.0], dtype=torch.float)

            # Create Data object
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(go_vector, dtype=torch.float)
            )
            dataset.append(data)
        except Exception as e:
            print(f"Skipping sample {idx}: {str(e)}")
            continue

    # Train model
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model = ProteinGCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(50):
        model.train()
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)

            # Reshape for loss calculation
            out_flat = out.view(-1)
            target_flat = batch.y.view(-1).float()

            loss = criterion(out_flat, target_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
