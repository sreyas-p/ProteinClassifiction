import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from scipy.sparse import csr_matrix
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from transformers import AutoTokenizer

AA_MAPPING = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert")

def process_protein(pdb_path, label_vector):
    try:
        parser = PDBParser()
        structure = parser.get_structure('protein', pdb_path)
        model = structure[0]

        residues = [res for res in model.get_residues() if 'CA' in res]
        seq = [AA_MAPPING.get(res.resname, "X") for res in residues]

        inputs = tokenizer(" ".join(seq),
                         add_special_tokens=False,
                         return_tensors="pt")
        x = inputs["input_ids"].squeeze()

        num_res = len(residues)
        dist_matrix = np.zeros((num_res, num_res))
        for i, res_i in enumerate(residues):
            for j, res_j in enumerate(residues):
                dist_matrix[i,j] = res_i['CA'] - res_j['CA']

        adj_matrix = (dist_matrix < 10.0).astype(int)
        edge_index = torch.tensor(np.stack(adj_matrix.nonzero()), dtype=torch.long)

        mask = (edge_index[0] < num_res) & (edge_index[1] < num_res)
        edge_index = edge_index[:, mask]

        if edge_index.size(1) == 0:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            y=torch.tensor(label_vector, dtype=torch.float).view(1, -1)  # 2D shape
        )

    except Exception as e:
        print(f"Error processing {pdb_path}: {str(e)}")
        return None

class ProtBertGCN(nn.Module):
    def __init__(self, vocab_size=tokenizer.vocab_size, output_dim=30954):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.conv1 = GCNConv(128, 256)
        self.conv2 = GCNConv(256, 256)
        self.classifier = nn.Linear(256, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x).squeeze(1)
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.classifier(x)

def main():
    go_data_path = '/content/drive/MyDrive/ProteinClassificationData/GO_data'
    gene_cond_df = pd.read_csv(f'{go_data_path}/gene_condition_source_id.txt', sep='\t')
    gaf_df = pd.read_table(f'{go_data_path}/goa_human.gaf', comment='!', header=None,
                          usecols=[2,9], names=["Gene", "GO"])

    subsample = gene_cond_df.sample(n=10)
    column_values = subsample['AssociatedGenes'].tolist()
    valid_genes = [g for g in column_values if not pd.isna(g) and g in gaf_df['Gene'].values]

    unique_go = gaf_df['GO'].unique()
    go_to_index = {go: i for i, go in enumerate(unique_go)}
    gene_go_vectors = []
    for gene in valid_genes:
        vec = np.zeros(len(unique_go))
        for term in gaf_df[gaf_df['Gene'] == gene]['GO']:
            if term in go_to_index:
                vec[go_to_index[term]] = 1
        gene_go_vectors.append(vec)

    pdb_files = ['/content/drive/MyDrive/ProteinClassificationData/PBD_data/2c0k.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/8d3u.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/5xix.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/7dmp.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/2da6.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/5cqr.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/7r63.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/5edp.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/4pr3.pdb', '/content/drive/MyDrive/ProteinClassificationData/PBD_data/3qbp.pdb'][:len(valid_genes)]

    dataset = []
    for pdb_file, go_vector in zip(pdb_files, gene_go_vectors):
        data = process_protein(pdb_file, go_vector)
        if data is not None:
            dataset.append(data)

    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = ProtBertGCN(output_dim=len(unique_go))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(10):
        model.train()
        total_loss = 0

        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            target = batch.y.view(out.shape)  # Match dimensions
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:02d} | Loss: {total_loss/len(loader):.4f}")

if __name__ == "__main__":
    main()
