import pathlib
import pandas as pd
import numpy as np
import torch
from gprofiler import GProfiler
import requests
from goatools.obo_parser import GODag
from goatools.associations import read_gaf
import os

# Load the GO DAG (Directed Acyclic Graph)
obo_dag = GODag("go.obo")

# Read the gene association file (GAF)
gene_associations = read_gaf("goa_human.gaf")


column_name = 'AssociatedGenes'  # Replace with the name of the column you want to extract
df = pd.read_csv("gene_condition_source_id.txt", sep="	")
gaf_df = pd.read_table("goa_human.gaf", comment='!', header=None, sep="\t", usecols=[2,9], names=["Gene", "GO"])
print(gaf_df)
#subsample = df.sample(frac=0.005)
subsample = df.sample(n=50)

if column_name in subsample.columns:
    column_values_list = subsample[column_name].tolist()
    print(f"Values in column '{column_name}' from the subsample:\n", column_values_list)
else:
    print(f"Column '{column_name}' does not exist in the dataset.")

# Function to parse OBO and GAF files and print GO annotations for input genes
def parse_go_annotations(gene_list):
    for gene in gene_list:
        go_terms = [0]
        go_terms = gaf_df[gaf_df['Gene'] == gene]['GO']
        print(f"Gene: {gene}")
        print(go_terms)
#        if go_terms:
#            print(f"GO Term: {go_terms[0]} -")
#        else:
#            print(f"No GO terms found for gene: {gene}")

obo_file = "go.obo"  # Path to the OBO file
gaf_file = "goa_human.gaf"  # Path to the GAF file

# Call the function
parse_go_annotations(column_values_list)

'''

def get_go_terms(gene_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{gene_name}&fields=go_id,go_p"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        go_terms = [entry['go'] for entry in data['results'] if 'go' in entry]
        return go_terms
    else:
        return None

for gene in column_values_list:
    go_terms = get_go_terms(gene)
    print(f"Gene: {gene}")
    if go_terms:
        for term in go_terms:
            print(f"GO Term: {term}")
    else:
        print("No GO terms found.")


gp = GProfiler(return_dataframe=True)
genes = ["A4GALT"]
for gene in column_values_list:
    result = gp.profile(organism='hsapiens', query=genes)
    go_terms = result[result['source'].str.startswith('GO')]
    
# Query gProfiler for GO terms

# Filter for GO terms (optional)

# Display the result
print(go_terms[['query', 'name', 'source']])

#df = pd.DataFrame("gene_condition_source_id.txt")
'''
