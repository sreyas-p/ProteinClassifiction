# ProteinClassifiction
To classify proteins with human functions, we begin with the handling and preprocessing the structural data of proteins. Each protein's structural information is typically stored in files 	from the Protein Data Bank (PDB), which contains atomic coordinates and other relevant information. In order to read this data in a Graph Convolutional Neural Network (GNN), we need to convert the 3D structure into a form suitable for neural network input. This is achieved by representing the protein structure as a graph, where the nodes represent amino acids (or atoms) and the edges represent interactions between them based on spatial proximity.

The first step is to compute a distance matrix for each protein, which measures the distances between the alpha carbon (CA) atoms of amino acids. From this distance matrix, we can generate an adjacency matrix by applying a threshold distance that defines whether two amino acids are considered neighbors. This adjacency matrix effectively captures the topology of the protein's structure, forming the basis for a graph representation that can be used by a GNN.

Simultaneously, we process the amino acid sequence of each protein. The sequence, which is a string of characters representing amino acids, needs to be converted into a numerical format for input into the model. We achieve this by one-hot encoding the sequence, where each amino acid is mapped to a unique binary vector. This encoding allows the sequence information to be fed into the model alongside the structural data.

For training the GNN, we require labels that indicate the biological functions of the proteins. To obtain these labels, we use the Gene Ontology (GO) annotations, which provide a hierarchical classification of gene functions. Specifically, we utilize the goa_human dataset, which contains GO annotations for human proteins. For each protein, we identify the associated gene and retrieve the first GO annotation, which serves as the functional label for that protein in the training process.

In parallel with the GNN processing, we incorporate a language model (LLM) that operates on the protein's amino acid sequence. The LLM processes the sequence data to capture any patterns or relationships that may not be apparent from the structural information alone. This LLM acts as a complementary model that learns from the sequential information, while the GNN focuses on the structural aspects of the protein.

Finally, the outputs from both the GNN and the LLM are combined. The integration of structural data from the GNN and sequential data from the LLM allows the model to leverage both sources of information, providing a comprehensive understanding of the protein's properties. The final model output can then be used to classify the proteins based on their functions, as annotated in the GO database.

Known Genetic Disease → Associated Gene → Protein Structure → Adjacency Matrix → GCN
				|__→ Protein Sequence → LLM/Quantitative Model 
					Binomial Softmax Forecast Collusion → output
