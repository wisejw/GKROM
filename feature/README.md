## Heterogeneous Graph Construction (For the MOOC Dataset):

1. Download GloVe using `wget http://nlp.stanford.edu/data/glove.6B.zip` command.
2. Unzip the glove6B.zip using `unzip glove*.zip` command.
3. Use this file (`glove.6B.300d.txt`) or file path in the glove_path variable in feature.py file.
4. To create heterogeneous graph construction, run `python feature/heterogeneous_graph_construction.py` file.