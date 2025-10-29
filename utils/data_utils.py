from torch_geometric.datasets import Planetoid
from torch_geometric.utils import add_self_loops

def load_data(name='Cora', root_dir='data/Cora'):
    dataset = Planetoid(root=root_dir, name=name)
    data = dataset[0]
    data.edge_index, _ = add_self_loops(data.edge_index)
    return dataset, data
