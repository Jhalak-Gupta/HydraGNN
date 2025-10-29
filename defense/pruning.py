import torch
import networkx as nx
import random
from torch_geometric.utils import to_networkx, from_networkx

def prune_and_restore_edges(data, threshold=0.001):
    G = to_networkx(data, to_undirected=True)
    degree_centrality = nx.degree_centrality(G)
    nodes_to_keep = [node for node, score in degree_centrality.items() if score > threshold]
    
    if len(nodes_to_keep) == 0:
        nodes_to_keep = list(G.nodes())
        
    G_pruned = G.subgraph(nodes_to_keep).copy()
    components = list(nx.connected_components(G_pruned))
    
    while len(components) > 1:
        comp1, comp2 = random.sample(components, 2)
        node1 = random.choice(list(comp1))
        node2 = random.choice(list(comp2))
        G_pruned.add_edge(node1, node2)
        components = list(nx.connected_components(G_pruned))
        
    defended_data = from_networkx(G_pruned)
    
    # Map old node indices to new ones based on the subgraph
    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(G_pruned.nodes())}

    kept_indices = list(G_pruned.nodes())
    kept_indices_tensor = torch.tensor(kept_indices, dtype=torch.long)

    defended_data.x = data.x[kept_indices_tensor]
    defended_data.y = data.y[kept_indices_tensor]
    
    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, mask_name):
            mask = getattr(data, mask_name)
            setattr(defended_data, mask_name, mask[kept_indices_tensor])
            
    return defended_data
