import torch
import random

def apply_poisoning(data, num_classes, target_class=0, poisoning_rate=0.05):
    num_nodes = data.x.size(0)
    num_fake = int(num_nodes * poisoning_rate)
    
    # Generate fake features based on dataset stats
    feature_means = data.x.mean(dim=0)
    feature_stds = data.x.std(dim=0)
    fake_features = torch.randn(num_fake, data.x.size(1)) * feature_stds + feature_means
    
    # Add fake nodes' features
    data.x = torch.cat([data.x, fake_features], dim=0)
    
    # Find nodes belonging to the target class
    target_indices = (data.y == target_class).nonzero(as_tuple=True)[0]
    
    new_edges = []
    for i in range(num_fake):
        target = random.choice(target_indices)
        # Add edges between new fake node and a target node
        new_edges.append([num_nodes + i, target])
        new_edges.append([target, num_nodes + i])
        
    new_edge_index = torch.tensor(new_edges).t().contiguous()
    data.edge_index = torch.cat([data.edge_index, new_edge_index], dim=1)
    
    # Add fake labels (randomly assigned)
    fake_labels = torch.randint(0, num_classes, (num_fake,))
    data.y = torch.cat([data.y, fake_labels])

    # Extend train_mask to include new fake nodes
    extended_train_mask = torch.zeros(len(data.x), dtype=torch.bool)
    extended_train_mask[:len(data.train_mask)] = data.train_mask
    extended_train_mask[len(data.train_mask):] = True  # Add fake nodes to train set
    data.train_mask = extended_train_mask

    # Extend val_mask and test_mask (without adding fake nodes to them)
    for mask_name in ['val_mask', 'test_mask']:
        if hasattr(data, mask_name):
            old_mask = getattr(data, mask_name)
            new_mask = torch.zeros(len(data.x), dtype=torch.bool)
            new_mask[:len(old_mask)] = old_mask
            setattr(data, mask_name, new_mask)
            
    return data
