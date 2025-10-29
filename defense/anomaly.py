import torch

def detect_anomalous_nodes(embeddings):
    scores = torch.std(embeddings, dim=1)
    threshold = scores.mean() + 2 * scores.std()
    return (scores > threshold).nonzero(as_tuple=True)[0]
