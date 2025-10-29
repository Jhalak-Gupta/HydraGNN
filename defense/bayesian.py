import torch
import torch.nn.functional as F

def bayesian_predict(model, x, edge_index, runs=10):
    model.train() # Enable dropout for Bayesian prediction
    preds = []
    with torch.no_grad():
        for _ in range(runs):
            out = model(x, edge_index)
            preds.append(F.softmax(out, dim=1))
    
    stacked = torch.stack(preds)
    mean = stacked.mean(0)
    std = stacked.std(0)
    model.eval() # Set back to eval mode
    return mean, std
