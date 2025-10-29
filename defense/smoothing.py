import torch

def randomized_smoothing(model, x, edge_index, sigma=0.1, samples=10):
    model.eval()
    predictions = []
    with torch.no_grad():
        for _ in range(samples):
            noisy_x = x + sigma * torch.randn_like(x)
            out = model(noisy_x, edge_index)
            predictions.append(out.argmax(dim=1))
    
    predictions = torch.stack(predictions)
    smoothed_preds = predictions.mode(0)[0]
    return smoothed_preds
