import torch
import torch.nn.functional as F
import time
import psutil
from memory_profiler import memory_usage

def train(model, data, optimizer, edge_index=None):
    model.train()
    optimizer.zero_grad()
    
    edge_idx = edge_index if edge_index is not None else data.edge_index
    
    out = model(data.x, edge_idx)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, data, edge_index=None, mask=None):
    model.eval()
    mask = data.test_mask if mask is None else mask
    edge_idx = edge_index if edge_index is not None else data.edge_index
    
    with torch.no_grad():
        out = model(data.x, edge_idx)
        pred = out.argmax(dim=1)
      
        if mask.sum() == 0:
            return 0.0
            
        correct = (pred[mask] == data.y[mask]).sum()
        acc = correct / mask.sum()
        return acc.item()

def calculate_asr(model, data, target_class=0):
    model.eval()
    target_mask = (data.y == target_class) & data.test_mask
    
    if target_mask.sum() == 0:
        return float('nan') 
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out if out.dim() == 1 else out.argmax(dim=1)
        misclassified = (pred[target_mask] != data.y[target_mask]).sum()
        
    return (misclassified / target_mask.sum()).item()

def measure_resources(func, *args):
    process = psutil.Process()

    cpu_times_start = process.cpu_times()
    start_time = time.time()
  
    mem_usage, result = memory_usage((func, args), retval=True, max_usage=True, interval=0.1)

    end_time = time.time()
    cpu_times_end = process.cpu_times()
    duration = end_time - start_time

    cpu_time_used = (cpu_times_end.user - cpu_times_start.user) + \
                    (cpu_times_end.system - cpu_times_start.system)
    
    cpu_percent = (cpu_time_used / duration) * 100 if duration > 0 else 0
    return duration, mem_usage, cpu_percent
  
