"""Device selection shared by portable YOLACT entry points."""
import torch

def select_device(requested='auto'):
    mps = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'mps' if mps else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable in this Python environment.')
    if requested == 'mps' and not mps:
        raise RuntimeError('MPS requested but unavailable. Use Apple Silicon and an MPS-enabled PyTorch build.')
    return torch.device(requested)

def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)
    elif device.type == 'mps': torch.mps.synchronize()
