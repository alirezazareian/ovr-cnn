import torch
from torch import nn
import torch.distributed as dist

def stats(tensor):
    t = tensor.cpu().detach().numpy()
    return {
        'device': tensor.device.index,
        'shape': tensor.shape,
        'min': float(tensor.min()),
        'max': float(tensor.max()),
        'mean': float(tensor.to(torch.float32).mean()),
        'std': float(tensor.to(torch.float32).std()),
    }

class LoggedModule(nn.Module):
    def __init__(self):
        super(LoggedModule, self).__init__()
        self.log_info = {}
        self._log_print = False
        self._log_raise_nan = False

    def log(self, name, tensor):
        s = stats(tensor)
        self.log_info[name] = s
        if self._log_print:
            print(f'RANK {dist.get_rank()}: {name}', s)
        if self._log_raise_nan and torch.isnan(tensor).any():
            raise ValueError()

    def log_dict(self, d):
        self.log_info.update(d)
        if self._log_print:
            print(f'RANK {dist.get_rank()}: {d}')
        if self._log_raise_nan:
            for v in d.values():
                if torch.isnan(v).any():
                    raise ValueError()
