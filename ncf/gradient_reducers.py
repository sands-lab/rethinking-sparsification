import datetime
import os
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import torch



class Reducer:
    def __init__(self, random_seed, device, timer):
        self.rng = np.random.RandomState(random_seed)
        M = 1024 * 1024
        self.precalc_numbers = (
            torch.from_numpy(self.rng.randn(128 * M)).to(device).type(torch.float32)
        )
        if torch.distributed.is_available():
            self.n_workers = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.n_workers = 1
            self.rank = 0
        self.device = device
        self.timer = timer

    def reduce(self, grad_in, grad_out, memory_out):
        """Return communicated bits"""
        raise NotImplementedError()

class TopKReducer(Reducer):
    """
    Use same amount as rank-based
    """
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0

        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                top_size = max(1, int(self.compression * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                top_size = max(1, int(self.compression * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
            
        return bits_communicated, params_transmitted

class GlobalTopKReducer(Reducer):
    def __init__(self, random_seed, device, timer, compression=1 / 244):
        super().__init__(random_seed, device, timer)
        self.compression = compression

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0

        with self.timer("reduce.flatpack"):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for tensor in grad_in:
                n = tensor.nelement()
                flatgrad_size += n
                tensor_idx.append(tensor_idx[-1] + n)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flatgrad = torch.empty(flatgrad_size, device=self.device)

            # Pack the flatgrad
            for tensor, start, end in zip(grad_in, flatgrad_start_idx, flatgrad_end_idx):
                flatgrad[start:end] = tensor.view(-1)

        top_size = max(1, int(self.compression * flatgrad.nelement()))

        with self.timer("reduce.topk", verbosity=2):
            _, positions = torch.topk(flatgrad.abs(), top_size, sorted=False)
            values = flatgrad[positions].contiguous()

        with self.timer("reduce.set_memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                local_positions = positions[(positions >= start) & (positions < end)] - start
                mem.data[:] = tensor
                mem.view(-1)[local_positions] = 0.0

        with self.timer("reduce.reduce", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, values, async_op=True)
                h2 = all_gather(worker_positions, positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [values]
                worker_positions = [positions]
            bits_communicated += n_bits(values) + n_bits(positions)
            params_transmitted += values.numel()

        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0.0
                for pos, val in zip(worker_positions, worker_values):
                    local_positions = pos[(pos >= start) & (pos < end)] - start
                    local_vals = val[(pos >= start) & (pos < end)]
                    out.view(-1)[local_positions] += local_vals / self.n_workers
            
        return bits_communicated, params_transmitted

class ThreshReducer(Reducer):
     
    def __init__(self, random_seed, device, timer, thresh=0.5):
        super().__init__(random_seed, device, timer)
        self.threshold = thresh

    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        bits_communicated = 0
        params_transmitted = 0
        
        tensors_compressed = []
        compressed_positions = []
        local_sizes = []
        with self.timer("reduce.threshold", verbosity=2):
            for tensor in grad_in:
                positions, =  torch.where(tensor.view(-1).abs()>=self.threshold)
                values = tensor.view(-1)[positions].contiguous()
                tensors_compressed.append(values)
                compressed_positions.append(positions)
                local_sizes.append(values.numel())

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, positions in zip(
                grad_in, memory_out, compressed_positions
            ):
                mem.data[:] = tensor
                mem.view(-1)[positions] = 0.0
                
        with self.timer("reduce.flatpack", verbosity=2):
            flatgrad_size = 0
            tensor_idx = [0]
            for local_size in local_sizes:
                flatgrad_size += local_size
                tensor_idx.append(tensor_idx[-1] + local_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)
            
        with self.timer("reduce.flatput", verbosity=2):
            for values, positions, start, end in zip(tensors_compressed, compressed_positions, flatgrad_start_idx, flatgrad_end_idx):
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.gather.context", verbosity=2):
            flatgrad_size = torch.tensor(flatgrad_size, device = self.device)
            flatgrad_start_idx = torch.tensor(flatgrad_start_idx, device = self.device)
            flatgrad_end_idx = torch.tensor(flatgrad_end_idx, device = self.device)
            if self.n_workers > 1:
                gathered_sizes = [torch.empty_like(flatgrad_size) for i in range(self.n_workers)]
                h1 = all_gather(gathered_sizes, flatgrad_size, async_op = True)
                gathered_start_indices = [torch.empty_like(flatgrad_start_idx) for i in range(self.n_workers)]
                h2 = all_gather(gathered_start_indices, flatgrad_start_idx, async_op = True)
                gathered_end_indices = [torch.empty_like(flatgrad_end_idx) for i in range(self.n_workers)]
                h3 = all_gather(gathered_end_indices, flatgrad_end_idx, async_op = True)
                h1.wait()
                h2.wait()
                h3.wait()
            else:
                gathered_sizes = [flatgrad_size]
                gathered_start_indices = [flatgrad_start_idx]
                gathered_end_indices = [flatgrad_end_idx]
                
        with self.timer("reduce.pad", verbosity=2):
            if self.n_workers > 1:
                max_size = max(gathered_sizes)
                if flatgrad_size != max_size:
                    padding_values = torch.empty(max_size-flatgrad_size, dtype=flat_values.dtype, device=flat_values.device)
                    padding_positions = torch.empty(max_size-flatgrad_size, dtype=flat_positions.dtype, device=flat_values.device)
                    flat_values = torch.cat((flat_values, padding_values), dim=0)
                    flat_positions = torch.cat((flat_positions, padding_positions), dim=0)
                
        with self.timer("reduce.gather.tensors", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
        
        with self.timer("reduce.combine", verbosity=2):
            for out, start_indices, end_indices in zip(
                grad_out, zip(*gathered_start_indices), zip(*gathered_end_indices)
            ):
                out.data[:] = 0
                for pos, val, start, end in zip(worker_positions, worker_values, start_indices, end_indices):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
                    
        return bits_communicated, params_transmitted
class AccordionTopKReducer(Reducer):
    """
    Modified from https://github.com/uw-mad-dash/Accordion
    """
    def __init__(self, random_seed, device, timer, k_low=0.1, k_high=0.99, detection_threshold=0.5, switch_freq=10):
        super().__init__(random_seed, device, timer)
        self.k_low = k_low
        self.k_high = k_high
        self.detection_threshold = detection_threshold
        self.switch_freq = switch_freq

    def reduce(self, grad_in, grad_out, memory_out, auto_scale_tensor, prev_norms, curr_norms, prev_lrs, curr_lrs, epoch_count):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        :auto_scale_tensor: tensor
        :prev_norms: list
        curr_norms: list
        prev_lrs:list
        curr_lrs:list
        """
        bits_communicated = 0
        params_transmitted = 0

        with self.timer("reduce.autoscale", verbosity=2):
            # Determine compression ratio for the next switch_freq epochs
            if epoch_count%self.switch_freq == 0:
                for i, grad in enumerate(grad_out):
                    curr_norms[i] = l2norm(grad)
                    if epoch_count == 0 or (prev_lrs[i] > curr_lrs[i]) or abs(prev_norms[i]-curr_norms[i])/prev_norms[i] > self.detection_threshold:
                        auto_scale_tensor[i] = self.k_high
                    else:
                        auto_scale_tensor[i] = self.k_low
                    prev_norms[i] = curr_norms[i]
                    prev_lrs[i] = curr_lrs[i]
                #Broadcast the low and high rank values from rank 0
                torch.distributed.broadcast(auto_scale_tensor, src=0)

        with self.timer("reduce.flatpack", verbosity=2):
            # Find the size of a flatpacked gradient
            flatgrad_size = 0
            tensor_idx = [0]
            for i, tensor in enumerate(grad_in):
                top_size = max(1, int(auto_scale_tensor[i].item() * tensor.nelement()))
                flatgrad_size += top_size
                tensor_idx.append(tensor_idx[-1] + top_size)
            flatgrad_start_idx = tensor_idx[:-1]
            flatgrad_end_idx = tensor_idx[1:]
            flat_values = torch.empty(flatgrad_size, device=self.device)
            flat_positions = torch.empty(flatgrad_size, device=self.device, dtype=torch.int)

        with self.timer("reduce.topk", verbosity=2):
            for i, (tensor, start, end) in enumerate(zip(grad_in, flatgrad_start_idx, flatgrad_end_idx)):
                top_size = max(1, int(auto_scale_tensor[i].item() * tensor.nelement()))
                _, positions = torch.topk(tensor.view(-1).abs(), top_size, sorted=False)
                #_, indices = (tensor.view(-1).abs()).sort(descending = True)
                #positions = indices[:top_size]
                values = tensor.view(-1)[positions].contiguous()
                flat_values[start:end] = values
                flat_positions[start:end] = positions

        with self.timer("reduce.memory", verbosity=2):
            for tensor, mem, start, end in zip(
                grad_in, memory_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                positions = flat_positions[start:end]
                mem.data[:] = tensor
                mem.view(-1)[positions.long()] = 0.0

        with self.timer("reduce.gather", verbosity=2):
            if self.n_workers > 1:
                worker_values = [torch.empty_like(flat_values) for i in range(self.n_workers)]
                worker_positions = [torch.empty_like(flat_positions) for i in range(self.n_workers)]
                h1 = all_gather(worker_values, flat_values, async_op=True)
                h2 = all_gather(worker_positions, flat_positions, async_op=True)
                h1.wait()
                h2.wait()
            else:
                worker_values = [flat_values]
                worker_positions = [flat_positions]
            bits_communicated = n_bits(flat_values) + n_bits(flat_positions)
            params_transmitted = flat_values.numel()
            
        with self.timer("reduce.combine", verbosity=2):
            for tensor, out, start, end in zip(
                grad_in, grad_out, flatgrad_start_idx, flatgrad_end_idx
            ):
                out.data[:] = 0
                for pos, val in zip(worker_positions, worker_values):
                    positions = pos[start:end]
                    values = val[start:end]
                    # out.view(-1)[pos].add_(1.0 / self.n_workers, val)
                    out.view(-1)[positions.long()] += values / self.n_workers
            
        return bits_communicated, params_transmitted


@torch.jit.script
def orthogonalize(matrix):
    n, m = matrix.shape
    for i in range(m):
        # Normalize the i'th column
        col = matrix[:, i : i + 1]
        col /= torch.sqrt(torch.sum(col ** 2))
        # Project it on the rest and remove it
        if i + 1 < m:
            rest = matrix[:, i + 1 :]
            # rest -= torch.matmul(col.t(), rest) * col
            rest -= torch.sum(col * rest, dim=0) * col


class ExactReducer(Reducer):
    def reduce(self, grad_in, grad_out, memory_out):
        """
        Reduce gradients between the workers in place
        :param grad_in: dictionary
        :param grad_out: dictionary
        :param memory_out: dictionary
        """
        with self.timer("reduce.zero_mem", verbosity=2):
            for mem in memory_out:
                mem.zero_()

        with self.timer("reduce.build_lists", verbosity=2):
            list_in = grad_in
            list_out = grad_out

        with self.timer("reduce.reduce", verbosity=2):
            bits_communicated, params_transmitted = reduce_mean_list(self.device, list_in, list_out, self.timer)

        return bits_communicated, params_transmitted


def reduce_mean_list(
    device: torch.device, list_in: List[torch.Tensor], list_out: List[torch.Tensor], timer
):
    if torch.distributed.is_available():
        n_workers = torch.distributed.get_world_size()
    else:
        n_workers = 1

    if n_workers == 1:
        for t_in, t_out in zip(list_in, list_out):
            t_out[:] = t_in
        return 0,0

    with timer("reduce.mean.pack"):
        buffer = TensorBuffer(list_in)

    with timer("reduce.mean.allreduce"):
        buffer.all_reduce()
        buffer.buffer /= n_workers
        bits_communicated = buffer.bits()
        params_transmitted = buffer.nelement()

    with timer("reduce.mean.unpack", verbosity=2):
        buffer.unpack(list_out)
        
    return bits_communicated, params_transmitted


def n_bits(tensor):
    return 8 * tensor.nelement() * tensor.element_size()

class TensorBuffer():
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """
    def __init__(self, tensors):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors = tensors

        self.buffer = torch.cat([t.view(-1) for t in tensors]) # copies
    
    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(*self._tensors[index].shape)

    def __len__(self):
        return len(self._tensors)

    def pack(self, tensors=None):
        # Optional. init already does this.
        if tensors is None:
            tensors = self._tensors
        for tensor, entry in zip(tensors, self):
            entry[:] = tensor

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor[:] = entry

    def nelement(self):
        return self.buffer.nelement()

    def element_size(self):
        return self.buffer.element_size()

    def bits(self):
        return 8 * self.nelement() * self.element_size()

    def all_reduce(self, async_op=False):
        return torch.distributed.all_reduce(self.buffer, async_op=async_op)
    
    def all_gather(self, async_op=False):
        n_workers = torch.distributed.get_world_size() if torch.distributed.is_available() else 1
        buffers = [torch.empty_like(self.buffer) for i in range(n_workers)]
        handle = all_gather(buffers, self.buffer, async_op=async_op)
        if async_op:
            return buffers, handle
        else:
            return buffers
    

def all_reduce(*args, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_reduce(*args, **kwargs)


def all_gather(out_list, in_tensor, **kwargs):
    if torch.distributed.is_available() and torch.distributed.get_world_size() > 1:
        return torch.distributed.all_gather(out_list, in_tensor, **kwargs)
    else:
        assert len(out_list) == 1
        out_list[0].data = in_tensor


@torch.jit.script
def l2norm(x):
    return torch.sqrt(torch.sum(x ** 2))


def normalize_(tensor):
    """Divide by L2 norm. In place"""
    tensor /= l2norm(tensor)
