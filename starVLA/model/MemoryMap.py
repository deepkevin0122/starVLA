import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union, Dict, Any
import math
import json
import os
import pickle
from pathlib import Path
import numpy as np
import torch.distributed as dist

class MemoryMap(nn.Module):
    def __init__(self, map_x: int, map_y: int, memory_dim: int,
                 update_rate: float = 0.1, drop_rate: float = 0.1, 
                 gnn_update_rate: float = 0.05, 
                 dtype=torch.float32):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.memory_dim = memory_dim
        self.update_rate = update_rate
        self.drop_rate = drop_rate
        self.gnn_update_rate = gnn_update_rate
        self.dtype = dtype
        self.norm_ext = nn.LayerNorm(memory_dim, dtype=dtype)
        self.norm_mem = nn.LayerNorm(memory_dim, dtype=dtype)
        self.input_norm = nn.LayerNorm(memory_dim, dtype=dtype)
        
        self.temperature = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(memory_dim)))
        
        if dist.is_initialized():
            dist.barrier()
        self._initialize_buffers()
        
        self.neighbor_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        self.attention_query_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        nn.init.xavier_uniform_(self.attention_query_proj.weight)
        

    def _initialize_buffers(self) -> None:
        mem_shape = (self.map_x, self.map_y, self.memory_dim)
        mem_tensor = torch.empty(mem_shape, dtype=self.dtype)
        nn.init.trunc_normal_(mem_tensor, std=0.02)
        occ = torch.ones((1, 1, self.map_x, self.map_y), dtype=self.dtype)
        k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=self.dtype).view(1, 1, 3, 3)
        count = F.conv2d(occ, k, padding=1)
        self.register_buffer("memory", mem_tensor)
        self.register_buffer("gnn_norm", count)
        self.register_buffer("update_counts", torch.zeros((self.map_x, self.map_y), dtype=torch.long))
        self.register_buffer("last_update_step", torch.zeros(1, dtype=torch.long))
        
    def forward(self, ext: torch.Tensor, 
                upd_poss: Optional[List[List[List[int]]]] = None,
                update: Optional[bool] = True) -> torch.Tensor:
        """
        Args:
            ext: [B, VLM_embedding_size] 外部输入特征
            upd_poss: 三维列表，每个batch包含多个[x,y]位置坐标
            update: 是否更新memory
        Returns:
            global_embedding: [B, memory_dim] 在ext注意力下的全局embedding
        """
        self.update = update
        batch_size = ext.shape[0]
        processed_ext = self._process_external_input(ext)
        print("self.update", self.update, flush=True)
        if self.update:
            self._update_memory_dynamic(processed_ext, upd_poss)
            self.last_update_step += 1
        global_embedding = self._compute_similarity_based_embedding(processed_ext)
        
        return global_embedding
    
    def _process_external_input(self, ext: torch.Tensor) -> torch.Tensor:
        """
        处理外部输入，调整到 memory_dim 维度并添加梯度约束。
        """
        processed = torch.tanh(self.input_norm(ext))
        if self.update:
            processed.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
        return processed.to(ext.device)
    

    def _update_memory_dynamic(self, ext: torch.Tensor, upd_poss: List[List[List[int]]]):
        device = ext.device
        if self.memory.device != device:
            self.memory = self.memory.to(device)
        coords = []
        val_indices = []
        for i, pos_list in enumerate(upd_poss):
            for x, y in pos_list:
                if 0 <= x < self.map_x and 0 <= y < self.map_y:
                    coords.append([x, y])
                    val_indices.append(i)
        
        if not coords:
            return current_mem

        coords_tensor = torch.tensor(coords, device=device, dtype=torch.long)
        flat_indices = coords_tensor[:, 0] * self.map_y + coords_tensor[:, 1]
        
        B, D = ext.shape[0], self.memory_dim
        M = self.map_x * self.map_y

        updates = ext[val_indices] # [N, D]
        old_vals = self.memory.view(-1, self.memory_dim)[flat_indices]
        
        diff = (updates - old_vals) * self.update_rate
        
        delta = torch.zeros((self.map_x * self.map_y, self.memory_dim), 
                            device=device, dtype=self.dtype)
        delta.index_add_(0, flat_indices, diff)

        if dist.is_initialized():
            delta_d = delta.detach()
            dist.all_reduce(delta_d, op=dist.ReduceOp.AVG)
            delta = delta_d

        new_mem = self.memory.view(-1, self.memory_dim) + delta
        new_mem = new_mem.view(self.map_x, self.map_y, self.memory_dim)

        updated_memory = self._apply_gnn_propagation_graph_safe(new_mem)
        with torch.no_grad():
            self.memory.data.copy_(updated_memory.data)
    
    def _apply_gnn_propagation_graph_safe(self, mem_in: torch.Tensor) -> torch.Tensor:
        """
        使用卷积形式实现的张量化 GNN 传播，保证计算图完整。
        mem_in: [H, W, C] - 刚刚更新完、带有 ext 梯度的记忆张量
        """
        device = mem_in.device
        alpha = float(self.gnn_update_rate)
        K = 2  # 传播步数
        
        kernel = torch.tensor([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=mem_in.dtype, device=device)
        
        kernel = kernel.view(1, 1, 3, 3).expand(self.memory_dim, 1, 3, 3)

        src = mem_in.permute(2, 0, 1).unsqueeze(0)
        
        norm_factor = self.gnn_norm.clamp(min=1.0).to(device)

        for _ in range(K):
            neighbor_sum = F.conv2d(src, kernel, padding=1, groups=self.memory_dim)
            
            neighbor_mean = neighbor_sum / norm_factor
            
            src = (1.0 - alpha) * src + alpha * neighbor_mean

        # 3. 还原维度 [1, C, H, W] -> [H, W, C]
        return src.squeeze(0).permute(1, 2, 0)
    
    def _compute_similarity_based_embedding(self, ext: torch.Tensor) -> torch.Tensor:
        """强化后的数值稳定注意力机制"""
        batch_size = ext.shape[0]
        
        ext_normalized = self.norm_ext(ext)
        mem_normalized = self.norm_mem(self.memory.reshape(-1, self.memory_dim))
        queries_norm = F.normalize(self.attention_query_proj(ext_normalized), p=2, dim=-1)
        memory_norm = F.normalize(mem_normalized, p=2, dim=-1)
        t = self.temperature.clamp(min=0.01)
        similarity = torch.matmul(queries_norm, memory_norm.T) * t
        attention_weights = F.softmax(similarity, dim=-1)
        if self.update:
            attention_weights.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
        global_embedding = torch.matmul(attention_weights, self.memory.reshape(-1, self.memory_dim))
        
        return global_embedding