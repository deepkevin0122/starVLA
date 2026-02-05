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
                 persistence_path: str = "/memory",
                 dtype=torch.bfloat16):
        super().__init__()
        self.map_x = map_x
        self.map_y = map_y
        self.memory_dim = memory_dim
        self.update_rate = update_rate
        self.drop_rate = drop_rate
        self.gnn_update_rate = gnn_update_rate
        self.persistence_path = persistence_path
        self.dtype = dtype
        # 1. 引入 LayerNorm 稳定输入分布
        self.norm_ext = nn.LayerNorm(memory_dim, dtype=dtype)
        self.norm_mem = nn.LayerNorm(memory_dim, dtype=dtype)
        self.input_norm = nn.LayerNorm(memory_dim, dtype=dtype)
        self.input_projection = nn.Linear(memory_dim, memory_dim, bias=False, dtype=dtype)
        self.current_mem = None
        nn.init.orthogonal_(self.input_projection.weight)
        
        # 2. 可学习的缩放因子（代替固定的 sqrt(D)）
        # 初始化为 1/sqrt(D)，给 Softmax 一个平滑的开始
        self.temperature = nn.Parameter(torch.ones(1) * (1.0 / math.sqrt(memory_dim)))
        self.persistence_path = persistence_path
            
        # 确保保存目录存在
        if not dist.is_initialized() or dist.get_rank() == 0:
            os.makedirs(self.persistence_path, exist_ok=True)
        
        if dist.is_initialized():
            dist.barrier()
        self._initialize_buffers()
        
        # 四相邻偏移量
        self.neighbor_offsets = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        # 注意力查询投影（可学习）
        self.attention_query_proj = nn.Linear(memory_dim, memory_dim, bias=False)
        nn.init.xavier_uniform_(self.attention_query_proj.weight)
        self.load_memory_state()

    def _initialize_buffers(self) -> None:
        mem_shape = (self.map_x, self.map_y, self.memory_dim)
        mem_tensor = torch.empty(mem_shape, dtype=self.dtype)
        nn.init.trunc_normal_(mem_tensor, std=0.02)
        self.memory = mem_tensor
        
        # 4. 其他统计信息保持为 buffer
        occ = torch.ones((1, 1, self.map_x, self.map_y), dtype=self.dtype)
        k = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=self.dtype).view(1, 1, 3, 3)
        # count 记录了每个位置实际有几个邻居（2, 3 或 4）
        count = F.conv2d(occ, k, padding=1)
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
        batch_size = ext.shape[0]
        
        processed_ext = self._process_external_input(ext)
        
        if self.current_mem is not None and self.current_mem.grad is not None:
            self.current_mem.grad.detach_()
            self.current_mem.grad = None
            del self.current_mem
            self.current_mem = None
        if self.training and upd_poss is not None:
            self.current_mem = nn.Parameter(self.memory.clone().detach().require_grad_(True))
            self._update_memory_dynamic(self.current_mem, processed_ext, upd_poss)
            self.last_update_step += 1
        else:
            self.memory = self.memory.detach()
        global_embedding = self._compute_similarity_based_embedding(processed_ext)
        
        return global_embedding
    
    def _process_external_input(self, ext: torch.Tensor) -> torch.Tensor:
        """
        处理外部输入，调整到 memory_dim 维度并添加梯度约束。
        """
        processed = torch.tanh(self.input_norm(ext))
        if self.training:
            processed.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
        return processed
    

    def _update_memory_dynamic(self, current_mem: torch.Tensor, ext: torch.Tensor, upd_poss: List[List[List[int]]]):
        """
        通过掩码融合生成带有梯度流的新记忆副本
        """
        device = ext.device

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
        device = ext.device

        updates = ext[val_indices] # [N, D]
        old_vals = current_mem.view(-1, self.memory_dim)[flat_indices]
        
        diff = (updates - old_vals) * self.update_rate
        
        delta = torch.zeros((self.map_x * self.map_y, self.memory_dim), 
                            device=device, dtype=self.dtype)
        delta.index_add_(0, flat_indices, diff)

        if dist.is_initialized():
            dist.all_reduce(delta, op=dist.ReduceOp.AVG)

        new_mem = current_mem.view(-1, self.memory_dim) + delta
        new_mem = new_mem.view(self.map_x, self.map_y, self.memory_dim)

        self.memory = self._apply_gnn_propagation_graph_safe(new_mem)
    
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
        
        norm_factor = self.gnn_norm.clamp(min=1.0)

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
        if self.training:
            attention_weights.register_hook(lambda grad: torch.clamp(grad, -0.1, 0.1))
        global_embedding = torch.matmul(attention_weights, self.memory.reshape(-1, self.memory_dim))
        
        return global_embedding
    
    def save_memory_state(self, suffix) -> str:
        """
        保存记忆状态到文件
        
        Args:
            suffix: 文件名后缀，用于区分不同版本
        
        Returns:
            保存的文件路径
        """
        if dist.is_initialized() and dist.get_rank() != 0:
            return ""   
        filename = f"memory_state_{self.map_x}_{self.map_y}_{self.memory_dim}_{suffix}.pth"
        save_path = os.path.join(self.persistence_path, filename)
        state_dict = {
            'memory': self.memory.cpu(),
            'update_counts': self.update_counts.cpu(),
            'last_update_step': self.last_update_step.cpu(),
            'dtype': str(self.dtype)
        }
        torch.save(state_dict, save_path)
        print("save memory state to", save_path)
        return save_path
    
    def load_memory_state(self, filepath: Optional[str] = None) -> bool:
        """
        从文件加载记忆状态，支持 8 卡同步
        """
        
        # 定义一个变量来同步加载结果
        success = torch.tensor(0, device=self.memory.device)

        try:
            # 1. 只有主进程负责查找文件逻辑（如果是自动查找模式）
            if filepath is None and (not dist.is_initialized() or dist.get_rank() == 0):
                memory_files = list(Path(self.persistence_path).glob(f"memory_state_{self.map_x}_{self.map_y}_{self.memory_dim}_*.pth"))
                if memory_files:
                    latest_file = max(memory_files, key=os.path.getmtime)
                    filepath = str(latest_file)
            
            # 2. 如果是分布式环境，主进程将确定好的 filepath 广播给所有人（防止各读各的版本）
            if dist.is_initialized():
                # 注意：这里需要将字符串路径同步给所有进程
                path_list = [filepath]
                dist.broadcast_object_list(path_list, src=0)
                filepath = path_list[0]

            # 3. 如果路径依然为空，说明真的没找到
            if filepath is None:
                if not dist.is_initialized() or dist.get_rank() == 0:
                    print("No existing memory state found, using initial state")
                return False

            # 4. 所有卡同步读取文件
            # 使用 weights_only=True 是安全实践
            state_dict = torch.load(filepath, map_location=self.memory.device, weights_only=True)
            
            # 5. 复制数据到 buffer
            self.memory.data.copy_(state_dict['memory'].to(self.memory.device, dtype=self.dtype))
            self.update_counts.data.copy_(state_dict['update_counts'].to(self.update_counts.device))
            self.last_update_step.data.copy_(state_dict['last_update_step'].to(self.last_update_step.device))
            
            success += 1
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"✅ Memory state successfully loaded from {filepath}")

        except Exception as e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"❌ Failed to load memory state: {e}")
            success *= 0

        # 6. 【关键】设置屏障，确保所有人加载完毕再走
        if dist.is_initialized():
            dist.barrier()
        
        return success.item() > 0