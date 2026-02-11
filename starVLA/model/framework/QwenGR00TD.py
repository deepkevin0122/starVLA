# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Junqiu YU / Fudan University] in [2025]. 
# Design and Merged by [Jinhui YE / HKUST University] in [2025].
"""
Qwen-GR00T Framework
A lightweight implementation that Qwen-VL + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5,
"""
import sys
from pathlib import Path

# Add workspace root to Python path if not already there
_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os



from starVLA.training.trainer_utils import initialize_overwatch
from deployment.model_server.tools.image_tools import to_pil_preserve

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.model.MemoryMap import MemoryMap
from starVLA.model.tools import build_change_flow_map, _calculate_update_positions, _to_uint8_img


@FRAMEWORK_REGISTRY.register("QwenGR00TD")
class Qwen_GR00TD(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        # align dims --> we should put them to config or no?
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_vl_interface.model.config.hidden_size

        self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        self.map_x = self.config.framework.action_model.get("memory_map_x_size", 4)
        self.map_y = self.config.framework.action_model.get("memory_map_y_size", 4) 
        self.memory_update_topk = self.config.framework.action_model.get("memory_update_topk", 4)
        self.memory = MemoryMap(
            map_x=self.map_x,
            map_y=self.map_y,
            memory_dim=self.qwen_vl_interface.model.config.hidden_size,
            update_rate=self.config.framework.action_model.get("memory_update_rate", 0.5),
            drop_rate=self.config.framework.action_model.get("memory_dropout", 0.1),
            gnn_update_rate=self.config.framework.action_model.get("memory_gnn_update_rate", 0.05),
            dtype=torch.float32
        )
        self.image_size = (224, 224)  # default image size
        self.linear_action_pred = nn.Linear(
            self.qwen_vl_interface.model.config.hidden_size * 2,
            self.config.framework.action_model.get("action_dim", 7)
        )
        self.step = 0

    import numpy as np
    import cv2
    from typing import List, Tuple, Union, Any

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """

        """
        self.step = self.step + 1
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        batch_las_images = [example["las_image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        action_keys = [example["action_keys"] for example in examples] # [B, str]
        batch_las_action = [example["las_action"] for example in examples] # [B, len, 7]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        # Step 1: Build Change Map and FlowMap
        batch_change_map, batch_flow_map, batch_hsv_map = build_change_flow_map(batch_images, batch_las_images)

        os.makedirs("./debug/train/images", exist_ok=True)
        os.makedirs("./debug/train/text", exist_ok=True)
        bcm = batch_change_map[0][0]
        bfm = batch_flow_map[0][0]
        bhm = batch_hsv_map[0][0]

        _to_uint8_img(bcm).save(f"./debug/train/images/bcm_{self.step}.png")
        _to_uint8_img(bfm).save(f"./debug/train/images/bfm_{self.step}.png")
        _to_uint8_img(bhm).save(f"./debug/train/images/bhm_{self.step}.png")

        # Calculate Memory Map Update Positions based on HSV brightness
        upd_poss = _calculate_update_positions(batch_hsv_map, self.map_x, self.map_y, self.image_size, self.memory_update_topk)
        with open("./debug/train/text/upd_ppos_{self.step}.txt","w",encoding="utf-8") as f:
            for i in range(len(upd_poss)):
                f.write(",".join(map(str,upd_poss[i]))+"\n")
        # Step 2: QWenVL Inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs_pro(
            images=batch_images, 
            action_keys=action_keys, 
            change_maps=batch_change_map,
            flow_maps=batch_flow_map,
            instructions=instructions
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Step 3: Forward QwenVL
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            calc_hidden = last_hidden.mean(dim=1).float()
            memory = self.memory(
                ext=calc_hidden,
                upd_poss=upd_poss,
                update=True
            ).to(device=last_hidden.device, dtype=torch.float32)

            batch_las_action = torch.tensor(np.array(batch_las_action), device=last_hidden.device, dtype=torch.float32)
            batch_las_action = batch_las_action[:, -(self.future_action_window_size+1), :] 

            concat_feat = torch.cat([calc_hidden, memory], dim=-1) 
            las_action_pred = self.linear_action_pred(concat_feat) 
            las_action_loss = F.l1_loss(las_action_pred, batch_las_action).float()

            last_hidden = torch.cat([last_hidden, memory.unsqueeze(1).to(dtype=last_hidden.dtype)], dim=1)
            actions = torch.tensor(np.array(actions), device=last_hidden.device, dtype=last_hidden.dtype)
            actions_target = actions[:, -(self.future_action_window_size+1):, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
            state_repeated = None

            action_loss = self.action_model(last_hidden_repeated, actions_target_repeated, state_repeated)
        # Step 7: Return Losses
        return {"action_loss": action_loss, "memory_loss": las_action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        """
        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if type(examples) is not list:
            examples = [examples]
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        batch_las_images = [example["las_image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        action_keys = [example["action_keys"] for example in examples] # [B, str]
    
        # Step 1: QWenVL input format


        batch_change_map, batch_flow_map, batch_hsv_map = build_change_flow_map(batch_images, batch_las_images)
        # Calculate Memory Map Update Positions based on HSV brightness
        upd_poss = _calculate_update_positions(batch_hsv_map, self.map_x, self.map_y, self.image_size, self.memory_update_topk)
        print("finish data prepare",flush=True)
        # Step 2: QWenVL Inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs_pro(
            images=batch_images, 
            action_keys=action_keys, 
            change_maps=batch_change_map,
            flow_maps=batch_flow_map,
            instructions=instructions
        )
        print("QwenForwarding",flush=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # Step 3: Forward QwenVL
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            print("end of Forwarding",flush=True)
            last_hidden = qwenvl_outputs.hidden_states[-1]
            calc_hidden = last_hidden.mean(dim=1).float()
            memory = self.memory(
                ext=calc_hidden, 
                upd_poss=upd_poss,
                update=False
            ).to(device=last_hidden.device, dtype=torch.float32)
            print("memory",flush=True)
            last_hidden = torch.cat([last_hidden, memory.unsqueeze(1).to(dtype=last_hidden.dtype)], dim=1)
        state = None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(last_hidden, state)  # (B, chunk_len, action_dim)
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}



if __name__ == "__main__":
    from omegaconf import OmegaConf
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()
    # import debugpy  
    # debugpy.listen(("0.0.0.0", 10092))
    # print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    # debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    # cfg.framework.action_model.action_hidden_dim = 2048

    # cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Florence-2-large"
    

    model: Qwen_GR00T = Qwen_GR00T(cfg)
    print(model)



    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 14)).astype(np.float16), # action_chunk, action_dim
        "image": [image], # three views
        "lang": "Put all the toys in the child's room - the three board games (two on the bed and one on the table), the two jigsaw puzzles on the table, and the tennis ball on the table - inside the toy box on the table in the child's room.",
        "state" : np.random.uniform(-1, 1, size=(1, 14)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(examples=[sample]) #, state=[batch[0]["state"]]
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    vla_dataset_cfg = cfg.datasets.vla_data
    from torch.utils.data import DataLoader
    from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
    cfg.datasets.vla_data.include_state = "False"
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,  # For Debug
        collate_fn=collate_fn,
    )
    # 
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        batch
        break

    # try get model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model(batch)

    action = model.predict_action(examples=batch)
    print("Finished")
