def auto_get_module_keys(module, max_depth=0, prefix_list=None, current_depth=0, current_prefix=""):
    """
    get all submodule keys of a module, support setting recursion depth and prefix list.

    :param module: the module to traverse.
    :param max_depth: the maximum recursion depth, default is 1.
    :param prefix_list: only include modules with specified prefix, default is None means no restriction.
    :param current_depth: the current recursion depth, internal use.
    :param current_prefix: the current prefix, internal use.
    :return: the list of module keys.
    """
    if current_depth > max_depth:
        return []

    module_keys = []
    for name, sub_module in module.named_children():
        full_name = f"{current_prefix}.{name}" if current_prefix else name
        if prefix_list is None or any(full_name.startswith(prefix) for prefix in prefix_list):
            module_keys.append(full_name)
        module_keys.extend(auto_get_module_keys(sub_module, max_depth, prefix_list, current_depth + 1, full_name))
    return module_keys


def is_module_trainable(module):
    """
    check if a module is trainable: if the module itself has parameters, then all its parameters require_grad must be True;
    if the module itself has no parameters, then its trainability depends on its submodules.
    """
    params = list(module.parameters(recurse=False))
    if params:
        return all(p.requires_grad for p in params)
    else:
        # for container modules with no direct parameters, consider them trainable (the final result depends on their submodules)
        return True


def auto_get_trainable_modules(module, prefix="", max_depth=None):
    """
    recursively traverse the module, return the list of all trainable module names.
    if all submodules of a module are trainable, then only return the name of the parent module, no longer recursively output the names of its submodules.

    parameters:
      - module: the module to traverse.
      - prefix: the name prefix of the current module (internal use).
      - max_depth: the maximum recursion depth, None means infinite recursion.

    return:
      a list of module names.
    """
    # get all direct submodules of the current module
    children = list(module.named_children())

    # if the maximum depth is reached or there are no submodules, return the current module (if trainable and prefix is not empty)
    if (max_depth is not None and max_depth <= 0) or not children:
        return [prefix] if prefix and is_module_trainable(module) else []

    child_keys = []
    all_children_trainable = True
    for name, child in children:
        full_name = f"{prefix}.{name}" if prefix else name
        # recursively get the trainable keys of the submodules
        keys = auto_get_trainable_modules(child, full_name, None if max_depth is None else max_depth - 1)
        if not keys:
            # if the submodule does not return any further submodules, check the submodule itself
            if is_module_trainable(child):
                keys = [full_name]
            else:
                all_children_trainable = False
        else:
            # if the submodule returns multiple names, it means that it cannot be merged
            if len(keys) > 1:
                all_children_trainable = False
        child_keys.extend(keys)

    # if the current module is trainable and all submodules are trainable, return the name of the current module
    if is_module_trainable(module) and all_children_trainable and child_keys:
        return [prefix] if prefix else child_keys
    else:
        return child_keys


def print_freeze_status(self):
    """
    for each top-level submodule, if all its parameters are in the same state (all frozen or all trainable), only print the top-level module.
    if some top-level submodule has mixed parameter states (some frozen, some trainable), list the state of each parameter under the submodule.
    """
    from collections import defaultdict

    # collect the state of parameters under each top-level module
    status_dict = defaultdict(lambda: {"Frozen": 0, "Trainable": 0, "params": []})
    for full_name, param in self.named_parameters():
        # full_name is like "qwen_vl_interface.model.layer.weight"
        top_module = full_name.split(".", 1)[0]  # get the top-level module name
        state = "Frozen" if not param.requires_grad else "Trainable"
        status_dict[top_module]["params"].append((full_name, state))
        status_dict[top_module][state] += 1

    print("=== module parameter freezing status ===")
    for top_module, info in status_dict.items():
        frozen_count = info["Frozen"]
        trainable_count = info["Trainable"]

        if frozen_count > 0 and trainable_count == 0:
            # all frozen
            print(f"{top_module:40s}  |  all Frozen ({frozen_count} parameters)")
        elif trainable_count > 0 and frozen_count == 0:
            # all trainable
            print(f"{top_module:40s}  |  all Trainable ({trainable_count} parameters)")
        else:
            # mixed state, first print the module name summary, then list the state of each parameter
            print(f"{top_module:40s}  |  mixed state → Frozen: {frozen_count}, Trainable: {trainable_count}")
            for pname, pstate in info["params"]:
                print(f"    {pname:60s}  |  {pstate}")
    print("=========================\n")



class Registry:
    def __init__(self, name: str):
        self.name = name
        self._registry = {}

    def register(self, key: str):
        """Decorator: register a builder function or class"""
        def decorator(framework_class):
            if key in self._registry:
                # print(ImportWarning(f"{key} already registered to {self.name}"))
                pass
            self._registry[key] = framework_class
            return framework_class
        return decorator
    
    def __getitem__(self, key):
        return self._registry[key]
    
    def list(self):
        """
        List currently registered keys; if with_values=True (not used here) return mapping {key: value_obj}.
        Using class name as value is also intuitive, e.g., framework.__name__.
        """
        return {k: v for k, v in self._registry.items()}

FRAMEWORK_REGISTRY = Registry("frameworks")
import numpy as np
import cv2
from PIL import Image


def remove_small_components(mask, min_area=30):
    """
    去除小连通块（噪点）
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):  # 0 是背景
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    return clean


def make_change_heatmap(
    curr_img: np.ndarray,    # H,W,3 uint8
    prev_img: np.ndarray,    # H,W,3 uint8
    max_ratio: float = 0.9,  # 使用 max 的 90%
    min_area: int = 40,      # 最小连通区域
    diamond_radius: int = 3, # 扩散半径
):
    """
    生成变化热力图（轮廓 + 扩散）
    返回：RGB heatmap（uint8）
    """

    # ---------------------------
    # 1. 计算差分（灰度）
    # ---------------------------
    curr = curr_img.astype(np.int16)
    prev = prev_img.astype(np.int16)
    diff = np.abs(curr - prev).mean(axis=2).astype(np.float32)

    # ---------------------------
    # 2. 平滑（去噪）
    # ---------------------------
    diff = cv2.GaussianBlur(diff, (3, 3), 0)

    # ---------------------------
    # 3. 阈值（max 的 90%）
    # ---------------------------
    p = np.percentile(diff, 98)   # top 2%
    thr = 0.9 * p
    if max_val < 1e-6:
        return np.zeros_like(curr_img)

    thr = max_ratio * max_val
    seed = (diff >= thr).astype(np.uint8)

    # ---------------------------
    # 4. 去孤立噪点
    # ---------------------------
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, kernel, iterations=1)
    seed = remove_small_components(seed, min_area=min_area)

    if seed.sum() == 0:
        return np.zeros_like(curr_img)

    # ---------------------------
    # 5. 菱形扩散（视觉扩散模拟）
    # ---------------------------
    r = diamond_radius
    diamond = np.zeros((2*r+1, 2*r+1), dtype=np.uint8)
    for i in range(2*r+1):
        for j in range(2*r+1):
            if abs(i - r) + abs(j - r) <= r:
                diamond[i, j] = 1

    expanded = cv2.dilate(seed, diamond, iterations=1)

    # ---------------------------
    # 6. 提取边缘（轮廓）
    # ---------------------------
    eroded = cv2.erode(expanded, np.ones((3,3), np.uint8), iterations=1)
    edge = expanded - eroded

    # ---------------------------
    # 7. 生成热力图（RGB）
    # ---------------------------
    heatmap = np.zeros_like(curr_img, dtype=np.uint8)

    # 填充区域（暗红）
    heatmap[expanded.astype(bool), 2] = 160

    # 边缘（亮红）
    heatmap[edge.astype(bool), 2] = 255
    heatmap[edge.astype(bool), 1] = 80

    return heatmap




from starVLA.training.trainer_utils import initialize_overwatch
import os
import json
from pathlib import Path
from omegaconf import OmegaConf

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def read_mode_config(pretrained_checkpoint):
    """
    Same as read_model_config (legacy duplicate kept for backward compatibility).

    Args:
        pretrained_checkpoint: Path to a .pt checkpoint file.

    Returns:
        tuple:
            vla_cfg (dict)
            norm_stats (dict)
    """
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt`
        assert checkpoint_pt.suffix == ".pt"
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_yaml, dataset_statistics_json = run_dir / "config.yaml", run_dir / "dataset_statistics.json"
        assert config_yaml.exists(), f"Missing `config.yaml` for `{run_dir}`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir}`"

        # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        try:
            ocfg = OmegaConf.load(str(config_yaml))
            global_cfg = OmegaConf.to_container(ocfg, resolve=True)
        except Exception as e:
            overwatch.error(f"❌ Failed to load YAML config `{config_yaml}`: {e}")
            raise

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return global_cfg, norm_stats

