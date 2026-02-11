from typing import List, Tuple, Union, Any

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

def _to_uint8_hwc3(img: Any) -> np.ndarray:
    """
    Convert various image formats into HxWx3 uint8.
    Accepts:
    - np.ndarray (H,W,3) uint8/float
    - torch.Tensor (C,H,W) or (H,W,C) etc. (if torch is installed; handled by duck-typing)
    """
    # torch tensor -> numpy
    if hasattr(img, "detach") and hasattr(img, "cpu") and hasattr(img, "numpy"):
        img = img.detach().cpu().numpy()

    img = np.asarray(img)

    # If CHW, convert to HWC
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    # If grayscale, expand to 3 channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4 or HxW, got shape={img.shape}")

    # Drop alpha if exists
    if img.shape[2] == 4:
        img = img[:, :, :3]

    # Normalize to uint8
    if img.dtype != np.uint8:
        # If in [0,1] float, scale; otherwise clip directly
        if np.issubdtype(img.dtype, np.floating):
            if img.max() <= 1.5:
                img = (img * 255.0).round()
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img


def _compute_change_map(
    curr_bgr: np.ndarray,
    prev_bgr: np.ndarray,
    top_ratio: float = 0.90,
    blur_ksize: int = 5,
    morph_open: bool = True,
    morph_ksize: int = 3,
) -> np.ndarray:
    """
    Returns: change_map (H,W) uint8 in {0,255}
    """
    # grayscale diff
    curr_g = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    prev_g = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(curr_g, prev_g).astype(np.float32)

    # smooth
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    if k >= 3:
        diff = cv2.GaussianBlur(diff, (k, k), 0)
    flat = diff.reshape(-1)
    thr = np.max(flat) * (1.0 - top_ratio)
    mask = (diff > thr).astype(np.uint8) * 255
    if 'thr' in locals():
        num_over = np.sum(diff > thr)
        ratio = num_over / diff.size
        if ratio < 0.01:
            return mask * 0.0
    if morph_open:
        mk = morph_ksize if morph_ksize % 2 == 1 else morph_ksize + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return mask


def _compute_flow_map(
    curr_bgr: np.ndarray,
    prev_bgr: np.ndarray,
    fb_pyr_scale: float = 0.5,
    fb_levels: int = 3,
    fb_winsize: int = 15,
    fb_iterations: int = 3,
    fb_poly_n: int = 5,
    fb_poly_sigma: float = 1.2,
    mag_threshold: float = 0.5,
) -> (np.ndarray, np.ndarray):
    """
    计算并可视化 Farneback 稠密光流。
    返回:
        flow_bgr: 用于显示的 BGR 图像
        hsv: 原始 HSV 格式数据
    """
    # 1. 转换为灰度图（确保输入是连续内存）
    prev_g = cv2.cvtColor(np.ascontiguousarray(prev_bgr), cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(np.ascontiguousarray(curr_bgr), cv2.COLOR_BGR2GRAY)

    # 2. 计算光流 (H, W, 2)
    flow = cv2.calcOpticalFlowFarneback(
        prev_g, curr_g, None,
        pyr_scale=fb_pyr_scale,
        levels=fb_levels,
        winsize=fb_winsize,
        iterations=fb_iterations,
        poly_n=fb_poly_n,
        poly_sigma=fb_poly_sigma,
        flags=0
    )

    # 3. 笛卡尔坐标转极坐标
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)

    # 4. 构建 HSV 可视化矩阵
    # Hue (色调): 代表运动方向，0-180 对应 OpenCV 的 0-360度
    # Saturation (饱和度): 恒定 255
    # Value (亮度): 代表运动幅度
    h, w = curr_bgr.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 填充 Hue: OpenCV 中 Hue 范围是 [0, 179]
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255

    # 5. 幅度归一化处理（处理噪声与离群点）
    # 过滤掉极小的噪声，防止静止区域出现彩色斑点
    mag[mag < mag_threshold] = 0
    
    m_flat = mag.reshape(-1)
    if m_flat.size > 0:
        # 使用 95% 分位数作为亮度上限，比 99% 更平滑，能更好地观察主体运动
        m_ref = np.percentile(m_flat, 95.0)
        # 兜底：如果整幅图几乎没动，m_ref 会很小，这里限制最小参考值为 2.0 像素
        m_ref = max(m_ref, 2.0)
        
        val = np.clip(mag / m_ref, 0.0, 1.0) * 255.0
        hsv[..., 2] = val.astype(np.uint8)
    else:
        hsv[..., 2] = 0

    # 6. 转回 BGR 用于显示
    flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_bgr, hsv


def build_change_flow_map(
    batch_images: List[List[Any]],        # [B, [P,L,T]] each view is image-like
    batch_las_images: List[List[Any]],    # [B, [P,L,T]] previous views
    top_ratio: float = 0.90,
    blur_ksize: int = 3,
    morph_open: bool = True,
):
    """
    Args:
    batch_images: current views, [B, V]
    batch_las_images: previous views, [B, V] aligned with batch_images
    Returns:
    batch_change_map: [B, V] each is (H,W) uint8 mask {0,255}
    batch_flow_map:   [B, V] each is (H,W,3) uint8 BGR flow visualization
    """
    if len(batch_images) != len(batch_las_images):
        raise ValueError(f"B mismatch: {len(batch_images)} vs {len(batch_las_images)}")

    batch_change_map: List[List[np.ndarray]] = []
    batch_flow_map: List[List[np.ndarray]] = []
    batch_hsv_map: List[List[np.ndarray]] = []

    for b in range(len(batch_images)):
        views_curr = batch_images[b]
        views_prev = batch_las_images[b]
        if len(views_curr) != len(views_prev):
            raise ValueError(f"View count mismatch at b={b}: {len(views_curr)} vs {len(views_prev)}")

        cmaps_b: List[np.ndarray] = []
        fmaps_b: List[np.ndarray] = []
        vmaps_b: List[np.ndarray] = []

        for v in range(len(views_curr)):
            curr = _to_uint8_hwc3(views_curr[v])
            prev = _to_uint8_hwc3(views_prev[v])

            # Important: Farnebäck assumes same resolution
            if curr.shape[:2] != prev.shape[:2]:
                prev = cv2.resize(prev, (curr.shape[1], curr.shape[0]), interpolation=cv2.INTER_LINEAR)

            # OpenCV default uses BGR; if your arrays are RGB, you can swap here:
            # curr = curr[..., ::-1]
            # prev = prev[..., ::-1]

            change = _compute_change_map(curr, prev)
            change_map = np.repeat(change[:, :, None], 3, axis=2).astype(np.uint8)
            flow_map, hsv_map = _compute_flow_map(curr, prev)

            cmaps_b.append(change_map)
            fmaps_b.append(flow_map)
            vmaps_b.append(hsv_map)

        batch_change_map.append(cmaps_b)
        batch_flow_map.append(fmaps_b)
        batch_hsv_map.append(vmaps_b)

    return batch_change_map, batch_flow_map, batch_hsv_map

def _calculate_update_positions(batch_hsv_map, map_x, map_y, image_size, memory_update_topk):
    upd_poss = []
    for _ in range(len(batch_hsv_map)):
        upd_pos = []
        hsv_map = batch_hsv_map[_][0][:, :, 2]
        for i in range(map_x):
            for j in range(map_y):
                x0 = i * image_size[0] // map_x
                y0 = j * image_size[1] // map_y
                x1 = (i + 1) * image_size[0] // map_x
                y1 = (j + 1) * image_size[1] // map_y
                avg_brightness = np.mean(hsv_map[y0:y1, x0:x1])
                upd_pos.append([i, j, avg_brightness])
        upd_pos = sorted(upd_pos, key=lambda x: x[2], reverse=True)[:min(len(upd_pos), memory_update_topk)]
        upd_pos = [[pos[0], pos[1]] for pos in upd_pos]
        upd_poss.append(upd_pos)
    return upd_poss


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

def _to_uint8_img(x):
    # x: torch.Tensor / np.ndarray / PIL.Image / list
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().float().numpy()
    except Exception:
        pass

    # PIL
    try:
        from PIL import Image
        if isinstance(x, Image.Image):
            return x
    except Exception:
        pass

    x = np.array(x)

    # shape normalize: (C,H,W) -> (H,W,C)
    if x.ndim == 3 and x.shape[0] in (1, 3, 4) and x.shape[0] < x.shape[-1]:
        x = np.transpose(x, (1, 2, 0))

    # if single channel, squeeze to (H,W)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]

    # normalize to uint8
    if x.dtype != np.uint8:
        x = x.astype(np.float32)
        # common cases: [0,1] or [0,255] or arbitrary
        if np.nanmax(x) <= 1.0 + 1e-6:
            x = x * 255.0
        else:
            # rescale robustly if needed
            mn, mx = np.nanmin(x), np.nanmax(x)
            if mx > mn:
                x = (x - mn) / (mx - mn) * 255.0
        x = np.clip(x, 0, 255).astype(np.uint8)

    from PIL import Image
    return Image.fromarray(x)