import os
import re

# 文件名匹配
file_pattern = re.compile(r"eval_steps_(\d+)_libero_(.+)\.log")

# success rate匹配
import re
rate_pattern = re.compile(
    r"Total success rate:\s*([0-9.]+)\s*eval_libero.py:245\s*INFO\s*| >> Total episodes:\s*500",
    re.DOTALL
)

ckpt_pattern = re.compile(
    r"CKPT:\s*/.*?libero4in1_Qwen3GR00TD_H800_(.+?)/checkpoints"
)

results = []

for root, dirs, files in os.walk("."):
    for fname in files:
        m = file_pattern.match(fname)
        if not m:
            continue

        x = int(m.group(1))
        y = m.group(2)
        # print(x,y)

        path = os.path.join(root, fname)
        with open(path, "r") as f:
            text = f.read()

        ckpt_match = ckpt_pattern.search(text)
        if not ckpt_match:
            continue
        q = ckpt_match.group(1)
        
        m = rate_pattern.search(text)
        if m:
            z = float(m.group(1))
            results.append((q,x,y,z))

    # 输出结果
for q, x, y, z in sorted(results):
    print((q, x, y, z))