import os, glob, csv, numpy as np
from tensorboard.backend.event_processing import event_accumulator

SCALE_TAG = "Episode Rward"      # 要导出的 scalar 名
LOGDIR    = "./dynamic/DDPG/max_speed_1_collsion_dist_0.45_k_250_threshold_0.45_1/reward+0.05+40.0+0.0+33.0+740.0+0.0+0.0+600.0+-0.0003"              # 替换为单条实验目录
SMOOTH    = 0.6                  # == TensorBoard 滑条
DS_LIMIT  = 2048                 # 前端最大点数
CSV_NAME  = "Episode_Rward-DDPG.csv"

# ------ 1. 读 scalar ------
ev_file = glob.glob(os.path.join(LOGDIR, "events.out.tfevents.*"))[0]
ea = event_accumulator.EventAccumulator(ev_file); ea.Reload()
steps, vals = zip(*[(e.step, e.value) for e in ea.Scalars(SCALE_TAG)])

# ------ 2. 纯观测序列做 EMA (A + B) ------
w = SMOOTH
vals = np.asarray(vals)
ema  = np.empty_like(vals)
ema[0] = vals[0]
for i in range(1, len(vals)):
    ema[i] = w * ema[i-1] + (1-w) * vals[i]

# ------ 3. 按 TensorBoard down-sample (C) ------
xs = np.arange(len(ema))
if len(xs) > DS_LIMIT:
    bins = np.linspace(0, len(xs), DS_LIMIT, dtype=int)
    keep = set()
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]-1
        seg = slice(lo, hi+1)
        keep.update([lo+np.argmin(ema[seg]),
                     lo+np.argmax(ema[seg]),
                     hi])
    keep = np.array(sorted(keep))
    xs, ema = xs[keep], ema[keep]

# ------ 4. 写 CSV ------
with open(CSV_NAME, "w", newline="") as f:
    csv.writer(f).writerows([("step", "smoothed_val"), *zip(xs, ema)])

print(f"✅ 写入 {CSV_NAME}  ( {len(xs)} 点 )")
