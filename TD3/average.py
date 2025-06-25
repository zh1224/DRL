import os, glob, csv, sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from tensorboard.backend.event_processing import event_accumulator

# ======================================================================
# 0. 可调参数
# ======================================================================
SCALARS         = ["Success Rate",  "Episode Rward"]

# -------- 平滑方式 --------
SMOOTH_METHOD   = "ema"     # "ema" 或 "sma"
EMA_ALPHA       = 0.996    # 当 method="ema" 时使用
SMA_WINDOW      = 200       # 当 method="sma" 时使用

# -------- 其它参数 --------
DS_LIMIT        = 2048      # down-sample 上限 (同 TensorBoard)
SKIP_THR        = 1e-3      # 过滤占位 0
PLOT_SINGLE_RUN = False
PLOT_AVG        = True
PNG_DPI         = 150
PREFIX          = "APF"    # 输出文件前缀
# ======================================================================


# ----------------------------------------------------------------------
# 1. 读取单目录 scalars ⇢ {tag: [(step,val), ...]}
# ----------------------------------------------------------------------
def load_scalars_from_logdir(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    ev_files = glob.glob(os.path.join(log_dir, "**", "events.out.tfevents.*"), recursive=True)
    if not ev_files:
        print(f"⚠️  {log_dir} 无 event 文件")
        return {}
    scalar_dict = {}
    for ev in ev_files:
        ea = event_accumulator.EventAccumulator(ev, size_guidance={event_accumulator.SCALARS: 0})
        try:
            ea.Reload()
        except Exception:
            continue
        for tag in ea.Tags().get("scalars", []):
            scalar_dict.setdefault(tag, []).extend((e.step, e.value) for e in ea.Scalars(tag))
    for tag in scalar_dict:
        scalar_dict[tag].sort(key=lambda t: t[0])
    return scalar_dict


# ----------------------------------------------------------------------
# 2. 平滑函数 (EMA 或 SMA)
# ----------------------------------------------------------------------
def smooth_series(arr: np.ndarray) -> np.ndarray:
    if SMOOTH_METHOD == "ema":
        w = EMA_ALPHA
        if w <= 0:
            return arr.copy()
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = w * out[i - 1] + (1 - w) * arr[i]
        return out

    elif SMOOTH_METHOD == "sma":
        win = max(1, min(SMA_WINDOW, len(arr)))
        kernel = np.ones(win) / win
        return np.convolve(arr, kernel, mode="same")

    else:
        raise ValueError(f"未知平滑方法 {SMOOTH_METHOD}")


# ----------------------------------------------------------------------
# 3. TensorBoard down-sample (可选)
# ----------------------------------------------------------------------
def tb_downsample(xs: np.ndarray, ys: np.ndarray, limit: int = 2048):
    n = len(xs)
    if n <= limit:
        return xs, ys
    bins = np.linspace(0, n, limit, dtype=int)
    keep_idx = set()
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1] - 1
        if lo >= hi:
            keep_idx.add(lo); continue
        seg = slice(lo, hi + 1)
        keep_idx.update((lo + int(np.argmin(ys[seg])),
                         lo + int(np.argmax(ys[seg])),
                         hi))
    keep = np.array(sorted(keep_idx), dtype=int)
    return xs[keep], ys[keep]


# ----------------------------------------------------------------------
# 4. 累加（先平滑，再 down-sample，再 sum/cnt）
# ----------------------------------------------------------------------
def accumulate_scalar(sum_dict: dict, name: str, data: List[Tuple[int, float]]):
    if not data:
        return
    steps, vals = zip(*data)
    steps = np.asarray(steps, dtype=int)
    vals  = np.asarray(vals,  dtype=float)

    # 补全空缺 step
    full = np.full(steps[-1] + 1, np.nan, dtype=float)
    full[steps] = vals
    isnan = np.isnan(full)
    if isnan.any():
        idx_valid = np.where(~isnan)[0]
        full[isnan] = np.interp(np.where(isnan)[0], idx_valid, full[idx_valid])

    # 平滑
    full = smooth_series(full)
    # down-sample
    xs, ys = tb_downsample(np.arange(len(full)), full, DS_LIMIT)

    for st, v in zip(xs, ys):
        if abs(v) < SKIP_THR:
            continue
        d = sum_dict.setdefault(name, {}).setdefault(int(st), {'sum': 0.0, 'cnt': 0})
        d['sum'] += float(v)
        d['cnt'] += 1


# ----------------------------------------------------------------------
# 5. CSV 保存
# ----------------------------------------------------------------------
# 5. CSV 保存
def save_csv(sum_dict: dict, name: str, max_step: int = 5000):
    if name not in sum_dict:
        return
    stats = sum_dict[name]
    # ✅ 加入截断逻辑
    steps = sorted(s for s in stats if s <= max_step)
    avg   = [stats[s]['sum'] / stats[s]['cnt'] for s in steps]
    fname = f"{name.replace(' ','_').replace('/','_')}-{PREFIX}.csv"
    with open(fname, "w", newline="") as f:
        csv.writer(f).writerows([("step", "average_val"), *zip(steps, avg)])
    print(f"✅ 写入 {fname} ({len(steps)} 行)")
    return steps, avg



# ----------------------------------------------------------------------
# 6. 绘图
# ----------------------------------------------------------------------
def plot_curve(xs, ys, title, fname,
               x_major=500, y_major=10,
               y_lim=None,
               figsize=(10,3), dpi=PNG_DPI):
    if len(xs) == 0:
        print(f"[plot_curve] 空序列，跳过 {fname}")
        return
    os.makedirs(os.path.dirname(fname) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xs, ys, label=title, linewidth=1.5)
    ax.set_xlim(xs[0], xs[-1])
    if y_lim:
        ax.set_ylim(*y_lim)
    ax.xaxis.set_major_locator(MultipleLocator(x_major))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(y_major))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which='major', linestyle='--', alpha=.4)
    ax.grid(which='minor', linestyle=':',  alpha=.2)
    ax.set_xlabel("Steps"); ax.set_ylabel(title)
    ax.legend(); plt.tight_layout()
    plt.savefig(fname, dpi=dpi); plt.close()
    print(f"📈 保存 {fname}")


# ----------------------------------------------------------------------
# 7. 主流程
# ----------------------------------------------------------------------
if __name__ == "__main__":

    # ======== 根据实际情况修改日志路径 ==============
    BASE   = "./dynamic/max_speed_1_collsion_dist_0.45_k_250_threshold_0.45_1"
    SUBFMT = "reward+0.05+40.0+180.0+{idx}.0+740.0+0.0+0.0+600.0+-0.0003"
    IDX_RANGE = range(37, 38)
    EXCLUDE   = set()              # e.g. {1,4,7}
    # ===============================================

    sum_data, valid_runs = {}, 0
    for idx in IDX_RANGE:
        if idx in EXCLUDE:
            continue
        logdir = os.path.join(BASE, SUBFMT.format(idx=idx))
        scalars = load_scalars_from_logdir(logdir)
        if not scalars:
            continue
        valid_runs += 1
        for tag in SCALARS:
            if tag in scalars:
                accumulate_scalar(sum_data, tag, scalars[tag])
                if PLOT_SINGLE_RUN:
                    xs_raw, ys_raw = zip(*scalars[tag])
                    ys_smooth = smooth_series(np.asarray(ys_raw))
                    xs_ds, ys_ds = tb_downsample(np.asarray(xs_raw), ys_smooth, DS_LIMIT)
                    plot_curve(xs_ds, ys_ds,
                               title=f"{tag}-{idx}",
                               fname=f"{tag.replace(' ','_')}-{idx}.png",
                               y_major=10 if "Episode Rward" in tag else 0.2,
                               y_lim=(-45, 0) if "Episode Rward" in tag else None)
            else:
                print(f"run {idx}: 无 scalar '{tag}'")

    print(f"共汇总 {valid_runs} 条实验")

    for tag in SCALARS:
        res = save_csv(sum_data, tag, max_step=4500)

        if PLOT_AVG and res:
            steps, avg = res
            plot_curve(steps, avg,
                       title=f"{tag}-{PREFIX}-AVG",
                       fname=f"{tag.replace(' ','_')}-{PREFIX}-AVG.png",
                       y_major=10 if "Episode Rward" in tag else 0.2,
                       y_lim=(-45, 0) if "Episode Rward" in tag else None)