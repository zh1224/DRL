#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多实验、多指标（每个指标独立 CSV）日志平滑脚本
=================================================
目录结构示例
├─ dynamic/SAC/max_speed_1.../
│   ├─ reward+...+5.0+.../
│   │   ├─ success_rate.csv
│   │   └─ episode_reward.csv
│   └─ reward+...+6.0+.../
│       ├─ success_rate.csv
│       └─ episode_reward.csv

每个 CSV:
    step,value
    0,0.12
    1,0.15
    2,0.18
    ...

运行:
    python smooth_metrics_separate.py
"""

# ==================== 0. 可调参数 ====================
BASE_DIR   = "./DDPG/max_speed_1_collsion_dist_0.45_k_250_threshold_0.45_1"
SUBFMT     = "reward+0.05+40.0+0.0+{idx}.0+740.0+0.0+0.0+600.0+-0.0003"
IDX_RANGE  = range(54,55)
EXCLUDE    = set()

# 指标名 → 对应 CSV 文件名
SCALAR_FILES = {
    "Success Rate":   "Succcess rate.csv",
    "Episode Reward": "Episode Rward.csv",
    "Actor-Loss":"Actor-Loss.csv",
    "Critic-Loss":"Critic-Loss.csv",
    "episode step": "episode step.csv"
    # 若有别的指标，继续往下加
}

STEP_COL   = "Step"
VAL_COL    = "Value"

SMOOTH_MODE = "ema"          # "ema" / "sma" / None
EMA_ALPHA   = 0.7
SMA_WINDOW  = 200
DS_LIMIT    = 6000
SKIP_THR    = 1e-6

PLOT_SINGLE = False
PLOT_AVG    = True
OUT_DIR     = "./"
PNG_DPI     = 150
# ====================================================

import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs(OUT_DIR, exist_ok=True)

# ==================== 1. 工具函数 ====================
def ema(values, alpha=0.997):
    sm, prev = [], None
    for v in values:
        prev = v if prev is None else alpha * prev + (1 - alpha) * v
        sm.append(prev)
    return sm

def sma(values, window=100):
    if window <= 1:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smooth = (cumsum[window:] - cumsum[:-window]) / window
    return np.concatenate([values[:window-1], smooth])

def downsample(steps, vals, limit=2048):
    if len(steps) <= limit:
        return steps, vals
    idx = np.linspace(0, len(steps)-1, limit, dtype=int)
    return steps[idx], vals[idx]

def smooth_series(steps, vals, alpha_override=None):
    """
    steps, vals : 原始序列
    alpha_override: 仅 EMA 时生效；若为 None 则用全局 EMA_ALPHA
    """
    if SMOOTH_MODE == "ema":
        w = alpha_override if alpha_override is not None else EMA_ALPHA
        vals_sm = ema(vals, w)
    elif SMOOTH_MODE == "sma":
        vals_sm = sma(vals, SMA_WINDOW)
    else:
        vals_sm = vals
    vals_sm = np.asarray(vals_sm)
    return downsample(np.asarray(steps), vals_sm, DS_LIMIT)


def read_scalar_csv(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if STEP_COL not in df.columns or VAL_COL not in df.columns:
        raise ValueError(f"{path} 缺少 '{STEP_COL}' 或 '{VAL_COL}' 列")
    steps = df[STEP_COL].to_numpy(dtype=int)
    vals  = df[VAL_COL].to_numpy(dtype=float)
    mask  = np.abs(vals) > SKIP_THR
    return steps[mask], vals[mask]

def accumulate(sum_dict, tag, series):
    steps, vals = series
    if tag not in sum_dict:
        sum_dict[tag] = {'steps': steps, 'vals': vals}
        return
    # 对齐 step 再相加
    ref_steps = sum_dict[tag]['steps']
    if np.array_equal(ref_steps, steps):
        sum_dict[tag]['vals'] += vals
    else:
        common = np.intersect1d(ref_steps, steps)
        if len(common) == 0:
            print(f"[WARN] tag={tag} 无共同 step，跳过累加")
            return
        idx_a = np.isin(ref_steps, common)
        idx_b = np.isin(steps, common)
        sum_dict[tag]['vals'][idx_a] += vals[idx_b]
        sum_dict[tag]['steps'] = ref_steps[idx_a]

def save_smoothed_csv(tag, steps, vals, fname):
    pd.DataFrame({STEP_COL: steps, VAL_COL: vals}).to_csv(fname, index=False)

# ==================== 2. 主流程 ====================
def main():
    sum_data, run_cnt = defaultdict(dict), 0

    for idx in IDX_RANGE:
        if idx in EXCLUDE:
            continue
        exp_dir = os.path.join(BASE_DIR, SUBFMT.format(idx=idx))
        if not os.path.isdir(exp_dir):
            print(f"[WARN] 不存在: {exp_dir}")
            continue

        scalar_dict = {}
        for tag, fname in SCALAR_FILES.items():
            series = read_scalar_csv(os.path.join(exp_dir, fname))
            if series is not None:
                scalar_dict[tag] = series
            else:
                print(f"[WARN] run-{idx} 缺少 {fname}")

        if not scalar_dict:
            continue
        run_cnt += 1

        # 单条曲线绘制
        if PLOT_SINGLE:
            for tag, (steps, vals) in scalar_dict.items():
                s_steps, s_vals = smooth_series(steps, vals)
                plt.figure(figsize=(6,4))
                plt.plot(s_steps, s_vals, lw=1.2)
                plt.title(f"{tag} | run-{idx}")
                plt.xlabel(STEP_COL); plt.ylabel(tag)
                plt.tight_layout()
                plt.savefig(os.path.join(OUT_DIR, f"{tag}_run{idx}.png"),
                            dpi=PNG_DPI)
                plt.close()
                save_smoothed_csv(tag, s_steps, s_vals,
                                  os.path.join(OUT_DIR, f"{tag}_run{idx}.csv"))

        # 加总
        for tag, series in scalar_dict.items():
            accumulate(sum_data, tag, series)

    if run_cnt == 0:
        print("✗ 没有找到任何有效实验")
        return

    # 求平均 & 保存
    # 求平均 & 保存
    if PLOT_AVG:
        for tag, data in sum_data.items():
            steps = data['steps']
            vals  = data['vals'] / run_cnt

            # ▶ 这里按需指定 α
            if tag.lower().startswith("success"):
                alpha = 0.8
            elif tag=="Episode Reward":
                alpha = 0.994          # 例如 Episode Reward / Episode Step
            elif tag=="episode step":
                alpha = 0.99
            else:
                alpha=0.98

            s_steps, s_vals = smooth_series(steps, vals, alpha_override=alpha)

            plt.figure(figsize=(6,4))
            plt.plot(s_steps, s_vals, lw=1.8)
            plt.title(f"{tag} | {run_cnt} runs avg  (α={alpha})")
            plt.xlabel(STEP_COL); plt.ylabel(tag)
            if tag.lower().startswith("success"):
                plt.ylim([0, 1])
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{tag}_avg.png"), dpi=PNG_DPI)
            plt.close()

            save_smoothed_csv(tag, s_steps, s_vals,
                            os.path.join(OUT_DIR, f"{tag}-DDPG.csv"))

    print(f"√ 完成 {run_cnt} 条实验的平滑与保存，结果在 {OUT_DIR}")

if __name__ == "__main__":
    main()
