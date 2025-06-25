#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
comparison.py
=============

从多组 CSV 读取训练曲线，在 step ≤ 4500 范围内绘制对比图（无平滑）。

假设每个 CSV 至少包含两列：
    Step,Value

文件命名格式：
    <指标>-<算法>.csv
例如：
    Success Rate-TD3.csv
    Episode Reward-TD3.csv
"""

import os
import csv
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
plt.rcParams["lines.solid_joinstyle"] = "round"
plt.rcParams["lines.solid_capstyle"]  = "round"

# ----------- 可调常量 -----------
STEP_LIMIT = 4000          # 仅保留 step ≤ STEP_LIMIT
OUT_DIR    = "./"          # 图片输出目录

# CSV 文件名中的算法与指标
ALGOS   = ["TD3","APF","SAC","DDPG"]          # 如果还有 DDPG / SAC / APF，就加到列表里
METRICS = ["Success Rate", "Episode Reward","Actor-Loss","Critic-Loss","episode step"]

# CSV 的列名（区分大小写）
STEP_COL = "Step"
VAL_COL  = "Value"
# --------------------------------

os.makedirs(OUT_DIR, exist_ok=True)


# -------- 读取单个 CSV --------
def read_csv(path: str) -> Tuple[List[float], List[float]]:
    """
    返回升序排列的 (steps, values)。
    若文件不存在或列名不匹配则返回空列表。
    """
    if not os.path.isfile(path):
        print(f"⚠️  {path} 不存在，跳过")
        return [], []

    cache = {}
    try:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                step = float(row[STEP_COL])
                val  = float(row[VAL_COL])
                cache[step] = val          # 若同一 step 写多次，仅保留最后一次
    except Exception as e:
        print(f"❌ 读取 {path} 出错: {e}")
        return [], []

    # 移除占位行（可选）
    if 0.0 in cache and cache[0.0] == 0.0:
        cache.pop(0.0)

    steps_sorted = sorted(cache.keys())
    vals_sorted  = [cache[s] for s in steps_sorted]
    return steps_sorted, vals_sorted
# --------------------------------


def main():
    for metric in METRICS:
        plt.figure(figsize=(8, 3))
        any_curve = False  # 用于判断该指标是否至少画出一条曲线

        for algo in ALGOS:
            csv_file = f"{metric}-{algo}.csv"
            xs, ys = read_csv(csv_file)
            if not xs:
                continue

            # 截断到 STEP_LIMIT
            xs_cut = [s for s in xs if s <= STEP_LIMIT]
            if not xs_cut:
                print(f"⚠️  {csv_file} 无 step ≤ {STEP_LIMIT} 数据，跳过")
                continue
            ys_cut = ys[:len(xs_cut)]

            plt.plot(xs_cut,
                 ys_cut,
                 label=algo,
                 linewidth=1.6,
                 solid_joinstyle="round",   # ←↙ 这两行决定圆 or 尖
                 solid_capstyle="round")    # ←
            any_curve = True

        if not any_curve:
            print(f"⚠️  {metric} 没有任何可绘制曲线，跳过生成图片")
            plt.close()
            continue

        # -------- 画布格式 --------
        ax = plt.gca()
        ax.set_xlim(0, STEP_LIMIT)
        ax.xaxis.set_major_locator(tk.MultipleLocator(500))
        ax.grid(True)

        plt.xlabel("Steps")
        plt.ylabel(metric)
        plt.legend()
        if metric.lower().startswith("success"):
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_locator(tk.MultipleLocator(0.1))
        elif metric=="episode step":
            # 可按需要固定 y 轴范围，例如：
            # ax.set_ylim(-50, 0)
            ax.yaxis.set_major_locator(tk.MultipleLocator(20))
        elif metric=="Episode Reward":
            ax.yaxis.set_major_locator(tk.MultipleLocator(10))
        out_png = os.path.join(OUT_DIR, f"{metric}_compare.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=500)
        plt.close()
        print(f"✅ 已保存 {out_png}")

    print("🎉 所有指标绘制完成！")


if __name__ == "__main__":
    main()
