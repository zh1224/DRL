import random
from collections import deque
import numpy as np


class ReplayBuffer(object):
    """
    双缓冲 (成功 / 失败) + 时间队列 order_buf：
    - 保证总样本数 ≤ buffer_size
    - add() O(1)；sample_batch() O(k)
    - 调用接口与旧版一致
    """

    def __init__(self, buffer_size, random_seed=12):
        self.buffer_size = int(buffer_size)
        self.succ_buf = deque()          # 专门存成功经验
        self.fail_buf = deque()          # 专门存失败经验
        self.order_buf = deque()         # 记录插入顺序：True=succ, False=fail
        self.count = 0
        random.seed(random_seed)

    # ---------- 写入 ----------
    def add(self, s, a, r, t, s2, U_s, omega):
        # ★ 成功判定逻辑：按需修改此行
        is_success = (t == 1) and (r > 0)

        exp = (s, a, r, t, s2, U_s, omega)

        # 若总容量已满 => 先弹出最旧经验（依据 order_buf）
        if self.count >= self.buffer_size:
            oldest_is_succ = self.order_buf.popleft()
            (self.succ_buf if oldest_is_succ else self.fail_buf).popleft()
            self.count -= 1

        # 追加到对应 deque，并在 order_buf 记录类别
        if is_success:
            self.succ_buf.append(exp)
        else:
            self.fail_buf.append(exp)

        self.order_buf.append(is_success)
        self.count += 1

    # ---------- 读取 ----------
    def sample_batch(self, batch_size):
        if self.count == 0:
            raise ValueError("ReplayBuffer is empty")

        # 目标比例：成功 30 %，失败 70 %
        k_succ = int(batch_size * 0.6)
        k_fail = batch_size - k_succ

        # 随机抽样（若不足则抽可用最大量）
        succ_sample = random.sample(self.succ_buf, min(k_succ, len(self.succ_buf))) if k_succ else []
        fail_sample = random.sample(self.fail_buf, min(k_fail, len(self.fail_buf))) if k_fail else []

        # 类别不足时用另一类补齐
        shortfall = batch_size - (len(succ_sample) + len(fail_sample))
        if shortfall > 0:
            pool = self.succ_buf if len(self.succ_buf) > len(self.fail_buf) else self.fail_buf
            extra = random.sample(pool, min(shortfall, len(pool)))
            (succ_sample if pool is self.succ_buf else fail_sample).extend(extra)

        batch = succ_sample + fail_sample
        random.shuffle(batch)            # 打乱成功/失败顺序

        # 拆分张量
        s_batch   = np.array([e[0] for e in batch])
        a_batch   = np.array([e[1] for e in batch])
        r_batch   = np.array([e[2] for e in batch]).reshape(-1, 1)
        t_batch   = np.array([e[3] for e in batch]).reshape(-1, 1)
        s2_batch  = np.array([e[4] for e in batch])
        U_s_batch = np.array([e[5] for e in batch])
        omega_batch = np.array([e[6] for e in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch, U_s_batch, omega_batch

    # ---------- 其他接口 ----------
    def size(self):
        return self.count

    def clear(self):
        self.succ_buf.clear()
        self.fail_buf.clear()
        self.order_buf.clear()
        self.count = 0
