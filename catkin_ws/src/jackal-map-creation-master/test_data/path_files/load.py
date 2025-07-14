# -*- coding: utf-8 -*-
import numpy as np

# 读取 npy 文件
file_path = 'path_0.npy'  # 替换为实际的文件路径
data = np.load(file_path)

# 打印读取的数据
print("数据内容：", data)
print("数据类型：", type(data))
print("数据形状：", data.shape)
