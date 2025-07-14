#!/bin/bash

# 初始化参数值
param1=0.05
param2=40
param3=0
param4=89
param5=740
timeout_duration=18000
param6=0
param7=0
param8=600 # 500
param9=-0.0003

#param11=8
num_executions=50
for ((i=0;i<num_executions;i++)); do
    echo ">>> episode $i"

    # 把 python 放进新进程组；超时后连子进程一起杀
    timeout -k 10s $timeout_duration bash -c \
        "exec setsid python3 APF_Liquidity_Backstop.py $i"

    # 无论是否正常结束，都执行清理
    ./shutdown-test.sh
    sleep 10

done