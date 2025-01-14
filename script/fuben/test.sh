#!/bin/bash

# 启用调试模式，打印每一行执行的命令
exec > script.log 2>&1  # 将标准输出和错误输出都重定向到日志文件
set -x                 # 开启调试模式

# 示例命令
echo "Start of the script"
ls -l
pwd
date
echo "End of the script"

# 关闭调试模式
set +x"
通过set +x 可以把每一行指令，也就是python脚本的参数也输出到log 中，
docker 如果目前跑的正常就不用动