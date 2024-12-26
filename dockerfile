# 使用Python官方提供的Python镜像作为基础镜像
FROM python:3.9
# 设置工作目录
WORKDIR /middle_rope
# 将当前目录下的所有文件复制到容器的/app目录中
COPY . /middle_rope
# 安装Python依赖
RUN pip install -r requirements.txt
# 设置容器启动时的默认命令
CMD ["/bin/bash"]