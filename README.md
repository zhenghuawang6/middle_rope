## 改进地方！

1. **确定一套或者几套 conda 或者 docker 容器的环境**  
   增加一个 `requirement.txt` 或者 `dockerfile`（在常用的机器上部署即可）。推荐使用`docker`.

2. **固定库版本**  
   将 "最新版本的 `transformers` 固定下来，因为 `transformers` 库更新的比较快。

3. **实验脚本输出与管理**  
   把结果对应的 **脚本参数**（或者脚本本身），加上当前的 **commit 版本号**（通过 `git rev-parse --short HEAD` 获取），也在 `log` 里面复制一份，或者干脆就作为文件名。这样之后复现实验时就比较方便。

4. **文件结构建议**  
   - `src` 放 `inference*.py` 文件。在 `utils` 中的文件夹 `modify arch` 可以放进去。  
   - 最外层新开一个 `eval` 文件夹，放几个测试的 `.py` 文件。
   - 如果是从别的 `git` 库里面引用的，可以用 `git module`（可选）。

```
data/
    src/
	modify_arch/
        inference*.py
    script/
        mutilevel*.sh
    eval/
        mutilevel/
            result/
                4d7cf3b-xxx-llama.log
                4d7cf3b-xxx-llama.sh
        qa/
            result/
                4d7cf3b-xxx-llama.log
                4d7cf3b-xxx-llama.sh

```


