# Antigravity IDE Prompt — FreeGuide Server Experiments

请阅读本项目的实验执行协议文档：

```
experiments/SERVER_EXPERIMENT_GUIDE.md
```

这是一份完整的实验部署和执行指南。

## 你的任务

按照该文档的指示，自主完成以下工作：

1. **部署实验环境**（Part 1）
   - 创建 conda 环境 `freeguide`（Python 3.10）
   - 配置中国镜像源（阿里 pip + 清华 conda）
   - 安装 PyTorch（cu118）、dm_control、mujoco 等所有依赖
   - 安装 TD-MPC2：`cd tdmpc2 && pip install -e .`
   - 验证环境：跑 5000 步 walker-run 确认无报错

2. **按优先级启动 P1 主实验**（Part 2，共 75 runs）
   - 3 个方法（tdmpc2, tdmpc2_rnd, freeguide）× 5 个任务 × 5 seeds
   - 按文档中的 GPU 分配方案和批次顺序启动
   - 每张 GPU 同时并行 4 个实验

3. **P1 跑完后启动 P2 消融实验**（Part 3，共 36 runs）
   - 30 个组件消融 + 6 个 Ensemble K 消融
   - tdmpc2 和 freeguide 的 walker-run/humanoid-run 结果复用 P1

4. **定期检查实验进度**
   - 运行 `bash experiments/scripts/check_progress.sh` 查看进度
   - 运行 `bash experiments/scripts/check_gpu.sh` 查看 GPU 使用情况

## 执行规则

- **所有训练用 `nohup` 后台运行，启动后立即返回，不要等待完成**
- 按照文档中的 GPU 分配方案分配任务（GPU 0/1/2）
- 遇到报错自行根据文档中 Part 5（Fault Handling）的指南修复
- **不要修改 `tdmpc2/` 下的源码**，只执行实验命令
- 如果某个实验 OOM，加 `buffer_size=500000` 重跑
- 如果某个实验 NaN 崩溃，记录到 `logs/failed_experiments.txt` 然后继续
- 每个 batch 用 `wait` 等前一批完成后再启动下一批

## 代码路径

代码应该已经通过 rsync 同步到了服务器。如果代码在非默认路径，请将文档中所有 `/home/miller/FreeGuide/` 替换为实际路径。

训练脚本工作目录：`{项目根目录}/tdmpc2/tdmpc2/`

现在开始执行，从环境部署（Part 1）开始。
