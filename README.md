# LEPPO
算法

## 仿真与调试

完整模型、默认参数、安装步骤及实现假设见 [README_SIMULATION.md](README_SIMULATION.md)，实测检查记录见 [VALIDATION.md](VALIDATION.md)。

常用修改位置：

- `configs/default.json`：AP/用户数、任务量、功率、电池和奖励权重。
- `leppo/environment.py`：信道、采能、时延、能耗、约束和奖励计算。
- `leppo/policy.py`：GRU、Actor、Critic 和 PPO 更新。
- `leppo/experiment.py`：训练、评估和绘图流程。

安装依赖后运行：

```bash
python -m unittest discover -s tests -v
python -m leppo.experiment --steps 512 --eval-slots 100 --output results/debug
```

修改参数后使用新的结果目录；AP/用户数等变化会改变网络维度，需要重新训练。训练结果和虚拟环境不提交到 Git，运行后会在本地生成。
