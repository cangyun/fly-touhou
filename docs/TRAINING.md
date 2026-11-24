# 训练与局部微调 / Training and attack-local fitting

## 启动与保存

```powershell
fly-touhou train --game th08 --difficulty lunatic --objective nbnm --train-scope encoder-heads --game-exe "D:\Games\TH08\th08.exe" --workers 8 --dry-run
fly-touhou train --game th14 --difficulty lunatic --objective clear --train-scope joint --game-exe "D:\Games\TH14\th14.exe" --workers 8 --dry-run
fly-touhou console
```

`--trainable-head` 可重复指定输出组，例如 `movement` 和 `risk`。感觉编码仍沿脑网络获得梯度。作业目录保存命令、状态和日志，训练输出保存参数与 Adam。

`--resume` 读取同训练契约的 optimizer checkpoint。NBNM 默认使用对应的输出恢复，TH14 有限残机训练从新输出继续。切换原作、目标、训练范围或输入版本时，用 `--checkpoint` 显式选择兼容的模型参数并启动新优化器。新 checkpoint 记录原作、难度、目标、观测、动作、连接组和训练范围。

## TH08 符卡专项

```powershell
fly-touhou fit-attack --game th08 --difficulty lunatic --objective nbnm --guide results/attack-guides/th08-134 --dry-run
```

`--guide` 包含原作无伤完成此攻击的 `report.json` 及配套 `matched-history.npz`。脚本用固定脑重放感觉数据并优化该卡的动作读出。观测到的攻击编号和最小攻击年龄与权重一起保存。[99 号符卡数据](../results/attack-guides/th08-99) · [134 号符卡数据](../results/attack-guides/th08-134) · [原作续打验证工具](../scripts/evaluate_th08_prefix_window.py)

## English

The commands above preview the selected backend and budget. A matching `--resume` restores Adam. Use an explicit compatible `--checkpoint` when initializing a new optimizer. TH08 attack-local fitting uses a verified original-game certificate and matching sensory and action rows.
