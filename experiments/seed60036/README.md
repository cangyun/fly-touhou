# 第 73 批冻结模型 · 种子 60036

原作《东方辉针城》1.00b，Lunatic，灵梦 A。原通关版与静止加分 0.25 组共用同一份冻结权重，均从第一面开始，六面全程 0 Miss、0 Bomb。开局种子属于开发集。这份对照可以回答这个种子上的动作变化，跨种子成功率需要另行评估。[视频、RPY、权重和决策数据](https://github.com/cangyun/fly-touhou/releases/tag/v0.3.0-nbnm-seed60036)

| 结果 | 原通关版 | 静止加分 0.25 |
|---|---:|---:|
| 模型决策数 | 30435 | 30110 |
| 执行动作后的第六面结算帧 | 33273 | 32306 |
| 分数 | 402,494,300 | 431,515,330 |
| 安全时选择原地 | 188 / 1295（14.5%） | 791 / 1579（50.1%） |
| 连续两次移动时换向 | 16446 / 21170（77.7%） | 15678 / 20427（76.8%） |

第一次动作分歧发生在零起算的第 643 次决策，第一面原作帧 2578，正片 00:42.87。两局此前的观测和动作相同。此刻原版向右，0.25 组选原地。此后两局分别按自己的动作推进，各自的安全场景数也随之变化。同种子的另两组均在第四面发生首次碰撞，分别是纯静止加分 0.5（第 10288 帧）和静止 0.5 加保持方向 0.25（第 4516 帧）。[逐面原始计数与运行配置](results.json)

## 在原作里看

下载任一 RPY，复制到自己的 TH14 `replay` 目录，使用空闲编号，如 `th14_03.rpy`。两个文件要用不同的编号，以免覆盖已有录像。在 Replay 菜单中选择 `FLY60036`。不需要安装 Python 或下载模型。RPY 保留了六面的按键记录，FPS 显示字段填写为 60。

## 用 CPU 复算动作统计

从 Release 下载 `seed60036-traces.zip` 到仓库根目录。以下命令也适用于 Linux 和 macOS。

```shell
python -m pip install numpy
python experiments/seed60036/compare.py seed60036-traces.zip
```

脚本分别统计两局，并在两条轨迹首次分歧之前逐个比较 18 个动作分数。还可导入 `safe_idle.py` 中的 `apply_idle_bonus`，向尚未加分的动作分数应用同一规则。归档中的 `logits` 已包含该局实际采用的加分。

- 自机在正常作战带（原作纵坐标 280～360），不处于保护或特殊状态，附近无明确收点需要，无目标或已对齐 Boss。几何输入的可信标志有效，18 个候选动作在 4、12、24 帧的预测余量均达到 32 个原作坐标单位，才触发加分。
- 仅给动作 0（高速原地）的原始 logit 加 0.25，再选最高分动作并提交对应的脑状态。统计“原地”时，动作 0 和低速原地动作 9 都计入。0.25 是分数增量，不能按 25 个百分点解释。
- “移动时换向”只统计同面、间隔 1～4 原作帧，且相邻两次都选择移动的决策对。相邻方向不同算一次，高速与低速使用同一方向编号。

## 用冻结权重逐步计算动作

这里复算记录中实际遇到的 407 维观测，按模型自己选出的动作提交循环状态。所用网络有 138639 个神经元、8053 个视觉输入端口和 1299 个下降输出端口。每个候选的 70 个感觉输入经视觉端口送入同一个 LIF 网络。原仓库提供连接组核心、18 个候选的更新和三个价值头，本目录提供对应这份权重的走廊读出、静止评分和冻结稀疏计算。新开局的原作游玩可通过上面的 RPY 观看。旧版训练脚本使用自己的输入配置。

使用 Windows、NVIDIA CUDA 和仓库的 PyTorch 环境，在仓库根目录运行。先把 Release 中的 `cycle073-frozen.pt` 与 `seed60036-traces.zip` 放在根目录。已有固定版本的图谱时，直接用 `--connectome` 指向它。首次下载会从上游取 FlyWire v783 数据和带版本的端口注释。

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-repro.txt
python -m pip install -e .
python experiments/seed60036/fetch_graph.py --directory data/connectome
python experiments/seed60036/verify_model.py --connectome data/connectome --checkpoint cycle073-frozen.pt --traces seed60036-traces.zip --variant idle025 --steps 648
```

648 次决策覆盖第一次动作分歧。去掉 `--steps 648` 可验证整局。将 `--variant` 改为 `baseline` 可验证原版。脚本从零脑状态开始，逐项比较选中动作和 18 个最终分数。不同 GPU 或库版本可能出现数值差异，脚本报告首处分歧。发布权重包含全部 56 个训练参数或缓冲张量。已移除历史训练机器上的两个文件路径。

## 数据字段

ZIP 内的 `baseline.npz`、`idle025.npz` 各对应一次完整实机运行，均可用 `numpy.load(..., allow_pickle=False)` 读取。下表中每行对应一个决策时刻。

| 字段 | 含义 |
|---|---|
| `features` | 动作前的 407 维观察。前 129 项为作战与短程几何状态，随后 144 项为 18 个方向各 8 项路线量，8 项为任务上下文，最后 126 项是 18 个方向各 7 项走廊量 |
| `actions`, `logits`, `probabilities` | 实际动作及 18 个合法移动动作的最终分数、概率。0～8 高速，9～17 低速。每组依次为原地、左、右、上、下、左上、左下、右上、右下。NBNM 的 Bomb 概率恒为 0 |
| `frames`, `stage_frames`, `stages`, `phases`, `position` | 全局原作帧、面内原作帧、面编号、阶段和自机原作坐标。决策轴与 60 FPS 视频帧轴分别计数 |
| `population_rate`, `population_membrane` | 提交所选动作后的模拟群体均值，列顺序是脑内其余神经元、视觉输入端口、下降输出端口。来自对本局输入、动作和权重的神经活动重建 |

完整视频采用 1920×1080、60 FPS、HEVC 和立体声 AAC。画面使用各局自己的神经活动与动作概率，声音依据本局原作事件重建。18 秒并排片的声音来自 0.25 组。[附件大小与 SHA-256](assets.json)

本项目为东方 Project 非官方二次创作。游戏画面、角色、原作 BGM 和音效归上海爱丽丝幻乐团等相应权利人。MP4 作为个人项目的游玩演示提供。项目作者对自编代码、冻结权重和数值轨迹中可授权的部分采用本仓库 AGPL-3.0-or-later。FlyWire 数据与注释依上游条款使用。[第三方来源](../../THIRD_PARTY_NOTICES.md) · [东方 Project 视频发布指引](https://touhou-project.news/guideline_videoposting/)
