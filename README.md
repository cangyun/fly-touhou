# Fly Touhou / 果蝇 · 东方

使用完整 FlyWire 连接组约束的可训练脉冲神经网络，控制原作《东方辉针城》TH14 1.00b。Windows 原作接口、无渲染运行、教师辅助 PPO、神经风险读出与冻结评估的研究工程。

**本分支是已通关的第 87 批 v6 配置。** 已记录一次固定开局种子的 Lunatic 六面、不续关、有限残机通关；这不是稳定通关率的证明。推理时使用同一个果蝇网络，教师不接管动作。

English: A full-connectome-constrained spiking controller for the original Touhou 14. This branch preserves one verified fixed-seed Lunatic 1CC, not a claim of robust or multi-seed clearing. Code, weights, native backend source and evaluation evidence are released separately from the copyrighted game.

## 两个版本

| 分支 | 内容 | 权重与结果 |
|---|---|---|
| **[`main`](https://github.com/cangyun/fly-touhou/tree/main)** | 首次通关时归档的源码 + 第 87 批参数 + v6 推理解码 | [v0.1.0-clear](https://github.com/cangyun/fly-touhou/releases/tag/v0.1.0-clear)，一个固定种子通关 |
| [`experimental/latest`](https://github.com/cangyun/fly-touhou/tree/experimental/latest) | 第 101 批最新实验：下半场位置奖励、移动指导和后续工程修订 | [实验预发布](https://github.com/cangyun/fly-touhou/releases/tag/v0.2.0-experimental-cycle101)，冻结 20 局 0 通关 |

实验版保留负面结果以供研究，**不推荐用来替换通关基线**。请让源码分支、Release 权重和配置保持配套。

## 已验证的通关记录

| 项目 | 数值 |
|---|---:|
| 开局种子 | `3574003864` |
| 难度 / 规则 | Lunatic，六面，有限残机，禁止续关 |
| 原作累计帧 / 策略决策 | 135,934 / 33,828 |
| Miss / Bomb | 6 / 25 |
| 确认的决死 Bomb | 24 |
| 推理选择 | `family_mode` |
| 评估时训练更新 / 教师动作 | 0 / 0 |

原始数值报告见 [results/clear-v6.json](results/clear-v6.json)。这是开发阶段选定的种子，后续动作会影响原作 RNG；不要把一次通关表述为“100% 通关率”。

## 模型究竟是什么

- **138,639 个神经元、15,091,983 条有向连接**，采用固定版本的数据预处理结果，保留全图拓扑与连接符号。这里的节点数不是完整公开图谱的 139,255。
- 使用带代理梯度的 LIF 风格脉冲单元，脑内连接增益与神经元参数可学习。每个候选动作内部更新 16 次，不宣称这些更新对应真实生物毫秒。
- 原作几何、运动、资源、保护状态等组成 **129 维观察**；对 18 个移动候选构造各 **47 维**感觉输入，编码到全部 **8,053 个匹配的视觉投射端口**。
- 通过 **1,299 个下降神经元的滤波脉冲活动和模拟膜电位**读取风险、路线、Bomb 与三个价值头。输入编码器和这些读出头是人工模块。
- 18 个候选共享同一组脑参数与决策前状态，只保留实际所选动作的后继状态。不是 18 个专家，也不是每面一个模型。
- 19 个动作 = 9 种移动（含停留）× 高速/低速 + Bomb；持续射击。运行时还包括动作合法性处理、保护期风险组合及第六面 v6 解码设置。

这是**连接组约束的人工脉冲控制器**。它使用工程化的危险预测输入，不是原始像素端到端，也不等于真实果蝇神经活动的精确模拟。[方法说明](docs/METHOD.md)

## 安装与运行

支持路径：**Windows x64、Python 3.11、NVIDIA CUDA**。已使用 PyTorch 2.8.0 + CUDA 12.8 / RTX 5090；其他硬件尚未逐一验证。首次运行会编译小型 CUDA 内核。原作需要用户自行提供 **TH14 日文 1.00b**，不要用重制版或不同 EXE 偏移的汉化程序替换。

```powershell
git clone https://github.com/cangyun/fly-touhou.git
cd fly-touhou
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-repro.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/setup_release.py
.\.venv\Scripts\python.exe scripts/run_release.py --game-exe "D:/Games/TH14/th14.exe"
```

`setup_release.py` 下载本分支权重、无渲染 DLL，以及固定版本的连接组与注释数据；**不会启动游戏或训练**。所有生成内容默认放在仓库的 `models/`、`data/`、`outputs/`，可以把整个仓库放在空间充足的磁盘。

`run_release.py` 默认执行本分支保存的冻结评估计划；`main` 是一次首次通关种子，实验分支是已记录的 20 个开局。每次创建独立输出目录，查看其中的 `progress.json`、`development/report.json` 或 `failure.json`。它仍运行原作 EXE，只移除渲染和帧率等待，**不是独立重写的无头游戏模拟器**。运行速度取决于原作采样和全脑推理，不保证 60 FPS。

```powershell
# 仅打印配置和启动命令，不启动任何游戏
.\.venv\Scripts\python.exe scripts/run_release.py --game-exe "D:/Games/TH14/th14.exe" --dry-run
# 自行指定一组开发评估开局
.\.venv\Scripts\python.exe scripts/run_release.py --game-exe "D:/Games/TH14/th14.exe" --opening-seeds 123 456 789
```

需要原游戏的配套数据文件，与 EXE 保持正常安装布局。仓库与附件不提供游戏、BGM、角色图像或录制视频。不要把推理速度当成同步展示回放的播放帧率。

## 工程入口

| 路径 | 用途 |
|---|---|
| `scripts/run_release.py` | 本分支冻结模型的便携启动入口 |
| `scripts/setup_release.py` | 获取发布附件与连接组数据 |
| `scripts/train_compact_story_ppo.py` | 全流程教师辅助 PPO 训练；[训练说明](docs/TRAINING.md) |
| `src/fly_touhou/compact_story_model.py` | SNN、候选共享状态、风险/路线/Bomb/价值读出 |
| `src/fly_touhou/compact_full_story.py` | 原作整局训练与评估流程 |
| `native/fake_d3d9/` | 无渲染后端的完整 Rust 源码与上游许可证 |
| `configs/`、`results/` | 版本绑定的配置和数值结果 |

`src/`、`scripts/` 还保留早期 GRU、连续模型和诊断研究工具；它们不是通关模型的依赖清单，也不是每个脚本都能在没有历史中间数据时独立运行。支持的复现入口以上述 `setup_release.py` / `run_release.py` 为准。

进一步阅读：[复现边界](docs/REPRODUCING.md) · [版本来源与发布修改](docs/PROVENANCE.md) · [第三方来源](THIRD_PARTY_NOTICES.md)。

## 许可证与引用

项目代码采用 **AGPL-3.0-or-later**，见 [LICENSE](LICENSE)。保留上游源码的原始声明；原生无渲染后端保留 MIT OR Apache-2.0 双许可证。连接组和注释数据按各自上游来源取得，不能用本项目许可证代替数据来源的条款。

感谢 [THRL](https://github.com/touhourl/thrl)、[nokozero](https://github.com/astral4/nokozero)、[Drosophila brain model](https://github.com/philshiu/Drosophila_brain_model)、[FlyWire annotations](https://github.com/flyconnectome/flywire_annotations) 及其作者。方法借鉴、直接改编和数据来源的范围见第三方说明；本项目不是这些项目的官方分支。使用本工程请同时引用实际使用的上游数据与方法。
