# Fly Touhou / 果蝇 · 东方

使用完整 FlyWire 连接组约束的可训练脉冲神经网络，控制原作《东方辉针城》TH14 1.00b。Windows 原作接口、无渲染运行、教师辅助 PPO、神经风险读出与冻结评估的研究工程。

**本分支是 NBNM（不放 Bomb、不碰撞）专项训练实验。** 继承第 87 批的脑与移动参数，先提高无伤完成率，再缩短成功局用时。当前没有六面 NBNM 通关证明。已通关的有限残机版本仍在 [`main`](https://github.com/cangyun/fly-touhou/tree/main)。

English: NBNM training with a shared, full-connectome spiking controller. Bombs are disabled and the first native contact ends an attempt. The cycle087 policy supplies the warm start; successful spell-card practice is not evidence of a full-game NBNM clear. No new trained weights are released with this code branch.

## 三个版本

| 分支 | 内容 | 权重与结果 |
|---|---|---|
| **[`main`](https://github.com/cangyun/fly-touhou/tree/main)** | 首次通关时归档的源码 + 第 87 批参数 + v6 推理解码 | [v0.1.0-clear](https://github.com/cangyun/fly-touhou/releases/tag/v0.1.0-clear)，一个固定种子通关 |
| [`experimental/latest`](https://github.com/cangyun/fly-touhou/tree/experimental/latest) | 第 101 批最新实验：下半场位置奖励、移动指导和后续工程修订 | [实验预发布](https://github.com/cangyun/fly-touhou/releases/tag/v0.2.0-experimental-cycle101)，冻结 20 局 0 通关 |
| **[`experimental/nbnm`](https://github.com/cangyun/fly-touhou/tree/experimental/nbnm)** | 本分支：NBNM 奖励、281 维输入、单符卡与开局混合训练、CUDA Graph | 使用第 87 批作初始参数；NBNM 效果待冻结对照 |

实验版保留负面结果以供研究，**不推荐用来替换通关基线**。请让源码分支、Release 权重和配置保持配套。

## 本轮训练方案与证据边界

- 同一模型，默认 16 实例：12 个第三面 `BossCard1` 专项、4 个完整开局。每段 8,192 次决策，首轮 32,768 次决策、128 次 Adam。
- 成功 +1、首次碰撞/Miss/Bomb/Game Over −1；取消位置与生存驻留收入。不强制下半场，第一阶段不让速度奖励牺牲成功率。
- 规则教师给出全部 18 个移动方向的指导，学生执行自己的动作；保留风险监督和三个价值头。最终按成功率优先、成功局用时次之比较模型。
- `route` 和 `baseline` 结构相同；后者仅屏蔽新增的路线输入。专项前缀重放可能包含旧的 Bomb/Miss，因此单符卡成功不等于整局无伤。
- 训练采样的成功尝试不作为冻结成功率。初始模型本来就能通过部分符卡，必须在相同开发种子、同一动作解码下比较训练前后。

详见 [NBNM 方法与运行](docs/NBNM.md)、[试训进度分析](results/nbnm-pilot-status.json)。历史 [第 87 批通关](results/clear-v6.json) 和 [第 101 批负面结果](docs/EXPERIMENT-CYCLE101.md) 保留作来源记录，不属于本分支的新成绩。

## 模型究竟是什么

- **138,639 个神经元、15,091,983 条有向连接**，采用固定版本的数据预处理结果，保留全图拓扑与连接符号。这里的节点数不是完整公开图谱的 139,255。
- 使用带代理梯度的 LIF 风格脉冲单元，脑内连接增益与神经元参数可学习。每个候选动作内部更新 16 次，不宣称这些更新对应真实生物毫秒。
- 原 129 维状态加上 18 方向各 8 项路线信息、8 项任务信息，组成 **281 维观察**；对移动候选构造各 **63 维**感觉输入，编码到全部 **8,053 个匹配的视觉投射端口**。
- 通过 **1,299 个下降神经元的滤波脉冲活动和模拟膜电位**读取风险、路线、Bomb 与三个价值头。输入编码器和这些读出头是人工模块。
- 18 个候选共享同一组脑参数与决策前状态，只保留实际所选动作的后继状态。不是 18 个专家，也不是每面一个模型。
- 保留 19 维头以继承旧参数，实际只允许 18 个移动动作，Bomb 概率为 0。持续射击；NBNM 使用联合最大概率冻结解码，不继承旧的资源/Bomb/下半场控制目标。

这是**连接组约束的人工脉冲控制器**。它使用工程化的危险预测输入，不是原始像素端到端，也不等于真实果蝇神经活动的精确模拟。[方法说明](docs/METHOD.md)

## 安装与运行

支持路径：**Windows x64、Python 3.11、NVIDIA CUDA**。已使用 PyTorch 2.8.0 + CUDA 12.8 / RTX 5090；其他硬件尚未逐一验证。首次运行会编译小型 CUDA 内核。原作需要用户自行提供 **TH14 日文 1.00b**，不要用重制版或不同 EXE 偏移的汉化程序替换。

```powershell
git clone --branch experimental/nbnm https://github.com/cangyun/fly-touhou.git
cd fly-touhou
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-repro.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/setup_release.py
.\.venv\Scripts\python.exe scripts/prepare_nbnm_experiment.py --game-exe "D:/Games/TH14/th14.exe"
```

`setup_release.py` 复用 `v0.1.0-clear` 的第 87 批初始权重、无渲染 DLL，以及固定版本的连接组与注释数据；**不会启动游戏或训练**。所有生成内容默认放在仓库的 `models/`、`data/`、`outputs/`。已有环境和数据可复用，无须每个分支重新安装 CUDA 或复制连接组，见 [复用与启动](docs/NBNM.md#启动暂停与恢复)。

`run_release.py` 仍用于第 87 批有限残机基线的兼容评估，不是 NBNM 训练入口。NBNM 的入口与起点准备如下。原作后端移除渲染和帧率等待，仍运行原作 EXE，**不是独立重写的游戏模拟器**。

```powershell
# 规则教师采集起点：会运行原作，不训练网络
.\.venv\Scripts\python.exe scripts/prepare_nbnm_start_bank.py --config outputs/nbnm-preparation/environment.json --output outputs/nbnm-start-bank --train-seeds 4 --development-seeds 4
# 预检不会训练；加 --execute 才开始
.\.venv\Scripts\python.exe scripts/train_nbnm.py --config outputs/nbnm-preparation/route.json
.\.venv\Scripts\python.exe scripts/train_nbnm.py --config outputs/nbnm-preparation/route.json --execute
```

需要原游戏的配套数据文件，与 EXE 保持正常安装布局。仓库与附件不提供游戏、BGM、角色图像或录制视频。不要把推理速度当成同步展示回放的播放帧率。

## 工程入口

| 路径 | 用途 |
|---|---|
| `scripts/prepare_nbnm_experiment.py` | 生成便携的游戏环境、route/baseline 配置 |
| `scripts/train_nbnm.py` | NBNM 试训、恢复、独立冻结评估 |
| `scripts/prepare_nbnm_start_bank.py` | 原作单符卡前缀采集 |
| `scripts/run_release.py` | 已发布第 87 批基线的兼容评估 |
| `scripts/setup_release.py` | 获取发布附件与连接组数据 |
| `scripts/train_compact_story_ppo.py` | 全流程教师辅助 PPO 训练；[训练说明](docs/TRAINING.md) |
| `src/fly_touhou/compact_story_model.py` | SNN、候选共享状态、风险/路线/Bomb/价值读出 |
| `src/fly_touhou/compact_full_story.py` | 原作整局训练与评估流程 |
| `native/fake_d3d9/` | 无渲染后端的完整 Rust 源码与上游许可证 |
| `configs/`、`results/` | 版本绑定的配置和数值结果 |

`src/`、`scripts/` 还保留早期 GRU、连续模型和诊断研究工具；不是每个历史脚本都能在没有中间数据时独立运行。本分支支持上述 NBNM 入口，历史通关复现请使用 `main` 的原配源码。

进一步阅读：[复现边界](docs/REPRODUCING.md) · [版本来源与发布修改](docs/PROVENANCE.md) · [第三方来源](THIRD_PARTY_NOTICES.md)。

## 许可证与引用

项目代码采用 **AGPL-3.0-or-later**，见 [LICENSE](LICENSE)。保留上游源码的原始声明；原生无渲染后端保留 MIT OR Apache-2.0 双许可证。连接组和注释数据按各自上游来源取得，不能用本项目许可证代替数据来源的条款。

感谢 [THRL](https://github.com/touhourl/thrl)、[nokozero](https://github.com/astral4/nokozero)、[Drosophila brain model](https://github.com/philshiu/Drosophila_brain_model)、[FlyWire annotations](https://github.com/flyconnectome/flywire_annotations) 及其作者。方法借鉴、直接改编和数据来源的范围见第三方说明；本项目不是这些项目的官方分支。使用本工程请同时引用实际使用的上游数据与方法。
