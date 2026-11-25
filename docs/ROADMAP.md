# 训练方案与扩展方向 / Training plan and roadmap

[中文](#中文) · [English](#english)

## 中文

这套流程用于下一轮完整训练。已验证的模型、通关报告和当前可用的命令见 [复现说明](REPRODUCING.md) 与 [训练入口](TRAINING.md)。

### 五个训练阶段

| 阶段 | 数据和训练方法 | 预期学到的能力 |
| --- | --- | --- |
| 1. 初始化果蝇脑 | 从 FlyWire 连接组构建脉冲循环网络，保留神经元、有向连接和已有的兴奋／抑制符号，同步初始化原作感觉编码器与动作、风险和价值读出。 | 让游戏状态进入连接组，并从脑活动中读出决策。 |
| 2. 避弹监督预训练 | 用几何预测和原作碰撞记录，为各候选移动构造 **4／12／24 原作帧**的风险标签。用原作结果校验预测，屏蔽未覆盖的标签。 | 从脑活动区分安全和危险的移动方向。 |
| 3. 真人录像路线模仿 | 优先使用**同一原作的真人完整 NBNM 录像**。在原作中回放，逐帧配对观测与真人输入，按时间顺序训练有状态的果蝇策略。该阶段只做离线监督。 | 学习道中站位、安全收点、引弹、转向时机、符卡走法和段落衔接。 |
| 4. 实机 PPO 学习普通通关 | 从预训练权重出发，由果蝇自主操作。用学生实际采到的轨迹和环境奖励做 PPO。中弹可继续，允许自主 Bomb，目标为有限残机、不续关的 1CC。 | 处理偏离示范路线后遇到的新弹幕局面，同时管理残机和 Bomb。 |
| 5. 迁移到 NBNM | 继承普通通关策略，禁用 Bomb，首次原作碰撞即结束。继续实机训练，必要时从学生实际失败局面收集经原作验证的纠错路线，用于道中或符卡局部微调。 | 维持全程避弹与路线衔接，最终自主完成一次零 Miss、零 Bomb、零续关通关。 |

真人录像的整局留出评估用于检查模仿是否泛化。实机阶段由学生执行动作，冻结评测按原作结局和接触记录验收。目前普通通关入口支持《辉针城》，《永夜抄》的普通通关入口待适配。

### 扩展方向

| 方向 | 后续工作 |
| --- | --- |
| 感觉编码器与输出头的容量 | 固定连接组、输入特征和训练预算，分别扩大感觉编码器、动作头或风险头，比较避弹能力、冻结通关率和每次决策耗时。 |
| 奖励函数 | 对照贴边惩罚、火力和得分奖励，记录全程贴边时间、首次碰撞、无伤完成率和成功局用时。用真人路线教安全收点及撤回时机。 |
| 几何预测器的速度 | 复用同一快照的预测，重点减少候选分支的重复推演。用原作对照短程碰撞、长程出口和预测覆盖。 |
| 超参数与训练配方 | 对照学习率、采样温度、回放配额和监督权重，记录到达同等冻结成绩所需的决策数与时长。 |
| 探索、课程采样与能力保持 | 给困难道中和符卡分配额外样本，同时保留整面与开局连打，用未参与训练的整局验证前段能力。 |
| 扩展到更多东方原作 | 为各作适配原作步进、感觉特征和弹幕规则。比较冻结果蝇脑微调接口与全网络训练的迁移效果。 |

## English

This roadmap describes the next full training pipeline. See [reproduction](REPRODUCING.md) and [training commands](TRAINING.md) for the verified checkpoints, run reports, and supported entry points.

### Five training stages

| Stage | Data and training method | Capability |
| --- | --- | --- |
| 1. Initialize the fly brain | Build a recurrent spiking network from FlyWire. Keep its neurons, directed edges, and known excitatory and inhibitory signs. Initialize the game sensory encoder and the action, risk, and value readouts. | Carry game state through the connectome and read decisions from neural activity. |
| 2. Pretrain bullet avoidance | Use geometric predictions and original-game contact records to label each movement candidate over **4, 12, and 24 original frames**. Check predictions against native outcomes and mask labels with missing coverage. | Distinguish safe and dangerous movement from brain activity. |
| 3. Imitate human replay routes | Prefer a **complete human NBNM replay of the same original game**. Run it in the original game, pair observations with human input frame by frame, and train the stateful fly policy in temporal order. This stage uses offline supervision only. | Learn positioning, safe item collection, bullet luring, turn timing, spell-card routes, and transitions. |
| 4. Learn a regular clear with in-game PPO | Start from the pretrained weights. Let the fly play and update PPO with its own trajectories and environment rewards. Allow Bombs and continued play on remaining lives with a finite-life, no-continue 1CC objective. | Handle situations beyond the demonstration and manage lives and Bombs. |
| 5. Transfer to NBNM | Inherit the regular-clear policy, disable Bombs, and terminate at the first original-game contact. Continue in-game training. Fine-tune repeated road or spell failures with correction routes verified in the original game from states visited by the student. | Keep avoidance and route transitions intact throughout a no-Miss, no-Bomb, no-continue clear. |

Hold out entire human replays to evaluate imitation. The student executes the actions during in-game learning. Frozen evaluation checks native contact and the ending. The regular-clear entry point supports TH14, and the TH08 regular-clear adapter is pending.

### Expansion directions

| Direction | Follow-up |
| --- | --- |
| Sensory encoder and readout capacity | Hold the connectome, input features, and training budget fixed. Widen the sensory encoder, action head, or risk head individually and compare avoidance, frozen clear rate, and decision latency. |
| Reward design | Compare the side-wall cost, Power bonus, and score bonus. Record wall residence, first contact, no-Miss completion, and the duration of successful runs. Teach safe item collection and retreat with human routes. |
| Geometric forecasting speed | Reuse forecasts from the same snapshot and reduce repeated work across candidate branches. Check short-horizon contact, longer exits, and coverage against the original game. |
| Hyperparameters and training recipe | Compare learning rates, sampling temperatures, rehearsal quotas, and supervision weights. Measure the decisions and wall-clock time needed to reach the same frozen performance. |
| Exploration, curriculum, and retention | Allocate extra samples to difficult road segments and spell cards. Keep whole-stage and opening-to-ending attempts and test retention on held-out runs. |
| More original Touhou games | Adapt stepping, sensory features, and projectile rules to each game. Compare transfer with a fixed brain and trainable interfaces against joint training. |
