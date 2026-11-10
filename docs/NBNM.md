# NBNM 训练准备

目标顺序：首先提高无 Bomb、无 Miss 的完成率，再缩短成功局的原作帧数。
这是新的训练任务；已通关的第 87 批参数和公开版本不被覆盖。

## 模型与输入

- 保留 138,639 个神经元、15,091,983 条有向连接、现有兴奋/抑制符号和 LIF 脉冲动力学。
- 8,053 个视觉投射端口输入，1,299 个下降神经元读出；每个候选动作仍执行 16 次内部更新。
- 原 129 维压缩状态保留；新增 18 个候选方向各 8 项路线信息，以及 8 项阶段/任务信息，总计 281 维。
- 路线信息包括较宽范围的 24 帧净空、条件线性 48/96 帧净空及有效性、18 个后续转向的出口比例、最佳余量和相反方向出口。
- 48/96 帧结果只基于当前已见对象。它不预测未来生成的子弹，不作为真实碰撞标签。已有 4/12/24 帧短程标签与实际原作碰撞记录继续用于校准。
- 感觉、动作、风险和脑内原有可训练参数均可更新。新增输入权重从零开始，原有脑和移动权重继承第 87 批；三个价值头的末层重新初始化，因为奖励语义改变了。
- 数值输入全部经过原有 VP 端口和脑网络。没有外置教师替学生执行动作，也没有另加绕过果蝇脑的路线动作筛选器。

`route` 与 `baseline` 共用同一结构、预算、教师和任务条件；`baseline` 仅将 144 维新路线输入置零。两组的教师仍看到相同的路线信息，避免把教师差异误当成输入效果。两组按顺序运行，默认先运行 route。

## 奖励和动作

三个价值通道：

| 通道 | 奖励 | 第一阶段 PPO 权重 | 提速阶段权重 |
|---|---|---:|---:|
| NBNM 完成 | 完成目标 +1；首次原作碰撞、Miss、Bomb 或 Game Over −1；其余 0 | 1 | 1 |
| 成功用时 | 仅成功终局得到 exp(−原作帧数 / 时间尺度) | 0 | 0.05 |
| 攻击进度 | 实测 Boss HP 进度，仅作辅助价值学习 | 0 | 0 |

使用 gamma=1、GAE lambda=0.995。失败得早不会少扣时间分；没有每帧扣时、下半场驻留收益、位置势函数、收资源收益或旧生存收入。

19 维输出保留以继承旧参数；Bomb 的概率严格为 0，环境也拒绝 Bomb 动作。第一触碰的原作 state4 就判失败，不等到 Miss 计数增长，更不能靠决死 Bomb 挽回。持续射击保持。

不强制下半场。无伤优先不等于蹲底，最快击破也不必然意味着待在下半场。站位通过真实完成率、全部方向的短程安全指导和后续出口学习，速度只在后续阶段参与策略目标。

移动监督采用学生当前场景的安全方向、未来净空、实际碰撞和 18 方向规则分布；决死专项监督权重为 0。旧资源教师的上收、普通 Bomb、下半场指导及额外旧轨迹回放不继承。

## 先验证单符卡，再扩大任务

默认试运行聚焦第三面 `BossCard1`，同时保留开局连打：

- 同一个共享果蝇模型，16 个原作实例：12 个进入专项、4 个从开局开始。
- 每段收集 8,192 次真实决策，严格分配 75% 专项、25% 连打；组内哪个实例就绪先处理哪个。
- 每段全部样本学习一遍，即 32 次 Adam 更新；没有 KL 早停、KL 学习率阈值或 8 次更新封顶。
- 初始预算 32,768 次决策、128 次 Adam。中间不插整局冻结评估；最后用未参与训练的原作种子作符卡冻结评估。
- 更新时原作暂停，之后从原位置继续。只有真实任务终局才重开，采样段结束不算终局。
- 用当前权重对前缀最后 64 个真实动作预热脑状态，已知动作只需计算一个候选。PPO 回放保留这份预热状态。
- 专项入口通过真实输入在原作中快速重放；并非进程内存快照或直接修改 ECL。前缀可能有 Bomb/Miss，**专项无伤成功不等于整局 NBNM**。
- 跨关加载可能让过关道具多移动少量帧、视觉随机数及 Bomb 库存有差异。NBNM 起点记录这些差异，不把已禁用的 Bomb 库存当成落点条件；最终仍核对人物、弹幕、道具、生命、火力、分数、脚本相对位置、玩法 RNG 和实际 Miss/Bomb/点道具事件。没有为了匹配落点而改写这些状态。进程地址只在比较时规范化。原有通关录像的严格重放规则保持原样。

起点库当前是小规模试运行集合，不能由它证明跨种子的稳定通关。有效后继续增加不同原作种子和第四至六面目标，仍更新同一个模型，并保留已通过内容的复习和完整开局验证。

速度阶段不会自动开启。当前工程门槛为同一任务至少 50 个不同原作种子的冻结评估、NBNM 完成率至少 90%，然后显式切换配置。0.05 的次要权重不是数学上严格的成功率约束：候选检查点仍必须先比较同组种子的成功数，成功数相同时才比成功局用时。专项用时尺度默认 3,600 帧，整局提速应改为约 36,000 帧。

## 提速与记录

- 使用全局就绪队列和任务组配额，避免等待每个实例都完成固定决策数。
- 不再额外重复执行旧 Bomb/资源教师回放；当前场景的监督与 PPO 共用一次脑网络反传。
- 采用已有的共享候选检查点反传、CUDA 稀疏乘法和融合边梯度。
- 推理可启用 CUDA Graph；每段采样结束释放图，更新后重新捕获，避免旧参数缓存。固定形状图减少 CPU 调度，但不减少 18×16 次脑计算，也不能据此承诺整轮训练提速数倍。[PyTorch 说明](https://docs.pytorch.org/docs/2.15/notes/cuda.html#cuda-graphs)
- 记录采样/学习用时、真实决策数、任务配额、原作碰撞、风险误判、NBNM 结果、成功用时及 PPO/价值/指导/风险损失。训练统计与冻结统计分开。
- 大文件默认放仓库 `outputs/`，可通过配置与缓存环境变量放到空间充足的磁盘；轮换保存最新训练状态和采样张量，另保留这四段各自的模型参数，JSON 日志保留。

## 启动、暂停与恢复

先按 README 下载第 87 批初始参数及连接组，再生成本机配置：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_nbnm_experiment.py --game-exe "D:/Games/TH14/th14.exe"
.\.venv\Scripts\python.exe scripts/prepare_nbnm_start_bank.py --config outputs/nbnm-preparation/environment.json --output outputs/nbnm-start-bank --train-seeds 4 --development-seeds 4
```

第二条会让规则教师运行原作以采集前缀，可能需要较长时间，但不更新网络。默认目标是第三面 `BossCard1`；修改目标时，两个准备脚本的 `--stage` 和 `--attack` 必须一致。起点库自行生成，不随源码分发。

配置为 `outputs/nbnm-preparation/route.json`。如需复用已有模型、连接组和环境，在准备命令传入 `--checkpoint`、`--connectome`、`--environment-config`；输出位置可由 `--run-root`、`--tensorboard-root` 设置。不同分支可共用同一个 Python 环境，PowerShell 启动器支持 `-Python D:/env/Scripts/python.exe` 或环境变量 `FLY_TOUHOU_PYTHON`，不必复制 PyTorch/CUDA。共享环境使用源码入口，不要将另一个分支的旧包安装结果误当成本分支源码。

准备检查（不会训练）：

```powershell
& ./.venv/Scripts/python.exe -B -X utf8 scripts/train_nbnm.py --config outputs/nbnm-preparation/route.json
```

双击项目中的 `启动NBNM专项训练.cmd`，才会正式开始（需要 PowerShell 7）。`暂停NBNM专项训练.cmd` 请求在当前采样段/Adam 更新保存后暂停。也可以直接给 `train_nbnm.py` 加 `--execute`。使用 `python -m tensorboard.main --logdir outputs/tensorboard` 查看训练与冻结评估曲线。

恢复命令（保留模型、Adam、已完成的更新以及当前学习段游标；进程退出后的原作实例重新进入合法起点）：

```powershell
$spec = Get-Content outputs/nbnm-preparation/route.json -Raw | ConvertFrom-Json
& ./scripts/launch_nbnm.ps1 -Mode route -Resume (Join-Path $spec.output 'latest-training.pt')
```

最后的四个开发种子只是筛查，不能用它宣称稳定 NBNM。不能将完成最快但频繁失败的模型选为最好模型，也不能把训练中不同权重的局合并成冻结成功率。

## 本次准备验证（2026-11-09）

- 4 个训练、4 个开发原作种子的入口都完成了真实重放；没有提前跳过目标符卡。
- 16 实例联调完成 256 次采样：192 次专项、64 次完整开局，Bomb 动作与决死监督均为 0；PPO 重算动作对数概率的最大误差为 7.15e-7。
- 该短窗口采样耗时 4.73 秒，约 54 决策/秒；初始化与预热耗时 75 秒。窗口较短且不含学习反传，不能外推为整轮训练速度。
- 继承第 87 批的模型在一个训练种子的单符卡探针中，以 2,348 原作帧完成了 NBNM 任务。此时没有进行新训练，这只证明专项流程能够运行到成功终局，不代表改进效果或六面 NBNM。
- CUDA Graph 的分数与脑状态和原推理路径一致；微基准的收益主要在小批量。17 项 NBNM 测试通过，原有相关模型、采样、共享反传与碰撞预测回归也通过。
- 上述是 2026-11-09 的准备验证，尚未训练。后续试训的公开摘要见 `results/nbnm-pilot-status.json`；历史本机完整输出不随源码分发。

## 2026-11-10 启动修复

训练环境补齐了 `tensorboard`，项目依赖声明也已登记。监控服务器的独立 Python 环境不自动供训练解释器导入；后续实验复用当前训练环境，不逐分支重装 PyTorch/CUDA。

启动入口现在将导入和初始化错误写入 `launch-route/startup-日期时间.log`，配置预检会实际导入 TensorBoard。修复了 NBNM 状态记录方法遗漏 `elapsed` 参数的问题，新增回归测试；18 项 NBNM 测试通过。失败记录保留，当前试训输出使用 `s3-card1-route-pilot-v2`，预算和训练方法不变。

## 试训结果与原生符卡入口（2026-11-10）

首轮试训完成了 32,768 次决策、128 次 Adam 更新。训练结束后的自动评估曾因向只读观测数组写入而报错；参数已经保存，现已修复评估器并重新运行独立冻结对照。

同样四个开发种子、相同 Story 前缀和脑状态预热输入下，第 87 批初始模型通过 4/4，试训后通过 3/4。新增失败为种子 35537、符卡第 1,340 帧。小样本没有证明避弹提升；其余配对耗时下降不能抵消失败。本轮不因此扩大到完整六面训练。

TH14 1.00b 自带 **Spell Practice**。新增入口直接使用它，保留原作脚本、碰撞和随机数控制，不需要回放一至三面的道中。已实测的 Lunatic 灵梦 A 入口如下；菜单行和原作 ID 都从 0 开始，不能混用。

| Stage | ECL | 本面菜单行 spell_index | 原作 spell_id |
|---|---|---:|---:|
| 3 | BossCard1 | 1 | 31 |
| 3 | BossCard2 | 2 | 35 |
| 4 | BossCard2 | 1 | 47 |

已有用户存档需解锁对应符卡。只向每个实例的私有 APPDATA 复制该存档，不改原存档；未解锁会明确报错。菜单按键发给所拥有的原作进程，不操作桌面键鼠。新进程到第一张卡的脚本起点约 3–4 秒，后续推理/反传速度仍取决于模型。

原生练习的满火力、人物位置和初始脑状态，与 Story 前缀不同：本入口在首个符卡脚本帧以零循环状态开始。两种入口在报告中分别记录，禁止跨入口直接配对声称改进。首触碰/Bomb 优先判失败；成功要求原作符卡结果菜单及正的取得奖励，不把 Boss 消失、外部超时或普通下一段当作原生符卡取得。单卡成功永远不是六面 NBNM。

从已有本机 NBNM 配置生成第三面第二张 Boss 符卡配置（只写配置，不训练）：

```powershell
python scripts/prepare_nbnm_spell_practice.py --config outputs/nbnm-preparation/route.json --score D:/Saves/scoreth14.dat --stage 3 --spell-index 2 --spell-id 35 --attack BossCard2 --output outputs/s3-card2-native.json --run-output outputs/s3-card2-native-training
```

冻结测试指定模型：

```powershell
python scripts/train_nbnm.py --config outputs/s3-card2-native.json --evaluate-only models/cycle087-parameters.pt --evaluation-output outputs/s3-card2-native-evaluation
```

同种子比较两个模型：

```powershell
python scripts/compare_nbnm_frozen.py --config outputs/s3-card2-native.json --trained outputs/nbnm-route/latest-parameters.pt --output outputs/s3-card2-native-paired
```

模型和 Python 环境继续复用。明确加 `--execute` 才训练，训练时全部实例从该符卡开始，首次碰撞即终止，重试时从训练分区重新抽原作种子；配置中的四个开发种子只是快速筛查。正式结论需扩大独立测试，且再次确认 Story 入场的适应性。
