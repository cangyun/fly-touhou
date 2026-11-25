# 训练与局部微调 / Training and attack-local fitting

## 启动与保存

```powershell
fly-touhou train --game th08 --difficulty lunatic --objective nbnm --train-scope encoder-heads --game-exe "D:\Games\TH08\th08.exe" --workers 8 --dry-run
fly-touhou train --game th14 --difficulty lunatic --objective clear --train-scope joint --game-exe "D:\Games\TH14\th14.exe" --workers 8 --dry-run
fly-touhou console
```

`--trainable-head` 可重复指定输出组，例如 `movement` 和 `risk`。感觉编码仍沿脑网络获得梯度。作业目录保存命令、状态和日志，训练输出保存参数与 Adam。

`--resume` 读取同训练契约的 optimizer checkpoint。NBNM 默认使用对应的输出恢复，TH14 有限残机训练从新输出继续。切换原作、目标、训练范围或输入版本时，用 `--checkpoint` 显式选择兼容的模型参数并启动新优化器。新 checkpoint 记录原作、难度、目标、观测、动作、连接组和训练范围。

## 自动回收与真人录像采集

使用公开预设新建训练时，感觉输入默认包含当前回收状态。复用已有的回收线距离和待收 P 点数量，新增当前能否回收、高速越线资格、低速越线资格、吸附中 P 点数四列。TH14 普通通关输入为 **133 维**，两作的走廊版 NBNM 输入为 **411 维**。四列投影到现有视觉端口，由策略自行决定何时上行、切低速和撤回。[两作回收规则与适配接口](original-game-interfaces.md#自动回收)

已有 checkpoint 的冻结评估按权重保存的输入版本运行。同目录续训沿用原设置。要从旧模型学习新输入，用 `--checkpoint` 载入其参数并新建训练输出。自定义 NBNM 配置用 `nbnm.item_collection_input`，TH14 普通通关环境用 `compact_item_collection_input`，可显式设为 `true` 或 `false`。

采集真人走位时，TH08 的 [`capture_th08_teacher_windows.py`](../scripts/capture_th08_teacher_windows.py) 和 TH14 的 [`capture_nbnm_human_replay.py`](../scripts/capture_nbnm_human_replay.py) 均接受 `--item-collection-input`。启用后把回收特征和真人动作一起保存，TH08 同时记录每步实际 Power 增量 `native_power_gain`。可用这些片段监督安全上行、越线时的移速和收点后的撤退。

## 通用离边引导

靠近边界时，教师比较**两个都有可靠短程余量、且后续出路相近**的动作。向场内腾出更多空间的动作获较高指导权重，留在边上或继续靠外的动作相对降低。若只有一个方向有可靠出路，优先用出路证据。角落里的候选还要保留另一侧的余量。贴墙键和原地等待如果落点相同，就按相同的活动空间比较。

该规则源自《辉针城》v9 的留空间偏好。公开配置在《辉针城》普通通关和 NBNM、《永夜抄》NBNM 的全部关卡生效。默认最多转移匹配外侧动作 **20%** 的教师概率质量，沿用原作已算好的短程与续走证据。它进入训练的动作监督，冻结时由网络按自己的分数选动作。NBNM 的采样日志记录触发行数和转移的教师概率质量。[新作接入约定](original-game-interfaces.md#离边引导)

## TH08 符卡专项

```powershell
fly-touhou fit-attack --game th08 --difficulty lunatic --objective nbnm --guide results/attack-guides/th08-134 --dry-run
```

`--guide` 包含原作无伤完成此攻击的 `report.json` 及配套 `matched-history.npz`。脚本用固定脑重放感觉数据并优化该卡的动作读出。观测到的攻击编号和最小攻击年龄与权重一起保存。[99 号符卡数据](../results/attack-guides/th08-99) · [134 号符卡数据](../results/attack-guides/th08-134) · [原作续打验证工具](../scripts/evaluate_th08_prefix_window.py)

## 永夜抄 NBNM 奖励

新训练默认使用 [TH08 配置](../configs/th08/nbnm.json) 的 `nbnm.reward_shaping`。下表上限是第三价值通道乘默认权重 **0.2** 后的量级。一次任务内一直累计，跨采样批次继续计数。

| 项目 | 触发条件 | 默认量级 |
| --- | --- | --- |
| 左右贴边 | 进入实机左右边界内 **24 像素**。只有候选动作实际向内移动至少 **4 像素**，且原作验证覆盖 **4／12／24 帧**、三档碰撞余量均至少 **4 像素**，才累计可撤离的贴边时长。连续 **48 原作帧**后，继续贴边时按深度和帧数扣分。主动回撤的当步免扣。 | 完整六面合计至多 **−0.036**，每面至多 **−0.006**。单独整面或单次攻击任务至多 **−0.036**。 |
| 火力 | 实际 Power 超过本次任务此前最高值时，按增量除以满火力 **128** 计分。 | 至多 **+0.020**。 |
| 分数 | 读取原作真实分数。奖励正常操作中实际增加的分数，以 **1 亿分**为半饱和尺度，随累计增量提高而递减。 | 至多 **+0.004**。 |

旧普通通关奖励主要在纵向和横向同时靠边时扣角落分。左右贴边改用可行撤离证据，给引弹和不得不贴边的时段留空间。火力和分数取自实际执行的轨迹，过面、跳过对话和终局的自动变化只更新基准。Power 与原作分数会随进度变化，保留 **+1／−1** 的 NBNM 终局回报和成功局速度奖励，用于把资源奖励保持在次要位置。原始三项分别记入 `nbnm_reward_shaping`。[TH08 分数计量](https://github.com/GensokyoClub/th08/blob/079f403dafd62ff4cbee127e9ee3a50a8aad03f9/src/Global.hpp#L403-L410)

当前速度奖励随成功局的用时衰减，对击破的激励要到任务成功才出现。后续如需强化攻击，可以对照同一可受伤攻击阶段中的**实际 Boss 伤害**，并要求短程安全。横向贴近 Boss 只是一种走位，实际命中还取决于射角、距离与 Boss 状态。真人录像可以直接教安全收点、对齐和撤离时机。

同一公开作业目录中的 `--resume` 复用既有设置。要启用新奖励，在新输出目录用 `--checkpoint` 载入上一版权重。把 `nbnm.reward_shaping` 设为 `null` 可做旧奖励对照。辅助项只进入训练回报，冻结推理由原策略打分。

## English

The commands above preview the selected backend and budget. A matching `--resume` restores Adam. Use an explicit compatible `--checkpoint` when initializing a new optimizer. TH08 attack-local fitting uses a verified original-game certificate and matching sensory and action rows.

### Automatic item collection and human replays

Fresh public training presets include native collection observations. The encoder reuses the collection-line distance and pending Power count and appends four cues for current eligibility, fast and focused movement above the line, and already homing Power. This gives 133 features in TH14 clear and 411 in each corridor-based NBNM preset. The added projection uses the existing visual ports. The policy controls movement. [Adapter contract](original-game-interfaces.md#automatic-item-collection)

Frozen checkpoints select their saved feature version. Resume with the original run settings, or warm-start with `--checkpoint` in a new output directory to learn the new inputs. Custom configurations use `nbnm.item_collection_input` for NBNM or `compact_item_collection_input` for TH14 clear.

Pass `--item-collection-input` to [TH08 replay capture](../scripts/capture_th08_teacher_windows.py) or [TH14 replay capture](../scripts/capture_nbnm_human_replay.py) to record the cues alongside human inputs. TH08 tapes also provide the observed `native_power_gain` for each recorded action.

### Shared wall-exit guidance

When two moves near a boundary have reliable short-horizon clearance and comparable follow-up exits, the teacher gives more weight to the move that creates more space inward. Better-supported continuations take priority. Each game supplies its own bounds and clamped endpoints. The published TH14 clear and NBNM presets and the TH08 NBNM preset enable the rule in every stage. The default transfers at most 20% of each matching outer move's teacher mass. It supplies movement supervision during training. The network selects actions in frozen evaluation. [Adapter contract](original-game-interfaces.md#wall-exit-guidance)

### TH08 NBNM rewards

New training uses `nbnm.reward_shaping` in the [TH08 preset](../configs/th08/nbnm.json). The side-wall cost starts after 48 consecutive original frames in the 24-pixel side band with an available inward move. The move must advance at least four pixels inward and have at least four pixels of verified original-game clearance over each of 4, 12, and 24 frames. A step that retreats inward pays zero wall cost. After the existing progress weight of 0.2, maximum charges are 0.036 per full attempt and 0.006 per stage within it. A standalone stage or attack has its own 0.036 budget.

The Power bonus rewards gains above the attempt's previous highest Power, up to 0.020 in weighted return. The score bonus uses actual native score gains with diminishing returns and a 100-million-point half-saturation scale, up to 0.004. The three raw totals are logged as `nbnm_reward_shaping`. The terminal success or failure reward and successful-run pace reward apply as before. Frozen action scoring uses the saved policy.

A same-output public `--resume` reuses its original settings. Use `--checkpoint` with a new output to start from old weights under the new rewards. Set `nbnm.reward_shaping` to `null` for a matched comparison. If attack incentives need further study, compare actual damage in the same damageable Boss phase under short-horizon safety. Human replays can teach when to collect Power, aim, and retreat.
