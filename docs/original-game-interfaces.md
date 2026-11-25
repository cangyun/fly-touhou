# 原作接口 / Original-game interfaces

接口由 [contracts.py](../src/fly_touhou/games/contracts.py) 定义，[TH14](../src/fly_touhou/games/th14/adapter.py) 和 [TH08](../src/fly_touhou/games/th08/adapter.py) 绑定到已有原作运行时。

## 原作适配器

统一入口选择原作 EXE、模式、难度与种子，使用各自支持的自机。适配器的 `observe()` 与 `step(action)` 提供原作帧号、自机、边界、危险对象、攻击、残机、Bomb、Miss、续关和终局。碰撞体与半径等几何数据保留各自的原作表示。NBNM 使用 18 个移动动作，普通通关使用原有 19 动作。

## 自动回收

`NativeFrame.item_collection` 由各作从同一输入点生成。新增原作时，在适配器中实现该字段，并使用该作的 Y 坐标、回收线、角色和火力条件。

| 字段 | 含义 |
| --- | --- |
| `line_y`、`distance_to_line` | 回收线的原作纵坐标，以及 `line_y - player_y`。正值表示自机已越线。 |
| `can_trigger_now` | 当前操作状态、位置和高低速是否满足越线回收条件。即使没有 P 点也可以满足条件。 |
| `fast_allowed_above_line`、`focused_allowed_above_line` | 保持当前游戏及火力条件，越线后两种移速各自是否可触发回收。 |
| `available_power_items`、`attracting_power_items` | 原作里尚可被越线回收的 P 点数量，以及已经进入回收吸附的 P 点数量。 |

未知字段用 `None`，两类 P 点的计数为零时表示确实没有观测到对应道具。此接口只读，回收由原作在玩家走位满足条件时执行。

- **TH08《永夜抄》** 从加载的自机射击配置读取回收线。低速越线可回收，普通自机高速越线需满火力，咏唱组和魔理沙单体高速越线也可回收。[原作回收条件](https://github.com/N0zoM1z0/th08/blob/main/src/ItemManager.cpp#L264-L279)
- **TH14《辉针城》** 使用当前自机对应的回收线，普通自机为 Y=128，魔理沙为 Y=148。高低速越线均可触发。[原作手册](https://thwiki.cc/附带文档:东方辉针城/游戏内Manual)

## 几何预测器

`predict(frame, candidates, horizons)` 接收同一原作输入点的候选动作带。默认是 4、12、24、48 和 96 原作帧。

- **`unknown`** 当前证据未覆盖时域。已知安全前缀的长度可单独报告。
- **`rule`** 原作规则或投影得到了结果。TH14 可读出同序列的候选余量与规则走廊。
- **`original`** TH08 可在原作恢复点实跑候选分支，返回首次接触和已验证的无接触长度。

走廊与验证长度可跨越多个动作段。无法确定的首次接触帧使用空值。几何预测如何参与评分由既有模型配置决定，最终动作由网络选出。

## 离边引导

接入新原作时，为教师提供当前自机坐标、**实际可移动边界**、各候选的第一拍落点及对应路线余量。至少需要可靠的 4／12／24 帧预测。已有更长走廊或不同出口数量时，一起参与比较。落点按原作的动作速度、输入延迟和边界钳位计算。缺少覆盖的时域留空。

[公共规则](../src/fly_touhou/nbnm_wall_exit_guidance.py) 在安全和接续相近时才提高向内动作的监督权重。离边的判定依据实际落点和可用空间，适用于每一面和每部接入的原作。风险标签仍来自几何和实际碰撞，冻结动作仍由网络选择。

## 脚本观测

`script(frame)` 提供可读取的原作计时、触发上下文与发弹信息。各作负责解释自己的 ECL、计时和内存布局。

## English

The adapter returns original frames, resources, collision objects, and terminal events. The geometry interface reports candidate movement tapes in the same frame. Its coverage distinguishes unknown, rule-projected, and original-executed results. The script interface exposes available native clocks, trigger contexts, and emission records. Each policy retains its versioned input and action ordering.

### Automatic item collection

The adapter exposes `NativeFrame.item_collection` from the same native input checkpoint. It supplies the native collection-line Y coordinate, signed clearance `line_y - player_y`, current trigger eligibility, above-line eligibility at each movement speed, and counts of pending and already homing Power items. A positive distance places the player above the line. Keep unavailable fields as `None`. The original game handles item pickup as the policy moves.

TH08 reads the line from the loaded shot data. At low speed, crossing the line enables collection. At high speed, the Magic Team and solo Marisa can also collect before full Power. TH14 uses the line for the selected character and allows collection above it at either speed. [TH08 item code](https://github.com/N0zoM1z0/th08/blob/main/src/ItemManager.cpp#L264-L279) · [TH14 manual](https://thwiki.cc/附带文档:东方辉针城/游戏内Manual)

### Wall-exit guidance

For a new game, provide the player's position, legal movement bounds, clamped first-command endpoints, and covered route clearances from the same native checkpoint. Use reliable 4-, 12-, and 24-frame evidence. Include longer witnessed corridors and distinct follow-up exits when available. Represent missing coverage explicitly. The [shared rule](../src/fly_touhou/nbnm_wall_exit_guidance.py) raises the teacher weight of an inward move when safety and continuation evidence are comparable. Frozen action selection remains with the policy.
