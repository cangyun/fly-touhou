# 原作接口 / Original-game interfaces

接口由 [contracts.py](../src/fly_touhou/games/contracts.py) 定义，[TH14](../src/fly_touhou/games/th14/adapter.py) 和 [TH08](../src/fly_touhou/games/th08/adapter.py) 绑定到已有原作运行时。

## 原作适配器

统一入口选择原作 EXE、模式、难度与种子，使用各自支持的自机。适配器的 `observe()` 与 `step(action)` 提供原作帧号、自机、边界、危险对象、攻击、残机、Bomb、Miss、续关和终局。碰撞体与半径等几何数据保留各自的原作表示。NBNM 使用 18 个移动动作，普通通关使用原有 19 动作。

## 几何预测器

`predict(frame, candidates, horizons)` 接收同一原作输入点的候选动作带。默认是 4、12、24、48 和 96 原作帧。

- **`unknown`** 当前证据未覆盖时域。已知安全前缀的长度可单独报告。
- **`rule`** 原作规则或投影得到了结果。TH14 可读出同序列的候选余量与规则走廊。
- **`original`** TH08 可在原作恢复点实跑候选分支，返回首次接触和已验证的无接触长度。

走廊与验证长度可跨越多个动作段。无法确定的首次接触帧使用空值。几何预测如何参与评分由既有模型配置决定，最终动作由网络选出。

## 脚本观测

`script(frame)` 提供可读取的原作计时、触发上下文与发弹信息。各作负责解释自己的 ECL、计时和内存布局。

## English

The adapter returns original frames, resources, collision objects, and terminal events. The geometry interface reports candidate movement tapes in the same frame. Its coverage distinguishes unknown, rule-projected, and original-executed results. The script interface exposes available native clocks, trigger contexts, and emission records. Each policy retains its versioned input and action ordering.
