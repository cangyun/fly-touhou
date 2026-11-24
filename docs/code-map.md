# 代码地图 / Code map

从 [README](../README.md) 安装、选择原作和启动任务。复现成绩见 [REPRODUCING](REPRODUCING.md)。扩展原作先读[原作接口](original-game-interfaces.md)。

| 任务 / Task | 入口 / Entry |
| --- | --- |
| 启动作业 / Dispatch | [public_app.py](../src/fly_touhou/public_app.py) → [public_runs.py](../src/fly_touhou/training/public_runs.py) |
| 管理作业 / Manage jobs | [console.py](../src/fly_touhou/training/console.py) → [public_jobs.py](../src/fly_touhou/training/public_jobs.py) → [Windows launcher](../scripts/background_job.py) |
| 连接组、脉冲与脑状态 / Connectome and spiking state | [brain](../src/fly_touhou/brain) |
| 共享感觉编码与读出 / Shared encoders and readouts | [policy](../src/fly_touhou/policy) |
| PPO、采样、存档与评估 / PPO, rollouts, checkpoints, and evaluation | [training](../src/fly_touhou/training) |
| 《永夜抄》观测、输入与原作预测 / TH08 observation, input, and forecasts | [games/th08](../src/fly_touhou/games/th08) |
| 《辉针城》观测、原作事件与 Bomb 奖励 / TH14 observation, events, and Bomb rewards | [games/th14](../src/fly_touhou/games/th14) |
| 后端与配套依赖 / Native backend | [native](../native) · [hooks](../src/fly_touhou/hooks) |
| 示例、权重与结果 / Presets, weights, and evidence | [configs](../configs) · [weights](../weights) · [results](../results) |

历史脚本的导入路径指向同一份迁移后的源码。原作内存布局、ECL、角色、关卡和原作判定留在各自适配层。模型输入的字段顺序与评分仍由各作的版本化实现管理。文档组织参考了 [TH06NC-Neural-Pilot code map](https://github.com/xialongxl/TH06NC-Neural-Pilot/blob/main/docs/en/code-map.md)。
