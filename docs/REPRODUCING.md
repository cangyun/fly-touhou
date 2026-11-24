# 复现 / Reproduction

按 [README](../README.md) 准备 CUDA 环境、Git LFS 权重及 FlyWire FAFB v783。已有 `Completeness_783.csv`、`Connectivity_783.parquet` 和配套注释时，可用 `python scripts/setup_release.py --data-source <数据目录>`。命令的 `--connectome` 也可以直接指向完整的数据目录。

TH14 需要日文原作 1.00b 和灵梦 A。TH08 需要配置锁定的 1.00d 原作和结界组，EXE 同目录需有 `th08.dat` 与 `thbgm.dat`。原生组件见 [native](../native/README.md)。

| 复现目标 | 示例配置 | 种子 | 权重和证据 |
| --- | --- | --- | --- |
| TH14 Lunatic 1CC | [有限残机](../configs/th14/clear-environment.json) | 3574003864 | [cycle087](../weights/published/th14-clear-cycle087.pt) · [报告](../results/clear-v6.json) |
| TH14 Lunatic NBNM | [NBNM](../configs/th14/nbnm.json) | 60036 | [cycle073](../weights/published/th14-nbnm-cycle073.pt) · [报告](../results/th14-nbnm-60036.json) |
| TH08 Lunatic NBNM | [NBNM](../configs/th08/nbnm.json) | 18233 | [TH08 18233](../weights/published/th08-nbnm-18233.pt) · [报告](../results/th08-nbnm-18233.json) |

```powershell
fly-touhou evaluate --game th14 --difficulty lunatic --objective nbnm --game-exe "D:\Games\TH14\th14.exe" --seed 60036
fly-touhou evaluate --game th08 --difficulty lunatic --objective nbnm --game-exe "D:\Games\TH08\th08.exe" --seed 18233
```

`--dry-run` 查看实际路径和执行命令。报告保留各自的验收范围、种子和原作终局字段。

## English

Use the CUDA environment in the [README](../README.md), exact game builds, and the pinned FAFB v783 graph. Each row links the matching preset, seed, checkpoint, and original-game evidence. `--dry-run` shows resolved paths and arguments.
