# 复现范围与运行细节

## 支持的入口

从对应分支克隆代码，按 README 安装后运行 `scripts/setup_release.py` 和 `scripts/run_release.py`。不要直接运行历史实验中写死的路径；公共导出已把本机路径换成相对占位路径，但历史数据并未全量发布。

已验证环境为 Windows、Python 3.11、PyTorch 2.8.0+cu128、RTX 5090、TH14 原版日文 1.00b。GPU 内核通过 PyTorch 随附的 NVRTC 编译；本机不需要为了推理重新编译 Rust 后端。仓库仅承诺提供归档配置和数值记录，不保证不同驱动/GPU 下长期脉冲阈值轨迹逐帧一致。

## 发布附件

- `cycle087-parameters.pt`：`main` 对应的原始参数文件。
- `cycle101-parameters.pt`：`experimental/latest` 对应的最新参数文件。
- `native-windows-x86.zip`：32 位无渲染 Direct3D9 替代 DLL及许可证。

权重是 PyTorch checkpoint。只加载可信来源的文件。下载脚本按分支清单选版本，不会悄悄选择 GitHub 上“最新”预发布。两个模型可同时存放在 `models/`，运行时只加载当前分支清单中的一个。

完整连接组从 `philshiu/Drosophila_brain_model` 的固定提交下载；注释表从 `flyconnectome/flywire_annotations` 固定的 v2.1.0 提交下载。小型端口顺序清单随源码提供。安装过程无需 FlyWire 登录，不会抓取游戏或任何游戏素材。

离线/复用已有下载：

```powershell
python scripts/setup_release.py --assets-directory "D:/Downloads/fly-touhou-assets" --data-source "D:/Existing/connectome"
```

目录中应有对应分支的权重附件和 `native-windows-x86.zip`；数据源应有完整 CSV、Parquet、许可证、model.py 和 annotations 文件。

## 运行、停止与结果

默认无渲染模式依然启动独立原作进程，在进程内控制输入。它不是脱离 EXE 的游戏模拟器，不会因为没有窗口就自动带来无限采样速度。

使用 `--output "D:/Results/new-evaluation"` 或 `--cache "D:/Cache/fly-touhou"` 可把日志与缓存放到其他盘。输出目录必须不存在；已有报告不覆盖。

运行中查看输出目录中的 `progress.json`。若需中止，在输出目录创建空的 `STOP` 文件；评估器会在控制边界退出并保留已有记录。中止不等于完成评估，应根据 `failure.json`/`progress.json` 判断状态。

每个开局种子对应一局，不续关。最终 `development/report.json` 记录通关、到达各面、Miss、Bomb 等；`result.json` 是摘要。开局种子相同并不意味着采取不同动作后原作随机状态仍相同。报告里的 `evaluation_split=development` 是真实实验定位，不能事后改称独立正式测试。

首次通关配置使用 `family_mode`：先考虑移动动作族与 Bomb 族，再选族内动作。直接对 19 个动作做 `argmax` 的 `joint_mode` 会改变行为，不是同一发布评估。

## 自行编译原生后端

需要 Rust >= 1.88、MSVC C++ 构建工具和 32 位目标：

```powershell
rustup target add i686-pc-windows-msvc
cargo build --release --locked --target i686-pc-windows-msvc --manifest-path native/fake_d3d9/Cargo.toml
Copy-Item native/fake_d3d9/target/i686-pc-windows-msvc/release/th14_fake_d3d9.dll native/th14_fake_d3d9.dll
(Get-FileHash native/th14_fake_d3d9.dll -Algorithm SHA256).Hash.ToLower() | Set-Content native/th14_fake_d3d9.dll.sha256 -Encoding ascii
```

构建出的 DLL 应与源码和工具链对应；重编译后请保留构建记录。固定的发布附件使用原工程的已运行 DLL。

## 本次开源整理的验证

公共入口有 CPU 单元测试、源码语法检查、配置/种子绑定检查和发布资源安装测试。可以运行：

```powershell
python -m unittest discover -s tests -p "test_release_*.py" -v
```

开源整理没有重新训练，也没有重新跑一遍六面来替换已归档的通关结果。原有运行证据与本次打包验证是两类不同的证据。
