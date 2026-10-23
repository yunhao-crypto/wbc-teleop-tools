# Vita S100 actor/Pico 部署

本仓库以 `vita_p123_20261015_sdkmain_opt021` 部署快照作为初始基线，
提供 Vita02A 的 `actor_g1.onnx` 推理及 Pico 实时输入链路。
**actor 部署所需的离线 wheel、Pico ARM 服务、定制绑定和构建源码均纳入 Git；模型单独交付。**
部署者需要提供指定版本的 `actor_g1.onnx`，安装脚本校验 SHA256 后复制到本地模型目录。
无需复制原开发目录的虚拟环境。

当前提供模型检查、离线回放、实时输入影子推理。影子入口接收真实 Pico 和
Body31 反馈，完成重定向与推理并记录结果；不申请 EXTERNAL，不发布动作命令。

## 在 Vita02A / S100 上部署

目标环境为 Linux ARM64、Python 3.10、glibc >= 2.35，需具备系统 venv/ensurepip。
这些是 SDK wheel 的平台要求。Pico 和机器人反馈仍需目标机已有的匹配固件、
设备自有 Zenoh peer 配置，以及已开启全身追踪和发送的头显客户端。

在一个**新克隆目录**中运行：

```bash
python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx
bash scripts/start_actor_pico.sh check --steps 100
bash scripts/start_actor_pico.sh shadow --seconds 60 --group default
```

安装脚本先校验全部交付文件，离线创建 `venv`、安装匹配依赖、解包 Pico 服务与
预编译绑定，再验证 SDK/绑定和动态库依赖，并执行 10 步无动作模型检查。
原有、非该脚本创建的 `venv` 会被保留并拒绝覆盖。已有安装可重跑以恢复或继续安装。

若希望在 S100 编译 Pico 绑定，额外准备 g++ 和 Python 3.10 开发头文件，然后运行：

```bash
python3.10 scripts/bootstrap_actor_s100.py --model /path/to/actor_g1.onnx --build-pico
```

仅 Pico Python 绑定和 double-conversion 需要此源码构建；SDK 使用匹配的官方
ARM64 wheel，厂商 Pico 服务使用仓库内固定版本的运行包。
详见 [S100 离线交付说明](deploy/actor_s100/README.md)和
[actor/Pico 运行说明](README_ACTOR_PICO.md)。

## Git 中包含什么

| 内容 | 路径 |
| --- | --- |
| 模型接口配置及 SHA256 | `config/actor_pico.json`（模型二进制单独交付） |
| 8 个匹配的离线 wheel | `deploy/actor_s100/wheelhouse/` |
| Pico ARM 服务、SDK 库、Qt 插件 | `deploy/actor_s100/runtime/pico_service_arm64.tar.gz` |
| 已验证的 Python 3.10 ARM64 绑定 | `deploy/actor_s100/runtime/pico_binding_cp310_aarch64.tar.gz` |
| Pico 定制源码和 double-conversion | `third_party/` |
| 可在干净克隆中使用的合成回放 | `tests/fixtures/actor_pico/pico_replay.npz` |
| 交付文件 SHA256 清单 | `deploy/actor_s100/manifest.json` |
| 源码、脚本和测试 | `src/`、`scripts/`、`tests/` |

运行生成的 `venv/`、`build/`、`vendor_packages/`、日志、输出和私有连接凭据仍不提交。
Pico 运行包不带已保存的设备/部署记录和用户连接配置，保留空白出厂数据库结构。
模型二进制由 `.gitignore` 排除，两个运行归档随 Git 提交。其他历史 HoloMotion P1/P2/P3
权重未纳入这次 actor 交付；其原源码仍保留，但不是本次离线资源闭环的目标。

## 验证提交内容能否独立使用

维护者在修改后先暂存文件、重新封存清单，再检查暂存区导出的干净副本：

```bash
git add .
python3 scripts/seal_actor_delivery.py
git add ACTOR-PICO-MANIFEST.json deploy/actor_s100/manifest.json
python3 scripts/verify_actor_delivery.py --host-inference --model /path/to/actor_g1.onnx
```

检查只接收显式指定的外部模型，不借用工作目录的 `venv`、`vendor_packages`、`reference` 或 `build`。
它会验证全部文件/归档哈希、离线解析完整 ARM wheel 依赖、从交付归档解包运行文件，
编译实际 Pico 原子快照 C++ 回归，并使用宿主 ONNX Runtime 运行 14 项 actor 测试、
3 项 Pico 读取测试、100 步模型检查和 100 步合成 Pico 回放。
宿主模式需已有 NumPy、ONNX Runtime、pip 和 C++ 编译器；它不执行 ARM 二进制。

S100 实际安装、推理性能、真实 Pico 输入质量和动作控制仍需分别在目标机验收。
参考窗口延迟为 0.92 秒；超出重定向误差阈值的姿态会停止推理。

## 来源

本次 Git 初始化不继承 `humanoid-sdk` 或原应用仓库历史。
SDK 锁定为 `locomotion-aorta 0.2.1`，记录源码提交
`f411caeff5c98a4b933f0cac084b03aabfefd763`；消息/二进制校验见
`config/sdk021-wheel-guard.json`。原升级记录标注的应用源码基线为
`b3e36cbb1e4992d225f890a67dba0b494c99326e`，较早 SDK 提交为
`b697f8bbdb76b3da4a8625e00aca212b93d5cd31`。其他来源见 `source_map.json`。
版本文件不等于与当前 vita-robot master 或实机有效控制配置完全一致。

初始快照部分旧文件带有历史空白格式提示，本次保留原文以维持来源哈希。
# wbc-teleop-tools
