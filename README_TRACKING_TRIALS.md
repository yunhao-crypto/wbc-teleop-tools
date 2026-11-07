# 42 三模型顺序测试（2026-11-06）

以下脚本已部署到 10.100.23.42 的 `/userdata/wbc-teleop-tools`。由操作员在现场确认机器人有支撑后手动启动，一次运行一条；不需要 PICO 输入。

```bash
# A31000：SuperMimic，腰 pitch 90/6，踝电机 30/1.2247449
bash /userdata/wbc-teleop-tools/scripts/test_supermimic_A31000.sh

# B25000：SuperMimic，腰 pitch 120/8，踝电机 40/1.4142136
bash /userdata/wbc-teleop-tools/scripts/test_supermimic_B25000.sh

# Sonic20000：腰 pitch 59.484104/3.7868757，踝电机 20/1
bash /userdata/wbc-teleop-tools/scripts/test_sonic_20000.sh
```

脚本会按原有流程进入 FIXED_STAND、EXTERNAL 和参考动作。播放结束后退出；需要中止时按 Ctrl+C，执行原有 CANCEL → PASSIVE 流程。必须确认上一条策略已经退出再启动下一条。

三条脚本使用相同的 01–09、11 顺序集，跳过 10，每个源动作最多 10 秒，11 也取开头 10 秒。动作之间保留 3 秒插值，首尾保留 0.6 秒缓入缓出，总参考长度 126.92 秒，50 Hz、6347 帧；另有启动过渡和末尾 1 秒保持。

参考 SHA256：`c787a13be9675527dc9f0b114bb12e3a573bf33814fb604068242b7f6f812956`。

每条命令默认开启 `--record-observations`，日志保存在：

```text
/userdata/wbc-teleop-tools/outputs/tracking_20261106_A31000_*.jsonl
/userdata/wbc-teleop-tools/outputs/tracking_20261106_B25000_*.jsonl
/userdata/wbc-teleop-tools/outputs/tracking_20261106_Sonic20000_*.jsonl
```

日志包含模型与参考指纹、模型输入与输出、参考关节、下发关节目标和 PD、实机位置/速度及 IMU、推理耗时和退出原因。没有附带外部摄像机录像。

在命令末尾加 `check` 可运行 100 步不创建电机控制接口的模型检查。检查日志也写到 outputs，通过日志头的 scope 与实机试验区分。

每个试验使用独立的 `trials/tracking_20261106/<标签>/` 配置、模型和运行环境。启动前检查模型、代码、参考、完整仿真记录以及本机有效配置的 SHA256。脚踝使用原生 `motor_theta`，`ankle_virtual_pd_intent=false`；PD 从各自 ONNX 合同读取。常规启动脚本的默认模型不因本轮试验而改变。

三者已在不带夹爪的 Vita02A 闭链模型上完整跑完同一套参考，没有触发摔倒判据：

| 模型 | 完成帧数 | 仿真关节 RMSE（度） | 最大身体倾角（度） |
| --- | --- | --- | --- |
| A31000 | 6347/6347 | 7.15 | 21.58 |
| B25000 | 6347/6347 | 5.83 | 27.76 |
| Sonic20000 | 6347/6347 | 5.29 | 24.52 |

这些是标称物理仿真结果，实机效果仍需实际测量。A/B 的 checkpoint 也不同，因此两者差异不能全部归因于 PD。

本地任务目录为 `/host/tracking_tests_20261106`。试验结束后在本地执行：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/vita-npz-review-venv/bin/python \
  /host/tracking_tests_20261106/collect_and_analyze.py
```

采集器下载每个模型最新一轮已结束的实机日志并校验 SHA256，保存到 `hardware/<标签>/<记录名>/`。生成逐关节、逐动作 tracking 指标、反馈数组以及完整试验的对比图。未完成的试验只报告已播放前缀，不参与完整套件排名；踝部使用实机反馈的虚拟 pitch/roll 与参考比较，避免将电机 theta 与虚拟参考混算。使用 `--local-only` 可只分析已经拉回的数据。

## Sonic9700 同机对照

昨天的 9700 实机记录来自 173。要与今天 42 上的 20000 比较，需要在 42 上补跑相同的 9700 顺序集：

```bash
bash /userdata/wbc-teleop-tools/scripts/test_sonic_9700.sh
```

该入口固定校验 9700 模型 SHA256，再调用现有的 `start_sonic_motion_sequence.sh`，动作、录制选项、启动流程与 20000 相同。日志使用原有的 `outputs/sonic_sequence_*.jsonl` 命名；本地采集器同时识别该路径，按模型指纹保存到 `hardware/Sonic9700/`，与 20000 的同机记录比较。末尾加 `check` 可进行 100 步无动作检查。

## Sonic16000 候选

```bash
bash /userdata/wbc-teleop-tools/scripts/test_sonic_16000.sh
```

模型为 `model_step_016000_curated_v2_closed_loop_g1.onnx`，SHA256 为 `bd9b7286f77a4b9bad670535ca2a5a4a0fadf007e25a902384d5d60ed3f94a0e`。使用独立的 `trials/tracking_20261106/Sonic16000/` 配置，参考动作及完整录制选项与 9700、20000 相同，观测合同、PD、动作缩放及原生踝控制方式与 20000 一致。日志保存为 `outputs/tracking_20261106_Sonic16000_*.jsonl`，本地采集器会识别并归档到 `hardware/Sonic16000/`。末尾加 `check` 可执行 100 步无动作检查。

## MorphMimic 全源 V3 72000 候选

```bash
bash /userdata/wbc-teleop-tools/scripts/test_morphmimic_72000.sh
```

模型来自 `vita02a_full_v3_72k_onnx_20261106.tar.gz`，SHA256 为 `ffa5f85b1f676ffda4712190aefcdbdbf90aa58c28e8811b9995727cfacff8ba`。使用交付包的配套配置，独立保存在 `trials/tracking_20261106/MorphMimic72000/`。接口为 MorphMimic 1600 维观测、28 维动作；腰 pitch PD 为 100/6.3661977，踝 A/B 电机 PD 为 30/1.5，保持原生 `motor_theta` 路径。该模型须通过 MorphMimic 入口运行。

参考动作与前述候选相同，01–09、11 每段最多 10 秒，含插值共 126.92 秒，默认完整记录观测和反馈。日志为 `outputs/tracking_20261106_MorphMimic72000_*.jsonl`，本地采集器按指纹归档到 `hardware/MorphMimic72000/`。末尾加 `check` 可执行 100 步无动作检查。包内 128 组输入输出样例已在本地及板端原生 ONNX 后端复核；这些样例验证数值推理，配合相同的特征合同检查，不代表已经完成训练工程端的完整输入构造对比。整套仿真使用不带夹爪的闭链模型和新模型 PD，通过后才允许该入口进入 live。
