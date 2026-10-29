# Vita02A / S100 actor + Pico

部署链路：Pico → **motion-stream-bridge Vita02A GMR** → Vita02A 关节参考 →
已审核的关节参考策略模型 → SDK 0.2.1 Body31。GMR 的算法、模型和参数固定自
`763f560`，已移除自写姿态 IK；它与 HoloMotion 的 SMPL 输入双策略是不同的链路。

**当前交付已合入 `main`，见已合并的 [PR #1](https://github.com/yunhao-crypto/wbc-teleop-tools/pull/1)。**
代码、21 个 ARM wheel、Pico 服务/绑定、GMR 模型与配置、测试资源均在 Git 中。
**策略权重不公开提交**：拉取后还需提供下列模型，以及目标 S100 已有的匹配固件、
私有 Zenoh peer 配置和有效硬件 YAML；无需原开发目录或另一个源码仓库。
本交付覆盖 `scripts/start_actor_pico.sh`；保留的历史 HoloMotion/P1/P2/P3 入口未随包提供权重。

## 适用环境和验证范围

- S100 本机：Linux aarch64、Python **3.10**、glibc ≥2.35、venv/ensurepip、Bash、`ip`、`flock`、`ldd`。
- 已验证的软件基线：机器人 `1.6.0-200` / `c9b4e80c074a8898d1290927a1be5350bc725d4a`，
  Aorta/messages `2026.10.20`，SDK `0.2.1`。Body31 协议 2、roster 1，28 策略关节加 2 颈关节。
  其他机器人发行版须先核对接口和有效配置；不能仅凭包日期选择 wheel。
- 目标机提供 `/app/script/env.sh`、部署自有 `/app_param/zenoh/s100_session_peer.json5`
  和完整 A-sample 有效 YAML。当前无夹爪配置应与真实硬件一致，不自动修改末端选择或将执行器设为 MOCK。
- 通过的交付/只读检查不等于运动验收。**真实 Pico 跟随质量、无支撑站立稳定性仍未完成验收。**
  当前应按有人看护、有可靠支撑的试验部署使用。

系统库明细、版本与回退见 [离线交付说明](deploy/actor_s100/README.md)。

## 首次部署：以下命令在目标 S100 执行

在 `/userdata` 下使用新目录，保留旧部署以便回退；`/root` 可能随机器人系统切换而丢失。
有网络时按下面获取，离线时将相同 Git 版本的完整归档复制到目标机。

```bash
git clone --branch main --single-branch \
  https://github.com/yunhao-crypto/wbc-teleop-tools.git /userdata/vita_actor_pico
cd /userdata/vita_actor_pico
git rev-parse HEAD
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --policy sonic --model /absolute/path/policy.onnx
# 使用 SuperMimic 时再安装它的模型（同一目录可重跑，已有 venv 会续用）：
python3.10 scripts/bootstrap_actor_s100.py --policy supermimic --model /absolute/path/supermimic_vita02a.onnx
bash scripts/start_actor_pico.sh sonic check --steps 100
bash scripts/start_actor_pico.sh sonic state-check --seconds 5
# 头显已完成全身追踪标定并点击 Send 后，运行只记录结果的影子推理：
bash scripts/start_actor_pico.sh sonic shadow --seconds 30 --human-height 1.75
```

**策略必须显式选择，没有默认值，也不会改用另一个策略。** 第一个参数写 `sonic` 或
`supermimic`（等价于 `--policy sonic|supermimic`）；在终端里不写时会弹出菜单要求选择，
直接回车或非交互运行都不会启动任何东西。所选策略的模型未安装或哈希不符、加速 GMR
依赖版本不符时直接报错退出，不会退回 sonic 或较慢的原版 GMR。

模型版本、文件大小、SHA256 和安装目标路径统一记录在
[`config/actor_pico.json`](config/actor_pico.json) 与
[`deploy/actor_s100/manifest.json`](deploy/actor_s100/manifest.json) 中。
示例使用通用文件名 `policy.onnx`；安装器按交付清单校验模型内容并复制到配置指定的位置，
随后离线解析全部 wheel 依赖、创建独立 `venv`，解包 Pico 运行库，检查 SDK/绑定/动态库，并运行无动作模型检查。
不会启动机器人服务或发布动作。不同的已有模型、第三方虚拟环境和运行库会被保留并报错；
升级请换新目录。只有重编译 Pico 绑定才需 `--build-pico` 和 g++/Python 开发头文件。

## 为本机生成无参数启动配置

安装和只读检查通过后，指定**本机实际 IP**与**运行中实际生效的完整 YAML**：

```bash
bash scripts/start_actor_pico.sh configure \
  --robot-id '<local S100 IPv4>' \
  --robot-config /absolute/path/to/running/effective.yaml
```

该命令只写入本机 `outputs/actor_pico_local.json`（0600），不连接控制、不授权运动。
它绑定机器标识、IP、末端选择和 YAML 摘要；相同配置可重跑，修改已有配置须用
`--replace` 并自动保留私有备份。不要复制其他机器的配置或私有凭据。

## 现场授权后的支撑试验

**下面的无参数命令会运动全身。** 使用前须针对本机确认可靠支撑、区域清空、有人看护和急停可用。

```bash
bash scripts/start_actor_pico.sh sonic        # GR00T G1 (SONIC) 策略
bash scripts/start_actor_pico.sh supermimic   # SuperMimic 策略
bash scripts/start_actor_pico.sh              # 终端菜单选择 1) sonic 2) supermimic，无默认
# 身高默认 1.75 米，其他操作者可覆盖：
# bash scripts/start_actor_pico.sh supermimic --human-height 1.68
```

启动时终端先打印 `ACTOR_POLICY: <策略>`，`ACTOR_LIVE_REVIEW` 中也包含 `policy` 与模型哈希；
日志头记录 `policy` 和 `gmr_engine`。

流程为等待阻尼结束 → FIXED_STAND → EXTERNAL → 持续 ONNX 站姿推理并等待 Pico。
双扳机同时按过 0.7 切入跟随，释放后再次同时按下返回 ONNX 站姿。
`STOP`/Ctrl-C 尝试 CANCEL → PASSIVE；PASSIVE 下承重肢体可能下落，软件退出不能代替急停。
EXTERNAL 没有自动到期释放；进程故障也不能假定控制权已释放。
可用 `--seconds 5` 限定客户端试验时长（如 `bash scripts/start_actor_pico.sh supermimic --seconds 5`）；故障后不自动重试运动。

控制线程采用 RR 优先级 1，需要相应 Linux 权限；GMR 子进程保持普通调度。
100 ms 机器人反馈、600 ms Pico 新鲜度及 SDK 控制权/看门狗检查保留。
详见 [完整运行说明](README_ACTOR_PICO.md)。

## 维护者验证提交内容

可只检查文件完整性：`python3 scripts/bootstrap_actor_s100.py --verify-only`。
该检查不验证单独交付的模型，也不代表连接或运动验收。

完整干净导出检查在 Linux x86_64 主机执行，需要 g++；主机依赖与目标 ARM wheel 分开：

```bash
python3.10 -m venv /tmp/vita-actor-host-check
/tmp/vita-actor-host-check/bin/python -m pip install -r deploy/actor_s100/requirements-host-check.txt
# 暂存本次修改后更新清单，再验证实际 Git 暂存区导出：
git add <intended-files>
python3 scripts/seal_actor_delivery.py
git add ACTOR-PICO-MANIFEST.json deploy/actor_s100/manifest.json
/tmp/vita-actor-host-check/bin/python scripts/verify_actor_delivery.py \
  --host-inference --model /absolute/path/policy.onnx
```

验证不借用开发目录的 `venv`、`build`、`vendor_packages`、`reference` 或外部源码树。
它验证 ARM 依赖闭环、原始 C++ 快照实现、GMR 对照、控制故障回归和 100 步推理/回放，
结果写入 `outputs/actor_delivery_validation.json`。真实绑定、只读连接、Pico 质量和运动须分别报告。
