# WBC 身体跟随与 Pico 双侧键抓握

工作分支 `feat/pico-brainco-grip-20261105`。SDK 源码基线是
`feat/sdk-shared-end-effector-client-s91220` 的 `c573e3289d68f2b944af59b98e3f06bb304501e1`，
与 2026-11-05 在机器人 10.100.23.42 的 `/userdata/sdk-demo` 读到的部署版本相同。
该 SDK 是开发候选，不代表正式发布；此文档描述本次开发验证，不授权替换机器人 SDK 或固件。

## 控制链路

原 WBC/ONNX 身体循环和双手 `EndEffectorClient` 共用同一个 `PullController`，
同一 runtime_id/control_id/session 和递增命令序列。身体为 BodyCmd/BodyState 协议 3，
控制协议 4，EE 为新端点协议。原 `HandClient` 已移除，没有旧协议回退。
身体目标显式声明 30 个物理关节，waist_roll 必须是零能力 MOCK 占位；其他身体关节不能 MOCK。
最终提交保留 SDK 的运行实例、所有权与权限检查。

双侧 grip 都大于 0.7 并持续 50 ms 后发送抓握目标；任意一侧降到 0.7 以下发送张开目标。
双侧键状态来自同一个 Pico 数据包，与双食指扳机切换身体跟随独立。
启动或输入失效后，必须先松开双侧键才能重新抓握。输入/包时间戳 200 ms 不更新时切回张开。
双手线程以 50 Hz 发布，身体循环不等待手部发布。左右手顺序提交，不是硬件原子同步。

采用 SDK 的原始位置样本和接收时间；纯遥测更新不刷新位置新鲜度。
最新状态的有效权限位必须允许输出；有效故障、位置过期或 SDK 异常终止本次控制，走 CANCEL。
退出前尽力发送一次张开目标；急停、失联或禁止输出时不保证物理回位。
进入 live 会发送配置的张开目标，实际回位速度由配置限定。

## 配置与上机准备

复制 `config/pico_brainco_grip.example.json` 为设备配置。示例是小幅、低速的初测目标，
不是已经标定的完整抓握。按实际 EndEffectorInfo 核对六轴名字、顺序和关节角域，
位置为 rad，速度为 rad/s。型号、模式、字段能力、有效限位和非 MOCK 检查在进入控制前执行。

必须使用本分支的 Pico C++ 绑定，旧二进制没有 `get_controller_grip_snapshot`，启动会拒绝。
准备好独立测试目录和运行依赖后，在 S100 构建：

```bash
bash scripts/build_pico_arm64.sh
```

本次 wheelhouse 已换为与 c573e328 对应的 SDK 和 gc783514130bb 消息包，
`config/sdk021-wheel-guard.json` 同时锁定 Python 源码、ARM64 原生库和 BFBS。
不可用旧 SDK 的 venv，也不可只改 PYTHONPATH 绕过校验。
原机器人部署、私有配置和模型保持不变；新测试目录需按既有部署流程准备依赖及相同模型。
若使用 bootstrap 构建测试目录，必须选择 `--build-pico`，不能使用历史预编译 Pico 包。

在已核实的 WBC `live --manual-pico` 命令末尾加：

```bash
--hand-grip-config /absolute/path/pico_brainco_grip.json
```

保留原 policy、robot-id、robot-config、group 和现场运动确认参数；auto-start 会推导 manual-pico。
不传此参数时不创建 EE 客户端，但本分支的身体路径仍要求新 SDK。
stand-only 不支持此参数。不要同时启动另一份 SDK 运动示例来争用会话。

## 本地源集成测试

SDK worktree：`/host/humanoid-sdk-pico-ee`，原 `/host/humanoid-sdk` 主工作区未修改。
测试使用真实 SDK、真实消息 wheel 和本地编译的 C++ Body/EE 解码器；机器人传输为模拟。
本地 `build/sdk-ee-bindings` 是校验 SHA-256 后解包的匹配消息包和 x86 Aorta 依赖。

```bash
python scripts/build_sdk_source_test_native.py --sdk-root /host/humanoid-sdk-pico-ee
PYTHONPATH=/host/humanoid-sdk-pico-ee/python:/host/humanoid-sdk-pico-ee/tests:build/sdk-ee-bindings \
  python tests/pico_ee_sdk_test.py
python tests/pico_hand_grip_test.py
python scripts/state_test_pico_snapshot.py
PYTHONPATH=src:build/sdk-ee-bindings:venv/lib/python3.10/site-packages \
  build/test-venv/bin/python -m unittest discover -s tests -p test_actor_pico_live.py -q
```

本地 venv 源码链接仅供离线源码测试，不能复制上机代替 ARM64 SDK。
SDK 源集成测试与实机部署验证分开；本地通过不能证明真实 Pico 延迟、机器人抓取效果或物理安全。
首次实机应无负载小幅测试：单侧不抓握、双侧抓握、任一松开张开，
身体跟随同时工作，输入断流和恢复释放门控正确。
