# robot67：控制时序、观测契约和 SDK / exec 对比

这些工具**默认纯离线**，不连接机器人、不创建控制句柄、不发布动作、不修改模型、增益或 locomotion 配置。`--audit-timing` 仅为已有 WTT live 程序增加可选记录，默认关闭。没有实现或部署新的 `locomotion exec` 推理器；下面的协议供 exec 实现者导出真实数据。

**任何报告都不是运动安全许可。** 现场人员负责支撑、防跌倒、急停、场地和退出确认。不要为了采集数据跳过原有 live 门槛。不要用不同人的 Pico 动作当配对实验，也不要把未观察到的 echo 当丢包。

## 1. 离线环境和现有日志

在分析工作站的独立环境运行，不升级正在工作的机器人或训练环境：

```bash
python3 -m venv .venv-audit
.venv-audit/bin/python -m pip install -r requirements-control-audit.txt
PY=.venv-audit/bin/python
$PY scripts/audit_actor_control.py /data/sdk.jsonl \
  --onnx /data/exact_deployed_wrapped_policy.onnx --output /data/sdk.audit.json
```

必须提供**实际部署的 wrapped 单输入 ONNX**，不能用原始多输入训练图代替。审计的数值重放当前限定 MimicLite V2 `[1,1224] → [1,28]`；无 `--onnx` 仅做记录/时序审计。未请求重放、缺字段、旧日志缺时钟声明等均保留为 `PARTIAL`，不是“默认通过”。`INVALID` 返回码 2，`PARTIAL` 可能返回 0，自动化必须读取 JSON 状态。

检查项目：
- 实际文件 SHA256 与 header 模型身份；CPU ORT 对所有记录的原始动作重放。
- default + scale × raw action、24 个串联关节的实际发送目标/Kp/Kd；四个并联踝不混用 motor_theta 和 virtual_pitch_roll。
- 10 帧 q/dq、gyro、gravity 历史，以及 3 帧**原始动作**历史。每个 runtime/session 前 10 帧明确排除。
- 同一原始命令的 source→first-output 匹配、重复 echo、覆盖率、源状态更新间隔、9 个历史间隔跨度、分阶段耗时、超期/跳帧事件。
- 协议 3 用 runtime/source-type/有效位区分来源；数值为零的 key 不自动等于无效。旧日志没有这些字段时，不追溯伪造为协议 3 完整证据。

`first-output` 是固件成功发送后锁存的首个影响输出周期，不是 CAN 完成、实际电机响应或机械稳定时间。跨 host monotonic / robot realtime / 传感器硬件时钟不能直接相减。旧 `source_age_ms` 是采集器报告值，不是独立校时结果。

## 2. 下一次由现场人员采集

先固定同一模型、NPZ、参考处理、body profile、有效 locomotion 配置和增益，保存 Git revision、文件 SHA256、SDK/wire schema 身份。现有模型配置仍按原来的 pin 工作，本次提交不会替换它。

安全的已有离线检查例子：

```bash
# 使用已有部署环境；检查不发布命令。
python scripts/actor_pico.py motion-check --policy mimiclite --backend auto \
  --motion-config /data/fixed_motion.json --record-observations --log /data/motion-check.jsonl
```

检查通过后，**由现场人员另行审批和执行已有的固定 NPZ live 流程**（见 `README_ACTOR_PICO.md`），只追加：

```text
--audit-timing --record-observations --log /data/sdk-fixed.jsonl
```

这里有意不提供一条可直接复制的自动实机启动命令。不变更 `--robot-id`、`--robot-config` 等原有现场确认；不启动第二个控制所有者。记录整个准备站立→lead-in→clip→end-hold→退出，不只截“最好看”的区间。同时使用现场已有 MCAP 记录流程保存原始 BodyState/命令回声及 schema。

新记录包括：
- `header.audit_timing`：单位、时钟声明、采集范围；`control_path="sdk"` 是发布路径，不等于 `--backend sdk/ort`。
- `control_timeline.events`：feedback_read、observation_begin/end、onnx_begin/end、beforepublish、publishreturn，全部 host monotonic ns。
- `sensor_timing`：SDK 已有的 joint/IMU 原始时间戳数组、状态/存在掩码和原始 IMU 值；缺失写 null，非有限原始值写 null。`imu_values` 的存在不代表物理有效或已校准。
- `feedback` 中可用的 protocol/runtime/association 有效位，及 SDK EXTERNAL `command_source_type=1`。
- 固定 NPZ 的 `motion_reference.joint_velocity`：实际使用的 dense-source virtual rad/s，不从稀疏未来帧重新差分。

记录的是**成功发布命令的循环**，不是所有伺服周期；失败、跳帧、超期还要看单独事件和原始 MCAP。时间戳采集及 JSON 序列化有开销，必须在现场比较开关前后的最大耗时/超期数；A/B 应保持一致的记录级别。旧 `publish_ms` 包含发布后少量整理开销，精确 SDK 调用区间应看 timeline 的 beforepublish→publishreturn。

## 3. 同输入 shadow：先验证观测，再验证动作

从新采集的固定 NPZ SDK 日志生成独立只读测试输入；输出文件已存在时拒绝覆盖：

```bash
$PY scripts/actor_control_shadow.py prepare /data/sdk-fixed.jsonl \
  --onnx /data/exact_deployed_wrapped_policy.onnx --output /data/frozen-input.npz
```

fixture 是 `wtt.shadow_fixture.v1`，包含：

| 字段 | 形状/意义 |
|---|---|
| metadata_json | 非 pickle JSON 标量；模型/NPZ/源日志 hash、body_order、reference_joint_names、future_steps |
| frame_id, source_state_us | N 个 uint64，保持精确整数 |
| q_body, dq_body | N×body_slots，按 body_order；SDK virtual body 坐标 |
| root_wxyz, gyro | N×4、N×3；实际消费的传感器数据 |
| reference_q_virtual, reference_dq_virtual | N×8×28，按 reference_joint_names；virtual rad / rad/s |
| reference_root_position, reference_root_wxyz | N×8×3、N×8×4；尚未应用 yaw_offset 的参考 |
| yaw_offset | N；参考对齐偏航 |
| teacher_raw_action | N×28；**用于每一步推理后更新动作历史** |
| expected_observation | SDK 实际构建的 N×1224；只用于校验，禁止当 native 输入代替重建 |
| evaluation_mask | 前 10 行 false，其余 true；补齐初始历史 |

exec 端应在**无控制句柄、零动作发送**的离线模式逐帧消费 q/dq/gyro/root 与选定参考，按相同命名/默认值/Jacobian/历史顺序构建 observation，使用同一 wrapped 模型。不要用候选输出推进下一步动作历史；必须 teacher-force `teacher_raw_action`。不能对这 8 个稀疏参考帧自行差分速度。它验证的是传感器+选定参考→观测→动作，不覆盖原始 NPZ 加载/参考选择，更不是闭环实验。

exec 导出的 `.npz`（`allow_pickle=False` 可加载）必须含：

```text
fixture_sha256: string scalar，实际 frozen-input.npz 的 SHA256
model_sha256: string scalar，实际模型文件 SHA256
control_path: string scalar "locomotion_exec"
actions_sent: integer scalar 0
control_handles_created: integer scalar 0
frame_id: integer [N]，与 fixture 完全同序、无删帧
observation: float [N,1224]，候选实际构建值
raw_action: float [N,28]，候选实际推理值
```

```bash
$PY scripts/actor_control_shadow.py compare /data/frozen-input.npz /data/exec-shadow.npz \
  --output /data/shadow-comparison.json
```

默认 obs atol=1e-6、action atol=5e-5。`NUMERICAL_PARITY` 仅为数值一致；更改容差会记录在报告里，不能为“通过”任意放宽。不要伪造零发送/来源声明或把 expected_observation 直接复制成候选结果。

## 4. 固定 NPZ 硬件 A/B：exec 端日志协议

只有 shadow 通过、原有安全检查通过且现场再次确认后，才由现场人员分别运行 A/B。建议在相近初态下重复 A/B/B/A，保持地面、支撑、负载、温度等条件；不能只挑一个窗口下因果结论。

exec 日志沿用 WTT JSONL 字段，不要求使用 Python/SDK 控制客户端：
- 一个 header：`control_path="locomotion_exec"`，`scope="supported_exec_trial"`，policy、robot_id、model.sha256、reference_source="npz"、motion、contract、head_control、comparison_config_sha256。
- `motion` 至少含 sha256、policy_frames、period_s、joint_names、joint_space="virtual_pitch_roll"、limit_corrections、end_hold_s、root_body_index；也应记录 velocity_source/units、source_control_space 等实际处理配置。
- 一份 `robot_config.ankle_path_comparison_sha256` 收据。沿用 WTT `comparison_config_sha256` 的 canonical JSON 算法（配置去掉 body_profile.ankle_command_space，sort_keys=True）及已有有效机器人配置比较收据算法。**必须依据实际生效配置重新计算，不能复制 SDK hash 冒充验证。** 若 exec 配置结构不同，应先设计并评审共同语义收据，现工具会保守拒绝，不静默忽略差异。
- 一个 motion_started，含 lead_frames。
- 每次实际推理/发布的 actor_inference：stage、motion_phase、motion_tick、policy_tick、body_order、observation、raw_action、decoded、body_targets、sent_body_targets、feedback、feedback_timing、compute/inference/observation/publish_ms、source_state_us、实际 session_id/sequence；不是预测“应该发出的”值。
- `feedback` 的 q_body/dq_body 使用与 SDK 同名 virtual body 坐标；root_wxyz/gyro、applied_session_id/applied_sequence/ts_first_output_us；协议 3 另记录 protocol_version/runtime_id/applied_key_valid/association_source_type。命令端记录实际 runtime_id、command_source_type，不能把零 key 当丢失。缺字段用 null，不造时间戳。
- `motion_reference`：实际选定窗口的 virtual q、base、root_wxyz、ref_time、yaw_offset；速度字段同第 2 节。不得混入原始未消费参考冒充实际输入。
- timeline 若提供须使用同版 schema，声明 exec 自己的时钟；不能把外部采集器接收时间当推理时间。
- 最后一个 summary：mode="live"、stop_reason="motion_complete"、exit_confirmed=true、actions_sent 等真实结果。异常/中断不伪造完整结束。

```bash
$PY scripts/audit_actor_control.py /data/exec-fixed.jsonl \
  --onnx /data/exact_deployed_wrapped_policy.onnx --output /data/exec.audit.json
$PY scripts/compare_actor_control.py /data/sdk-fixed.jsonl /data/exec-fixed.jsonl \
  --onnx /data/exact_deployed_wrapped_policy.onnx --output /data/ab.json
```

工具拒绝 Pico/同路径/不同模型或 NPZ/缺完整 clip/配置或实际增益不一致；核对实际消费的参考位置、关节位置/速度和姿态。默认排除 clip 前 2s；初态匹配默认关节差≤3°、根姿态差≤3°、速度差≤0.5rad/s，**只是配对筛选阈值，不是安全限制**。实际发送的 dq/tau 前馈也必须一致；目标-参考误差只报告串联关节，绝不拿踝 motor target 减 virtual reference。

`COMPARABLE_RECORDED_CONTRACT` 仅说明记录契约可比，不说明接触、负载、执行器内部状态完全相同。报告观测到的年龄/跟踪变化，不自动归因于“绕过 SDK”。

exec 必须另外记录并检查：推理的最坏耗时、deadline miss、队列积压/丢弃/覆盖、历史推进时机、超期后沿用/丢弃/退出行为，以及**底层伺服循环**最坏耗时。不得把 ONNX 阻塞调用直接塞入硬实时伺服线程后，仅凭平均 1ms 就判定安全。当前工具不实现或证明这些调度保障。

## 5. 原始 MCAP 与 IMU

提取只读取文件。使用与记录匹配的 SDK 生成绑定；需要 schema SHA pin，且保留全部原始 MCAP。默认验证支持的 CRC，不把缺 CRC 等同已验证。

```bash
$PY scripts/extract_actor_body.py /data/run/mcap/*.mcap \
  --actor-log /data/sdk-fixed.jsonl --sdk-python-path /data/matching-sdk/site-packages \
  --schema-sha256 VERIFIED_BODY_STATE_SCHEMA_SHA256 --output-dir /data/extracted
```

输出目录必须不存在；生成 body.npz、imu.npz、metadata.json。所有 uint64 时间戳/序号保持整数；缺失有 field_present/present/finite 掩码。记录输入/绑定文件 hash、协议、实际 schema 和 CRC 范围。

IMU 按真实 sample key 去重；无有效 key 的记录不会全部挤成一个假样本。同 key 但内容不同保留并标记 `identity_contradiction`；退出码 **3** 表示检测到这种矛盾，仍生成数据和收据，不是“干净提取通过”。姿态估计可能独立于原始加速度采样更新；这种标记本身不能证明通信损坏或振动原因。

**body_realtime_us 关联到 IMU 只是 PROXY。** sampled/acquired/ingested/assembled 可能属于不同设备时钟。没有协议/校时证据，不计算跨域年龄，不把 BodyState 发布率冒充 IMU 采样率。

可选 IMU 频带分析需另写 `wtt.imu_analysis.v1` 元数据，例如（先验证约定，不要盲照抄）：

```json
{
  "schema":"wtt.imu_analysis.v1",
  "evidence":"填写 SDK/传感器配置和坐标校验依据；若仅操作者声明，明确说明未独立校准",
  "time":{"key":"body_realtime_us","unit":"us","domain":"robot_realtime","semantics":"body_state_proxy","zero_is_missing":true},
  "acceleration":{"key":"accel_body","unit":"m/s^2","frame":"body"},
  "quaternion":{"key":"quaternion_wxyz","order":"wxyz","direction":"body_to_world"},
  "gyro":{"key":"gyro","unit":"rad/s","frame":"body"},
  "world_up_axis":"z",
  "sensor":{"selector_key":"location","value":0,"status_key":"status","valid_status":1,"unique_samples":true,
    "require_boolean_fields":{"identity_valid":true,"identity_contradiction":false}}
}
```

```bash
$PY scripts/audit_actor_imu.py /data/extracted/imu.npz --metadata /data/imu-analysis.json \
  --start-s 2 --end-s 20 --output /data/imu-window.json
```

时间窗口相对于**筛选后第一个有效时间戳**，不是 actor 日志自动对齐的原点；报告保存精确原点。选择真正采样时钟须提供对应 domain/units/semantics 的证据。工具拒绝非单调/重复时戳，不排序掩盖复位；跨 >50ms 缺口分段，至少 4s，Hann FFT、不补零，返回覆盖率及 2–8Hz 世界竖直加速度 RMS、可选 quaternion/gyro 一致性。仅 BodyState 代理时钟时结果明确标为代理；不足覆盖返回 null 而非 0。不同窗函数、分段或重采样方案的 RMS 不能与历史报告直接拼接比较。

HIP 传感器点加速度不等于仿真根原点加速度，姿态误差可把重力泄漏进频带；不能双积分这份数据宣称抖动位移，更不能以此独立证明执行器原因。

## 6. 回传清单与本地验证记录

回传：完整 JSONL（含 header、异常、summary）、固定 NPZ+配置及 SHA、实际模型 hash/导出元数据、有效 locomotion YAML、Git/SDK/wire schema 身份、原始 MCAP、工具输出 JSON/NPZ、shadow fixture/候选 trace、采集开关、现场支撑/负载/地面/温度/急停情况。不要只回传汇总均值或截图。

本次纯离线验证：hist10 7947 行、knee1000 9981 行完整 ORT 重放，最大误差分别 7.152557e-6 / 4.768372e-6；历史核对 7937 / 9971 行。旧 Pico 日志正确保持 PARTIAL，不能当固定 NPZ A/B。

全部 9 个原始 MCAP 提取了 54917 / 69308 个 BodyState；检测到 60 / 114 个 HIP 同 key 内容矛盾行并保留。计数覆盖整个 MCAP 会话，不等于先前选定活动窗口的样本数。本次没有实机新实验，没有 native-exec 实机结果。
