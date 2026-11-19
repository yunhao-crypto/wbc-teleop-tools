# 新增 pico_smpl / pico_teleop 两条输入路径

本变更在原 actor 项目上增加两个可选策略入口：

| 入口 | 输入 | 当前模型 |
| --- | --- | --- |
| `pico_smpl` | Pico 人体 XYZ、root 朝向、明确的站立腕部参考 | SMPL 35000 |
| `pico_teleop` | 校准后的双腕与头部三点参考、固定下身参考 | Teleop 35000 |

原 `sonic`、`supermimic`、`morphmimic` 的默认模型、配置、推理实现及普通
base/planner 构建保持原行为。35000 G1 的默认模型替换不属于这个两路径扩展。
旧文件名中的 `27000` 仅保留为路径兼容，模型身份由 SHA256 和检查点字段确定。

## 复用边界

五个策略使用同一个 `actor_pico_live.run_live`。新增路径通过显式参考接口提供
输入获取、heading 对齐、站立参考和模型观察量；启动、所有权、发布和取消仍由
原主循环处理。原策略使用默认 `JointReferences`，新增路径使用 `TaskReferences`。

入口仍为 `actor_pico.py` / `start_actor_pico.sh`，安装仍用
`bootstrap_actor_s100.py --policy ...`。没有额外的独立 live runner、BVH
导入工具或虚拟 workflow。新增模型使用原 encoder、FSQ
与共享 decoder；完整 ONNX 用于哈希与输出一致性验证。

## 构建与安装

原三条策略继续使用原部署目录、venv、SDK 0.3.0 和原 SDK 校验值。
显式传入 Pico 资产时，构建独立的 `actor-s100-pico-*` 包，只包含两条新增路径，
使用已固定的 `1.0.0rc1+bodycmdv4.6863ad7` SDK。它不包含原三策略的配置，
也不改写它们的 SDK 身份或依赖。

两个 SDK 的 Python 包名及原生库名相同，因此 Pico 包必须解压到一个全新目录，
使用自己的 venv。不得覆盖原部署目录或共用原 venv。安装器会在 pip、解包和
凭据更新之前拒绝复用旧 venv；原三策略也不能装入标记为 Pico R4 的 venv。
切换 SMPL 与 Teleop 时共用这份 Pico venv，原三策略始终从原部署目录启动。

```bash
python scripts/build_actor_delivery.py --variant base --pico-assets /absolute/pico-assets
```

`deploy/pico_27000/delivery_inputs.json` 明确列出所需模型、neutral 参考、离线
输入和 R4 wheel 的路径及哈希；不复制资产目录中的其他内容。旧
`--pico27000-assets` 是同一选项的兼容别名。未提交代码的候选包标记为 `dirty`。

将 Pico 包解压到全新目录后，在该目录按原入口安装：

```bash
source /app/script/env.sh
python3.10 scripts/bootstrap_actor_s100.py --policy pico_smpl --verify-only
python3.10 scripts/bootstrap_actor_s100.py --policy pico_smpl
bash scripts/start_actor_pico.sh configure --robot-id <S100-IP> --robot-config /absolute/effective.yaml
bash scripts/start_actor_pico.sh pico_smpl check --steps 240
```

新路径复用原安装器的实现，包括 pip、解包、ABI 和动态库检查，仅增加两路径的 ORT/SDK
模型检查及 BodyCmd4 字节序列化检查。安装和 live 共用 `build/operator.lock`。
新增路径的 live 在持锁后检查安装凭据和加载模型；失败安装不能保留其旧成功凭据。

现场确认有效配置、通信与支撑条件后，真实控制入口为：

```bash
bash scripts/start_actor_pico.sh pico_smpl
bash scripts/start_actor_pico.sh pico_teleop
```

原流程：站立准备 → 策略站立接管 → 等待 Pico → 双扳机跟随 → 回站立／丢流处理。
退出会请求 CANCEL/PASSIVE。`shadow` 和 `replay` 仅执行无动作推理，自动创建输出目录。

## 模型与控制契约

保持导出物的 `motor_theta`、原生踝 A/B 目标、原踝 IK/Jacobian、PD、动作范围、
腿部反馈和腿部输出。SMPL 使用已接收的 180ms 参考窗口及 100ms 回放裕量；
输入新鲜度为 100ms。Teleop 的参考 heading 在启动时固定，暂停或丢流不重置。
两条路径固定颈部；SMPL 的六个腕部关节参考取模型站立值。

2ms 接收进程为每次 press 携带对应 release 的时间。丢流恢复边界不得倒退；
重新跟随需要恢复边界之后的 release → press，旧缓存帧不能恢复输入。
本代码不修改机器人 YAML、固件或系统服务。

## 离线验证

```bash
python -m unittest discover -s tests -p 'test_pico_aligned_live.py' -v
python -m unittest discover -s tests -p 'test_pico_trigger_recovery.py' -v
python -m unittest discover -s tests -p 'test_pico_install_modes.py' -v
python -m unittest discover -s tests -p 'test_pico_checkpoint.py' -v
```

测试覆盖真实 `run_aligned` → 原 `run_live` 调用链、故障退出、恢复扳机、固定
heading、安装互斥、SDK 环境隔离及原默认模型／构建边界。默认入口的预热仍调用
原 `runtime.infer_motion(runtime.hold_motion(), ...)`；原配置变更注入测试保持有效。
模型和 controller 的模拟与实机分开。
`check_pico_aligned.py` 仅证明给定观察量时网络输出一致，不证明训练输入语义、
现场 Pico 接收或物理稳定性。实机仍沿用暂停状态。
