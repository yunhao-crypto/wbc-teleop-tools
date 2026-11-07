# Sonic 连续动作测试

本轮模型为 Sonic 009700，原生 A/B 脚踝配置。参考顺序为最新 `motion.rar` 的
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 11（各段源动作最多前 10 秒，02 保留完整 3.91 秒）。
第 10 条不加入。参考总时长 126.92 秒，另加播放器的站立准备、导入及结束保持。
注意：本数据集的 09 是篮球投掷，11 才是舞蹈。

板端在机器人有支撑、现场可停止的条件下，由操作员启动：

```bash
# 不连接电机控制接口的文件/推理检查
bash /userdata/wbc-teleop-tools/scripts/start_sonic_motion_sequence.sh check

# 启动一次连续播放，完整记录观测；完成后沿现有 CANCEL -> PASSIVE 流程退出
bash /userdata/wbc-teleop-tools/scripts/start_sonic_motion_sequence.sh
```

可按 Ctrl+C 走现有停止流程。脚本不会循环，也不需要 PICO Send。
记录保存在 `outputs/sonic_sequence_*.jsonl`，包括模型输入、参考窗口、推理结果、
SDK 状态和发布记录。各片段的定位见
`reference/sequences/sonic_max10/sequence_manifest.json`。
日志 `motion_tick` 扣除本次 `lead_frames` 后，对应清单里的 `start_frame/end_frame`；
源 NPZ 额外保存 `source_clip_id/source_time_s/sequence_phase`，过渡帧 ID 为 -1。

每个片段开始/结束各使用 0.6 秒余弦速度斜坡，使参考速度趋于零，内部保留原速。
这会让每个片段的播放时长增加约 0.6 秒：10 秒源区间播放约 10.6 秒。
片段间使用至少 3 秒的五次平滑曲线；若必要，按受控关节最大 1 rad/s 自动延长。
根部采用最短弧 SLERP，下一段仅作刚性 yaw/XY 对齐，保留相对位移及绝对高度。
未修改关节姿态轨迹幅度、模型、PD、EXTERNAL YAML 或虚拟限位。
整段仅有一次策略启动，跨片段不重置策略历史。原 `fixed_motion.npz` 和默认配置保持原用途。

离线准备工具：

```bash
python scripts/compose_actor_motion.py \
  --spec /path/to/spec.json \
  --policy-config config/actor_pico.json \
  --output /path/to/new-output-directory
```

依赖 NumPy 和 SciPy。输入 spec 为 `vita_actor_sequence_v1`，包含 `clips` 数组，
每项需要 `id/path/sha256`，可指定 `start_s/end_s/title`。相对输入路径以 spec 所在目录解析。
输出 `motion.json` 的路径以 actor `--root` 解析，部署时需改为对应的仓库相对路径。
工具不连接机器人；严格校验源哈希和虚拟关节限位，拒绝覆盖已有输出目录。
配置快照和整段连续仿真通过记录均与参考 SHA256 绑定，专用启动器拒绝配置漂移。

本轮连续名义仿真通过：6347 个参考帧全部完成，最大倾斜 23.44°，关节 RMSE 4.60°，无非足部地面接触。`continuous_sim_validation.json` 保存原始结果。
插值平滑只保证参考连续；单次名义仿真不保证实机稳定。需要用本次记录检查实际效果。
