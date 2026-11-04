# Pico 双侧键控制强脑手

开发 worktree：`/host/wbc-teleop-tools-pico-brainco`，分支 `feat/pico-brainco-grip-20261105`。
基于原 Pico 仓库本地 HEAD `37bc5b1`，包含其未推送的 9 个提交。

## 行为

显式传入 `--hand-grip-config` 启用。两侧 grip 均大于 0.7 并持续 50 ms 后，双手以 50 Hz 接收抓握目标；任一侧降到 0.7 以下立即切回张开目标。启动及输入中断恢复后必须先松开两侧。输入或包时间戳超过 200 ms 不更新，切回张开目标。实际到位速度由 `speed_rad_s` 限定。双食指扳机仍控制身体跟随开关。

手部反馈失效、输出被禁止或发布异常会终止本次身体控制，走原有 CANCEL 流程。退出前尽力发送一次张开命令；失联、急停、硬件禁止输出时不能保证物理回位。左右手是顺序发布，不是硬件原子同步。

## 测试准备

1. S100 运行包含 BrainCo 支持的 vita-robot 固件，并使用实际设备的 BrainCo 配置与身份校验参数；参考硬件配置中的 `P0_REQUIRED` 不能直接用于实机。SDK 必须提供 `locomotion_aorta.hand.HandClient`。
2. 新 worktree 不包含旧目录中未跟踪的 venv、模型、vendor runtime、设备私有配置等。沿用项目的 S100 bootstrap/deploy 流程准备运行依赖；不要把原目录的旧 Pico 二进制当作新绑定。
3. 在准备好依赖的 S100 新目录内重新构建：

   ```bash
   bash scripts/build_pico_arm64.sh
   ```

   构建脚本会验证 `get_controller_grip_snapshot`。旧绑定会在启动时被拒绝。
4. 复制 `config/pico_brainco_grip.example.json` 为设备配置。示例是低速、小幅闭合的初测目标，并非已校准的完整抓握。根据实机 HandInfo 核对六个关节的名字、顺序和限位；角度为 rad，速度为 rad/s。程序对型号、顺序、目标限位与速度上限进行检查。
5. 在原来已验证的 `live --manual-pico` 启动命令末尾追加：

   ```bash
   --hand-grip-config /absolute/path/pico_brainco_grip.json
   ```

   保留原来的 policy、robot-id、robot-config、group、confirm-motion 等参数。此选项不支持 stand-only 或非 manual Pico 模式。未传入选项时不创建手部客户端。

## 明天验收

先无负载小幅测试。进入 live 会发送配置的张开目标。确认单侧按下不抓握、双侧持续按下抓握、松开任意一侧张开、持续按住不反复切换。随后验证双扳机原有功能、Pico 断流后的张开目标、恢复连接必须先释放两侧。根据实际手型调整 grasp_rad，再做抓取测试。

离线验证：

```bash
python tests/pico_hand_grip_test.py
python scripts/state_test_pico_snapshot.py
```

这些测试不证明实机身份、SDK/固件版本兼容性或抓取效果。尚未进行 S100 ARM64 绑定编译和实机试验。
