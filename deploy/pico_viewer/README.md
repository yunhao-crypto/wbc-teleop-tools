# PICO 配置、相机视角与 Android 安装包

本指南从一台尚未安装 XRT 的 PICO 开始，完成安装、机器人连接、单目视频和四键切换。
机器人侧沿用 `wbc-teleop-tools` 的 `scripts/start_actor_pico.sh` 入口。

## 先准备这些条件

- 机器人已按本仓库的[首次部署说明](../../README.md)安装对应 SDK、运行环境和所选策略。
  机器人已有相机进程发布 `/image_front_raw/h265`；这里转发现有图像，不重新打开相机。
- PICO 和机器人之间可以相互访问 TCP **63901** 和 **13579**。两者可以位于不同网段，
  但网络必须有路由，不能开启阻断设备互访的 AP 隔离。
- 本次验证设备为 A9210、Android 14、ARM64、官方 XRT v1.1.1。
  APK 最低 Android 11/API 30，并要求支持 960×768@30 的硬件 HEVC 解码器。
  其他型号需先确认 PICO SDK 的控制器、全身追踪和解码支持。
- 只安装预编译 APK **不需要 Unity、Unity 账号或 Unity 许可证**。

| 通道 | 机器人端口 | 用途 |
|---|---:|---|
| XRT PC Service | TCP 63901 | XRT 界面中的“连接”，接收手柄、头部、全身位姿 |
| 相机转发 | TCP 13579 | 机器人 H.265 → PICO，眼镜主动建立连接 |
| 网络 ADB（可选） | PICO 的 5555 或配对界面指定端口 | 安装和日志诊断，不参与视频或位姿传输 |

**视频正常不代表 PC Service 已启动。** “连接失败”应先检查 63901；
“已连接但看不到相机”再检查 13579 和相机收帧。

## 一、给全新 PICO 安装应用

1. 在电脑下载并解压官方 [Android Platform Tools](https://developer.android.com/tools/releases/platform-tools)。
   以下命令中的 `adb` 来自此目录；Windows PowerShell 可写成 `.\adb.exe`。
2. 在 PICO 中进入“设置 → 通用 → 关于本机”，连续点击“软件版本号”，出现“开发者”菜单后，
   进入并打开“USB 调试”。使用可传输数据的 USB 线连接电脑，在眼镜内允许调试。
   菜单步骤参考 [PICO 官方说明](https://developer-cn.picoxr.com/document/pico-unity-quickstart/pdc-basic-info/)。
3. 在本代码目录或 Android 离线包的根目录执行：

   ```bash
   adb devices -l
   # 设备状态应为 device；unauthorized 表示还需在眼镜里允许调试。
   ```

4. 校验并安装本仓库附带的[通用 APK](releases/XRT-Vita-Camera-1.1.1-vita.2.apk)：

   ```bash
   # Linux/macOS；Windows 可用 Get-FileHash -Algorithm SHA256 对照 manifest.json。
   (cd deploy/pico_viewer/releases && sha256sum -c SHA256SUMS)
   adb install -r deploy/pico_viewer/releases/XRT-Vita-Camera-1.1.1-vita.2.apk
   ```

   若电脑同时连接多个设备，所有 ADB 命令加 `-s <设备序列号或IP:端口>`。
   输出 `Success` 后，PICO 应用库中会出现 **XRT Vita Camera**。
   包名为 `com.xrobotoolkit.client.vitacamera`，与原版 `com.xrobotoolkit.client` 并存。
   联调包使用 Android 调试签名；版本、SHA256、源码指纹见[发布清单](releases/manifest.json)。

5. 如需后续无线诊断，保持 USB 连接执行：

   ```bash
   adb -d tcpip 5555
   # 在 PICO Wi-Fi 设置中读取它自己的实际 IP，再拔掉 USB：
   adb connect <PICO的IP>:5555
   adb devices -l
   ```

   如果设备提供带配对码的无线调试，请使用页面上的配对端口和连接端口，不能假设它们是 5555。
   重启后必要时重新开启，参考 [Android 官方无线 ADB 流程](https://developer.android.com/tools/adb#wireless)。

## 二、先做连接和视频联调

以下机器人命令在 SSH 登录机器人后执行。这里假设部署入口为 `/userdata/wbc-teleop-tools`；
如果使用其他部署目录，先切换到相应目录。

```bash
cd /userdata/wbc-teleop-tools
bash scripts/start_actor_pico.sh connect
```

`connect` 模式不需要选择策略或加载权重，会一起启动：

- XRoboToolkit PC Service：TCP 63901；
- 前置相机 H.265 转发：TCP 13579。

它不创建机器人控制会话，也不执行跟随。等终端出现 `PICO_CONNECT: ready` 和
`PICO_VIDEO: ready`，再操作眼镜：

1. 停止当前跟随后，退出原 XRT，打开 **XRT Vita Camera**，按界面要求授予权限。
2. 在连接界面填写**机器人 IP**，例如 `10.100.23.173`，点“连接”。
   使用英文句点 `.`。通用 APK 的视频地址会跟随这里的地址，不固定为某台机器人。
3. 初次只看相机时可以先不启用全身发送。先松开 A、B、X、Y，再同时按住四键约 **0.5 秒**。
   两眼显示同一张完整单目图像；再次全部松开后，重复四键操作可返回第三人称。
4. 摄像头 GOP 为 30，首次接入或断流恢复可能需要等待约一秒的完整关键帧。
   无新鲜解码帧时会自动返回第三人称；相机恢复后会恢复此前选择的相机视角。
5. 在机器人终端按 **Ctrl-C** 退出连接联调。入口会关闭自己启动的两个服务并释放占用。
   随后才能在另一个入口启动正式跟随。

只做限时联调也可以：

```bash
bash scripts/start_actor_pico.sh connect --seconds 30
```

## 三、沿用原本的一键跟随方式

先按 PICO 官方流程完成所用全身追踪设备的配对和校准，在 XRT 中确认头部、手柄及
Full body 预览正常。控制器跟随还需打开相应 Controller 和 **Send** 设置。
本定制客户端把四个面键用于视角切换，**Send 请点击界面按钮**；不再用 A 键快捷开关 Send。

在已经完成本机身份、有效硬件配置和策略安装的机器人部署上：

```bash
cd /userdata/wbc-teleop-tools
bash scripts/start_actor_pico.sh supermimic
# 或显式选择另一套已安装策略：
bash scripts/start_actor_pico.sh sonic
```

这是原有的有人值守运动入口，包含原有站立准备与跟随流程。
策略必须明确选择；SDK 控制会话、头部 Kp/Kd、双 trigger 跟随切换仍由原有 actor 管理。
相机转发现在默认随 `live` / `auto` / `shadow` 一起启动；XRT PC Service 仍由 actor 自己管理，
不会再启动第二个 PC Service。按原流程等待 PICO 接收就绪，再在眼镜里连接和开启 Send。

- 正式入口退出时，先等待 actor 完成控制释放及其 PICO worker 清理，再停止入口自己启动的视频进程。
- 视频启动失败或运行中退出，会在终端报告日志位置；actor 仍按原有控制逻辑运行。
  用户仍能使用第三人称；不会因为视频进程失败自动切换机器人控制状态。
- `check`、`replay`、`state-check` 和 `configure` 不启动视频转发。
- 缺少相机或暂时不使用视频时：

  ```bash
  bash scripts/start_actor_pico.sh supermimic --no-camera
  ```

使用其他单目相机或 Zenoh 会话配置：

```bash
bash scripts/start_actor_pico.sh connect \
  --camera-topic /image_left_raw/humanoid/h265 \
  --camera-session-profile /app_param/zenoh/factory.json5
```

可选主题为 `/image_front_raw/h265`、`/image_left_raw/humanoid/h265`、
`/image_right_raw/humanoid/h265`；后两路只是替代单目视图，不是假定已校准的双目。
当前客户端要求 960×768@30。`--camera-port` 修改端口后还需同步修改客户端设备配置，
常规部署应保留 13579。相机配置默认使用 `/app_param/zenoh/factory.json5`，
不会覆盖 actor 自己审核后的 Zenoh peer 配置。

## 四、日志与常见问题

运行日志保存在部署根目录的 `outputs/`：

- `pico_camera_*.log`：视频初始化、订阅和转发日志；
- `pico_camera_*.json`：收帧、发帧、队列和连接统计；
- `pico_pc_*.log`：`connect` 模式的 PC Service 日志；
- `actor_pico_*.jsonl`、`pico_service_actor.log`：原有 actor/worker 记录。

| 现象 | 检查与处理 |
|---|---|
| XRT“连接失败” | 机器人 IP 是否正确；终端是否出现 `PICO_CONNECT: ready`；`ss -lntp 'sport = :63901'` 是否有监听。13579 正常不能替代这一项。 |
| 已连接但没有相机 | 查看 `PICO_VIDEO` 和相机 JSON，确认 `received_frames` 增长；检查 `ss -lntp 'sport = :13579'`。 |
| 按四键没有切换 | 确认打开的是 XRT Vita Camera；两只控制器均连接；先全部松开，再四键持续按住 0.5 秒；确认视频有新鲜帧。 |
| 再次按键没有响应 | 一次触发后要全部松开才能重新触发，持续按住不会重复切换。 |
| Another onboard operator | 前一个 `connect` 或跟随入口尚未退出；先在其终端 Ctrl-C。不能同时启动两个入口。 |
| 视频端口已占用 | 停止旧入口或旧手动转发进程；新入口不会杀掉它不拥有的进程。也可明确使用 `--no-camera`。 |
| 更新 APK 提示签名不一致 | 对比发布清单的签名指纹，确认来自同一构建来源；不要卸载原官方 XRT 来解决定制包签名问题。 |

旧联调版本曾使用独立 systemd 服务；迁移到本入口前，确认已结束跟随后执行一次：

```bash
systemctl stop vita-xrt-pc.service
systemctl stop vita-pico-camera.service
```

新的一键入口不依赖这两个 unit，也不会自动停止外部服务。
两种模式都使用原有 `build/operator.lock`，防止与其他本机操作入口同时运行。

取眼镜日志：

```bash
adb -s <PICO的IP>:5555 logcat -s Unity
```

新包会记录 `Vita camera view requested=`、`Vita camera view displayed=` 和
`state=streaming decoder=... frames=...`，可以区分按键请求、实际显示和仅解码状态。
帧年龄是本机新鲜度指标，不是跨机器端到端延迟。

## 五、维护者重新构建 APK

普通安装者可跳过本节。完整 Unity 工程固定为官方 v1.1.1，
commit `9f775b535d781618bd2bb7ef8d6c414c0531387c`；本仓库保存相机覆盖层、构建脚本和预编译 APK。

```bash
git clone --depth 1 --branch v1.1.1 \
  https://github.com/XR-Robotics/XRoboToolkit-Unity-Client.git /absolute/path/xrt-client
bash scripts/build_pico_camera_viewer.sh \
  /path/to/Unity-2022.3.16f1/Editor/Unity /absolute/path/xrt-client
```

构建机需要有效 Unity 许可证及 Android Build Support。
外部工具可通过 `VITA_ANDROID_SDK`、`VITA_ANDROID_NDK`（r23b）、`VITA_ANDROID_JDK`（JDK 11）指定。
默认构建通用包；仅为固定私有设备构建时才给脚本第三个参数传机器人 IPv4，不能把该包作为通用发行包。
构建使用 ARM64/IL2CPP、OpenGLES3 和独立包名，保持视频 shader。
脚本只对当前构建使用 .NET 配置轮询，避免共享 Linux 环境 inotify 配额耗尽导致 IL 后处理器无法启动。
输出为 `Builds/Android/XRT-Vita-Camera-1.1.1-vita.2.apk`，可用 `VITA_CAMERA_APK` 覆盖。

维护者更新 APK 时需同时更新 `releases/manifest.json`、`SHA256SUMS`，
保留上游 commit、覆盖层文件 SHA256、应用版本与签名指纹。
`prepare_pico_camera_viewer.py` 可以更新自己生成的覆盖层和设备 profile；发现人工修改会中止并保留原文件。
不应把 Unity 账号、密码、授权文件、签名私钥或现场录制图像提交到仓库。

可选的设备配置文件为应用 `persistentDataPath` 下的 `vita_camera.json`，字段为
`enabled`、`robot_host`、`port`、`width`、`height`、`fps`。
`robot_host` 为空时使用 XRT 界面地址；人工写入的持久配置优先于 APK 内置配置。
正常的新 PICO 安装不需要此文件。

## 验证范围

- Python 协议、GOP 丢帧恢复、队列边界、覆盖层生成、会话退出与进程清理测试。
- C# 四键状态机 20 项断言；真实 Unity 双眼层级 6 项断言；Java 分片读协议 8 项断言。
- Unity 完整 Android 构建、APK SHA256/签名和原 XRT 并存安装已验证。
- 173 机器人 → 电脑和 PICO 原生 HEVC 硬件解码，以及安装后的 Unity 接收器持续解码已验证。
- 新通用包的实际佩戴画面、实体四键切换、相机安装方向、解码期间跟随时序和端到端延迟，
  仍需现场操作验收；上述软件测试不能替代这些结论。

```bash
python -m unittest discover -s tests -p test_pico_camera_bridge.py
python -m unittest discover -s tests -p test_pico_session_services.py
python -m unittest discover -s tests -p test_pico_viewer_prepare.py
python scripts/check_pico_camera_stream.py --host <机器人IP> --seconds 5
# --decode 额外要求仅安装在检查电脑上的 PyAV。
```

转发器一次只服务一个视频客户端。PICO 已连接时，电脑上的接流检查会等待它断开，
不能在活动视频会话中把另一个检查客户端当作独立并行视频流。

## 维护者协议说明

视频使用独立的 VTC1 TCP 协议，复用上游 FBO 渲染器并增加 HEVC 接收器；
不通过原版 Remote Vision/Listen 控件建立视频。每条消息的头为 28 字节网络字节序
`>4sBBHIQQ`：`VTC1`、版本 1、消息类型、标志、负载长度、转发序号、源采集时间戳 ns。
类型 1 为 JSON 配置，2 为 Annex-B HEVC access unit，3 为 JSON 状态。
标志 bit 0 为独立关键帧，bit 1 为不连续；负载上限 2 MiB，客户端 JSON 上限 16 KiB。

转发队列最多六帧，队列年龄和 TCP 发送预算均为 200 ms。队列溢出、源时间戳间断、
SDK 丢帧或过旧帧都会使当前 GOP 失效；只有携带 VPS/SPS/PPS 的独立 IDR/BLA 才恢复，
并通知客户端重置解码器。不能丢一个参考 P 帧后继续把后续 P 帧当作完整图像解码。
源时间戳用于帧时序，各机单调时钟用于本机新鲜度，不能直接相减来声称端到端延迟。

单目图像放在随头部移动的平面上，背景为黑色，不产生虚假的双目视差或全景投影。
图像是否随机器人颈部转动，取决于相机实际安装位置；topic 名称不证明物理安装或光学校准。
