#!/usr/bin/env bash
# Default: read-only status. Connections occur only for explicit connect-* commands.
set -euo pipefail
vita_wifi_script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export VITA_WIFI_SCRIPT_DIR="$vita_wifi_script_dir"
exec python3 - "$@" <<'PY'
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys

IFACE = 'wlan0'
TARGETS = {
    'asus': ('ASUS-5G', 'vita-manual-ASUS-5G', 'wpa-psk'),
    'horizon': ('Horizon-Device', 'vita-manual-Horizon-Device', 'open'),
}
NMCLI = shutil.which('nmcli')
IP = shutil.which('ip')
SECRET = None


def fail(message):
    raise RuntimeError(message)


def redact(message):
    return message.replace(SECRET, '[REDACTED]') if SECRET else message


def run(args, *, allowed=(0,), timeout=40):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, 'LC_ALL': 'C'})
    if result.returncode not in allowed:
        fail('命令失败，未自动重试：' + redact(result.stderr.strip() or result.stdout.strip() or str(result.returncode)))
    return result


def nm(*args, **kwargs):
    return run([NMCLI, *args], **kwargs)


def private_bytes(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            fail('私有配置/密码文件必须是权限600的普通文件：' + str(path))
        if info.st_uid not in (0, os.geteuid()):
            fail('私有配置文件所有者不匹配')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            return stream.read(16384)
    finally:
        os.close(fd)


def status():
    print(nm('-f', 'GENERAL.DEVICE,GENERAL.TYPE,GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS',
             'device', 'show', IFACE).stdout, end='')
    print('当前 IPv4 默认路由（只读）：')
    print(run([IP, '-4', 'route', 'show', 'default']).stdout, end='')


def route_snapshot():
    # Compare all non-wlan0 routes without modifying any route or wired profile.
    def stable(value):
        if isinstance(value, dict):
            return {k: stable(v) for k, v in value.items() if k not in ('expires', 'lastuse', 'used', 'age')}
        if isinstance(value, list):
            return [stable(v) for v in value]
        return value
    result = {}
    for family in ('-4', '-6'):
        routes = json.loads(run([IP, '-j', family, 'route', 'show']).stdout)
        result[family] = sorted(json.dumps(stable(x), sort_keys=True) for x in routes if x.get('dev') != IFACE)
    return result


def load_target(target):
    global SECRET
    config_path = Path(os.environ.get('VITA_WIFI_CONFIG',
        str(Path(os.environ['VITA_WIFI_SCRIPT_DIR']).parent / 'config/wifi/vita_wifi.private.json')))
    conf = json.loads(private_bytes(config_path))
    if conf.get('schema') != 'vita_manual_wifi_v1' or conf.get('interface') != IFACE:
        fail('私有配置必须使用已审核格式和 wlan0')
    value = conf['networks'][target]
    if (value.get('ssid'), value.get('profile'), value.get('security')) != TARGETS[target]:
        fail('SSID/profile/安全类型与已审核目标不一致')
    password_path = None
    if value['security'] == 'wpa-psk':
        name = value['password_file']
        if Path(name).name != name or name in ('.', '..'):
            fail('密码文件必须位于私有配置同目录')
        password_path = config_path.parent / name
        data = private_bytes(password_path).decode('utf-8')
        prefix = '802-11-wireless-security.psk:'
        if not data.startswith(prefix) or not data.endswith('\n') or data.count('\n') != 1:
            fail('密码文件格式无效')
        SECRET = data[len(prefix):-1]
        if not (8 <= len(SECRET) <= 63 or re.fullmatch(r'[0-9a-fA-F]{64}', SECRET)) or '\r' in SECRET:
            fail('WPA密码格式无效')
    return value, password_path


def preflight_device():
    values = nm('-g', 'GENERAL.TYPE,GENERAL.STATE', 'device', 'show', IFACE).stdout.splitlines()
    if len(values) != 2 or values[0] != 'wifi':
        fail('wlan0不存在或不是NetworkManager WiFi设备；未修改网络')
    state = int(values[1].split()[0])
    if state <= 20:
        fail('wlan0未受管或不可用；不自动接管设备、不全局启用无线电')
    if nm('radio', 'wifi').stdout.strip() != 'enabled':
        fail('WiFi无线电未开启；本脚本不会修改全局无线电设置')


def existing_profile(profile, ssid, security):
    found = nm('-g', 'connection.uuid', 'connection', 'show', 'id', profile, allowed=(0, 10))
    if found.returncode == 10:
        return None
    ids = found.stdout.splitlines()
    if len(ids) != 1 or not re.fullmatch(r'[0-9a-fA-F-]{36}', ids[0]):
        fail('目标profile不唯一或UUID无效；未修改已有配置')
    fields = ['connection.type', 'connection.interface-name', '802-11-wireless.ssid',
              'connection.autoconnect', 'ipv4.method', 'ipv4.never-default',
              'ipv4.ignore-auto-routes', 'ipv4.ignore-auto-dns', 'ipv6.method',
              'ipv4.gateway', 'ipv4.routes', '802-11-wireless-security.key-mgmt']
    values = nm('-g', ','.join(fields), 'connection', 'show', 'uuid', ids[0]).stdout.splitlines()
    if len(values) != len(fields):
        fail('无法审核已有WiFi profile；未修改配置')
    expected = ['802-11-wireless', IFACE, ssid, 'no', 'auto', 'yes', 'yes', 'yes', 'disabled', '', '',
                'wpa-psk' if security == 'wpa-psk' else '']
    if [x if x != '--' else '' for x in values] != expected:
        fail('专用WiFi profile属性与保护有线路由的约定不一致；未自动修改或激活')
    return ids[0]


def connection_plan(value, password_path):
    profile = value['profile']
    identifier = existing_profile(profile, value['ssid'], value['security'])
    commands = []
    if identifier is None:
        args = ['connection', 'add', 'type', 'wifi', 'ifname', IFACE, 'con-name', profile,
                'ssid', value['ssid'],
                'connection.autoconnect', 'no', 'ipv4.method', 'auto',
                'ipv4.never-default', 'yes', 'ipv4.ignore-auto-routes', 'yes',
                'ipv4.ignore-auto-dns', 'yes', 'ipv6.method', 'disabled']
        if value['security'] == 'wpa-psk':
            args += ['wifi-sec.key-mgmt', 'wpa-psk', 'wifi-sec.psk-flags', '2']
        commands.append(args)
        # Let NetworkManager allocate the UUID. Some installed nmcli versions
        # reject connection.uuid even during connection add. Read it back and
        # validate the newly created profile before any activation.
        commands.append(['-g', 'connection.uuid', 'connection', 'show', 'id', profile])
        identifier = '<UUID_AFTER_CREATE>'
    up = ['--wait', '30', 'connection', 'up', 'uuid', identifier, 'ifname', IFACE]
    if password_path is not None:
        up += ['passwd-file', str(password_path)]
    commands.append(up)
    return commands


def main():
    arguments = sys.argv[1:]
    command = arguments[0] if arguments else 'status'
    if command in ('--help', '-h'):
        print('用法: vita_wifi.sh [status | dry-run [connect-asus|connect-horizon] | connect-asus | connect-horizon]')
        return
    if command not in ('status', 'dry-run', 'connect-asus', 'connect-horizon'):
        fail('未知命令；默认只读status，连接必须显式指定connect-asus或connect-horizon')
    if command == 'dry-run':
        if len(arguments) > 2 or len(arguments) == 2 and arguments[1] not in ('connect-asus', 'connect-horizon'):
            fail('dry-run目标仅支持connect-asus或connect-horizon')
        target = (arguments[1] if len(arguments) == 2 else 'connect-asus')[8:]
    elif len(arguments) > 1:
        fail('不支持多余参数')
    if not NMCLI or not IP:
        fail('需要NetworkManager nmcli及ip工具；未安装工具或修改网络')
    if command == 'status':
        status()
        return
    if command != 'dry-run':
        target = command[8:]
    value, password_path = load_target(target)
    preflight_device()
    commands = connection_plan(value, password_path)
    if command == 'dry-run':
        print(json.dumps({'dry_run': True, 'interface': IFACE, 'ssid': value['ssid'],
                          'profile': value['profile'], 'password_displayed': False,
                          'nmcli_arguments': commands, 'wired_profiles_changed': False,
                          'wifi_default_route': False, 'wifi_autoconnect': False}, ensure_ascii=False, indent=2))
        return
    before = route_snapshot()
    print('手工连接目标：' + value['ssid'] + '；仅wlan0，保留有线配置，不接管默认路由。', flush=True)
    created_identifier = None
    for args in commands:
        if args[:2] == ['-g', 'connection.uuid']:
            created_identifier = existing_profile(value['profile'], value['ssid'], value['security'])
            if created_identifier is None:
                fail('创建后未能读取WiFi profile UUID；未尝试连接，可检查后重新运行')
            continue
        if '<UUID_AFTER_CREATE>' in args:
            if created_identifier is None:
                fail('尚未校验新建WiFi profile；未尝试连接')
            args = [created_identifier if x == '<UUID_AFTER_CREATE>' else x for x in args]
        result = nm(*args)
        if result.stdout:
            print(redact(result.stdout), end='')
    after = route_snapshot()
    if after != before:
        fail('连接期间检测到非wlan0路由变化；未修改或自动恢复有线路由，请现场检查')
    print('WiFi连接完成；非wlan0路由与操作前一致。')
    status()


try:
    main()
except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
    print('vita_wifi: ' + redact(str(error)), file=sys.stderr)
    raise SystemExit(2)
PY
