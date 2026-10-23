#!/usr/bin/env python3
"""Package local ARM Pico runtime with fresh settings and an empty factory DB.

Used by maintainers after changing vendor inputs/binding builds. Does not run
the service or copy saved device records into the versioned runtime archive.
"""
import io
from pathlib import Path
import sqlite3
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = b'''[ChangeLanguage]
Language=EN
[ProtocolType]
Protocol=2
[RTC]
SpecifyServer=2
ServerIP=127.0.0.1
ConnectedBefore=false
ConnectedIP=127.0.0.1
mode=0
WanBusinessID=
WanTokenUrl=
[RunMode]
mode=0
[Log]
logToFile=1
[Service]
runAsService=false
'''


def add_file(tar, path, name, mode=0o644):
    info = tar.gettarinfo(str(path), arcname=name)
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    info.mtime = 0
    info.mode = mode
    with path.open('rb') as stream:
        tar.addfile(info, stream)


def main():
    sdk = ROOT/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    output = ROOT/'deploy/actor_s100/runtime'
    output.mkdir(parents=True, exist_ok=True)
    files = [sdk/'RoboticsServiceProcess']
    files += sorted(p for p in sdk.glob('*.so*') if p.is_file())
    for sub in ['lib', 'plugins', 'SDK']:
        files += sorted(p for p in (sdk/sub).rglob('*') if p.is_file())
    with tempfile.TemporaryDirectory(prefix='pico-factory-db-') as tmp:
        factory = Path(tmp)/'epDBBackup.s3db'
        source = sqlite3.connect('file:'+str(sdk/'epDBBackup.s3db')+'?mode=ro', uri=True)
        try:
            schemas = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'").fetchall()
        finally:
            source.close()
        with sqlite3.connect(factory) as database:
            for (sql,) in schemas:
                database.execute(sql)
        with tarfile.open(output/'pico_service_arm64.tar.gz', 'w:gz', compresslevel=6) as tar:
            for path in files:
                name = str(path.relative_to(sdk))
                add_file(tar, path, name, 0o755 if name == 'RoboticsServiceProcess' else 0o644)
            add_file(tar, factory, 'epDBBackup.s3db', 0o600)
            info = tarfile.TarInfo('setting.ini')
            info.size = len(SETTINGS)
            info.mode = 0o600
            tar.addfile(info, io.BytesIO(SETTINGS))
    with tarfile.open(output/'pico_binding_cp310_aarch64.tar.gz', 'w:gz', compresslevel=6) as tar:
        for name in ['xrobotoolkit_sdk.cpython-310-aarch64-linux-gnu.so', 'libdouble-conversion.so.3']:
            add_file(tar, ROOT/'build/pico'/name, name)
    print('Packaged Pico service, empty factory database, fresh settings and CPython 3.10 ARM64 binding.')


if __name__ == '__main__':
    main()
