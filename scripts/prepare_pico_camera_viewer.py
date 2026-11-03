#!/usr/bin/env python3
"""Apply the versioned video overlay to a clean XRoboToolkit v1.1.1 checkout."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = '9f775b535d781618bd2bb7ef8d6c414c0531387c'


def replace_once(path, before, after):
    text = path.read_text()
    if after in text:
        return
    if text.count(before) != 1:
        raise RuntimeError('Unsupported client source; refusing ambiguous edit: '+str(path))
    path.write_text(text.replace(before, after, 1))


def prepare(project, robot_host=None):
    project = Path(project).resolve()
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip()
    if actual != BASE_COMMIT:
        raise ValueError('Expected XRoboToolkit v1.1.1 commit '+BASE_COMMIT+', got '+actual)
    if robot_host is not None:
        address = ipaddress.IPv4Address(robot_host)
        if address.is_unspecified or address.is_loopback or address.is_multicast:
            raise ValueError('Use the actual robot LAN IPv4')
    overlay = ROOT/'deploy/pico_viewer/Assets'
    output = project/'Builds/vita_camera_preparation.json'
    previous = json.loads(output.read_text()) if output.exists() else {}
    previous_hashes = {item['path']: item['sha256'] for item in previous.get('overlay_files', [])}
    copied = []
    for source in sorted(overlay.rglob('*')):
        if not source.is_file():
            continue
        target = project/'Assets'/source.relative_to(overlay)
        if target.exists() and target.read_bytes() != source.read_bytes():
            old_hash = hashlib.sha256(target.read_bytes()).hexdigest()
            if previous_hashes.get(str(target.relative_to(project))) != old_hash:
                raise RuntimeError('Viewer overlay differs; preserve/review local edits first: '+str(target))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append(dict(path=str(target.relative_to(project)),
            sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    replace_once(project/'Assets/Scripts/UI/UIOperate.cs',
        'if (AcontrolerTog != null && AcontrolerTog.isOn)',
        'if (!VitaCameraView.ReserveFaceButtons && AcontrolerTog != null && AcontrolerTog.isOn)')
    replace_once(project/'Assets/Scripts/Li/toggle-objects-script.cs',
        'void Update()\n    {',
        'void Update()\n    {\n        if (VitaCameraView.ReserveFaceButtons) return;')
    head = project/'Assets/InteractionTools/Scripts/HeadDevice.cs'
    replace_once(head, 'state.confirm = Input.GetKey(KeyCode.JoystickButton0);',
        'state.confirm = !VitaCameraView.ReserveFaceButtons && Input.GetKey(KeyCode.JoystickButton0);')
    replace_once(head, 'state.cancel = Input.GetButton("Cancel");',
        'state.cancel = !VitaCameraView.ReserveFaceButtons && Input.GetButton("Cancel");')
    actions_path = project/'Assets/InteractionTools/PICO Input Actions.inputactions'
    actions = json.loads(actions_path.read_text())
    for action_map in actions['maps']:
        action_map['bindings'] = [binding for binding in action_map['bindings']
            if not (binding.get('action') in ('Quit', 'SendData')
                    and binding.get('path', '').startswith('<PXR_Controller>')
                    and binding.get('path', '').endswith(('/primaryButton', '/secondaryButton')))]
    actions_path.write_text(json.dumps(actions, indent=4)+'\n')
    profile = dict(enabled=True, robot_host=robot_host or '', port=13579,
                   width=960, height=768, fps=30)
    destination = project/'Assets/Resources/vita_camera.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        current_profile = json.loads(destination.read_text())
        if current_profile != profile and current_profile != previous.get('profile'):
            raise RuntimeError('Device camera profile has local edits; preserve/review first: '+str(destination))
    destination.write_text(json.dumps(profile, indent=2)+'\n')
    receipt = dict(schema='vita_pico_viewer_preparation_v1', upstream_commit=actual,
                   project=str(project), overlay_files=copied, profile=profile,
                   patched_files=[dict(path=name, sha256=hashlib.sha256((project/name).read_bytes()).hexdigest())
                       for name in ('Assets/Scripts/UI/UIOperate.cs',
                                    'Assets/Scripts/Li/toggle-objects-script.cs',
                                    'Assets/InteractionTools/Scripts/HeadDevice.cs',
                                    'Assets/InteractionTools/PICO Input Actions.inputactions')],
                   package='com.xrobotoolkit.client.vitacamera')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--robot-host', help='Optional private APK default; otherwise use PC Service IP')
    args = p.parse_args()
    print(json.dumps(prepare(args.project, args.robot_host), indent=2))


if __name__ == '__main__':
    main()
