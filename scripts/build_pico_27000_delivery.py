#!/usr/bin/env python3
"""Build isolated 27000 deliveries from tracked code and explicitly supplied assets."""
import argparse,email,gzip,hashlib,json,shutil,sys,tarfile,tempfile,zipfile
from pathlib import Path
from bootstrap_actor_s100 import inside,sha
from build_actor_delivery import source_revision,resolve_wheels,runtime_inputs,POLICY_CONFIGS
ROOT=Path(__file__).resolve().parents[1]

def validate_target_closure(stage):
    """Evaluate wheel metadata against Linux/aarch64, including extras, on Windows."""
    from pip._vendor.packaging.requirements import Requirement
    from pip._vendor.packaging.markers import default_environment
    from pip._vendor.packaging.utils import canonicalize_name
    from pip._vendor.packaging.version import Version
    env=default_environment();env.update(os_name='posix',sys_platform='linux',platform_system='Linux',
        platform_machine='aarch64',python_version='3.10',python_full_version='3.10.12',extra='')
    wheels={}
    for path in (stage/'deploy/actor_s100/wheelhouse').glob('*.whl'):
        with zipfile.ZipFile(path) as z:
            names=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
            if len(names)!=1:raise ValueError('Ambiguous wheel metadata')
            data=email.message_from_bytes(z.read(names[0]));name=canonicalize_name(data['Name'])
            wheels.setdefault(name,[]).append(data)
    queue=[Requirement(line.strip()) for line in (stage/'deploy/actor_s100/requirements.txt').read_text().splitlines()
        if line.strip() and not line.lstrip().startswith('#')];visited=set()
    while queue:
        requirement=queue.pop();name=canonicalize_name(requirement.name)
        if requirement.marker and not requirement.marker.evaluate(env):continue
        matches=[data for data in wheels.get(name,[]) if requirement.specifier.contains(data['Version'],prereleases=True)]
        if not matches:
            raise ValueError('Incomplete target wheel closure: '+str(requirement))
        data=max(matches,key=lambda item:Version(item['Version']))
        key=(name,tuple(sorted(requirement.extras)))
        if key in visited:continue
        visited.add(key)
        for raw in data.get_all('Requires-Dist',[]):
            dependency=Requirement(raw)
            if dependency.marker is None or any(dependency.marker.evaluate(dict(env,extra=extra))
                for extra in ('',*requirement.extras)):
                dependency.marker=None;queue.append(dependency)
    return len(wheels)

def refresh_json_bindings(stage):
    # Git text normalization changes bytes, not numerical/model contracts.
    for filename in ('actor_smpl.json','actor_threepoint.json'):
        path=stage/'config'/filename;settings=json.loads(path.read_text())
        for item in settings['artifacts'].values():
            target=inside(stage,item['path'])
            if target.suffix=='.json':
                item['sha256']=sha(target)
                if 'bytes' in item:item['bytes']=target.stat().st_size
        path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    path=stage/'config/pico_inputs.json';settings=json.loads(path.read_text())
    for item in settings['assets'].values():item['sha256']=sha(inside(stage,item['path']))
    path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    path=stage/'checks/workflow/fixture.json';settings=json.loads(path.read_text())
    for item in settings['files'].values():item['sha256']=sha(inside(stage,item['path']))
    path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')

def stage_delivery(root,assets,destination,*,allow_dirty=False):
    root,assets,destination=map(lambda p:Path(p).resolve(),(root,assets,destination))
    source=source_revision(root,allow_dirty)
    inputs=json.loads((root/'deploy/pico_27000/delivery_inputs.json').read_text())
    destination.mkdir(parents=True,exist_ok=False)
    seen=set()
    for item in inputs['source_files']:
        name=item['path'];src=inside(root,item['source'])
        if name in seen or not src.is_file() or src.is_symlink():raise ValueError('Invalid source input: '+name)
        seen.add(name);out=inside(destination,name);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out)
    for item in inputs['external_files']:
        name=item['path'];src=inside(assets,name)
        if name in seen or not src.is_file() or src.is_symlink() or src.stat().st_size!=item['bytes'] or sha(src)!=item['sha256']:
            raise ValueError('Missing or changed separately supplied asset: '+name)
        seen.add(name);out=inside(destination,name);out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out)
    refresh_json_bindings(destination)
    # Merge the original actor package into the SAME root/venv/manifest.
    legacy_names,external_models=runtime_inputs(root,'base')
    for name in legacy_names:
        name=name.replace('\\','/')
        if name in ('config/sdk021-wheel-guard.json','deploy/actor_s100/requirements.txt'):continue
        out=inside(destination,name);out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(inside(root,name),out)
    guard=json.loads((destination/'config/sdk021-wheel-guard.json').read_text())
    for path in (destination/'deploy/actor_s100/wheelhouse').glob('locomotion_aorta-*.whl'):
        if path.name!=guard['wheel']['filename']:path.unlink()
    for filename in POLICY_CONFIGS.values():
        path=destination/filename;settings=json.loads(path.read_text())
        settings['sdk_sources']={name:guard['source_sha256'][name] for name in settings['sdk_sources']}
        settings.update(sdk_version=guard['sdk_version'],sdk_commit=guard['sdk_commit'])
        path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    # Code bindings are sealed against this reviewed checkout; model hashes stay pinned.
    for route in ('pico_smpl','pico_teleop'):
        path=destination/f'config/{route}_live.json';settings=json.loads(path.read_text())
        settings['bindings']={name:sha(inside(destination,name)) for name in settings['bindings']}
        path.write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    manifest=inputs['template'];manifest['source']=source
    manifest['external_resources']=external_models
    manifest['pico_inputs_release']['unified_environment']=True
    if sys.platform=='win32':validate_target_closure(destination)
    else:
        with tempfile.TemporaryDirectory() as resolved:
            resolve_wheels(destination,Path('deploy/actor_s100/requirements.txt'),Path(resolved))
    for item in manifest['pico_inputs_release']['workflow_check'].values():
        item['sha256']=sha(inside(destination,item['path']))
    files=[dict(path=p.relative_to(destination).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(destination.rglob('*')) if p.is_file()]
    manifest['files']=files
    manifest['payload_sha256']=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    (destination/'deploy/actor_s100/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    from bootstrap_pico_inputs_s100 import verify_inputs
    verify_inputs(destination)
    return manifest

def build(root,assets,output,*,allow_dirty=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);reports=[]
    with tempfile.TemporaryDirectory(prefix='pico27000-') as temp:
        stage=Path(temp)/'stage';manifest=stage_delivery(root,assets,stage,allow_dirty=allow_dirty)
        for route in ('all',):
            top=f'actor-s100-base-{manifest["source"]["commit"][:12]}-{manifest["payload_sha256"][:12]}'
            archive=output/(top+'.tar.gz')
            with archive.open('xb') as stream,gzip.GzipFile(fileobj=stream,filename='',mode='wb',mtime=0) as zipped:
                with tarfile.open(fileobj=zipped,mode='w') as tar:
                    for path in sorted(stage.rglob('*')):
                        if not path.is_file():continue
                        info=tar.gettarinfo(str(path),arcname=top+'/'+path.relative_to(stage).as_posix())
                        info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o755 if path.suffix=='.sh' else 0o644
                        with path.open('rb') as data:tar.addfile(info,data)
            digest=sha(archive);archive.with_suffix('.gz.sha256').write_text(digest+'  '+archive.name+'\n',encoding='ascii')
            reports.append(dict(route=route,archive=str(archive),sha256=digest,payload_sha256=manifest['payload_sha256']))
    return reports[0]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'outputs/pico27000_delivery')
    p.add_argument('--allow-dirty',action='store_true');a=p.parse_args()
    print(json.dumps(build(ROOT,a.assets_root,a.output_dir,allow_dirty=a.allow_dirty),indent=2))
if __name__=='__main__':main()
