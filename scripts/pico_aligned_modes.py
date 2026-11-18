"""Git-style non-actuating modes for the two explicit policy modalities."""
import importlib.util,json,os,sys,time
from pathlib import Path
def run_mode(args,parser):
    root=args.root.resolve()
    from bootstrap_pico_inputs_s100 import verify_inputs,require_target
    verify_inputs(root)
    if args.mode=='check':
        from check_pico_aligned import check
        import platform
        backend=('sdk' if platform.machine()=='aarch64' else 'ort') if args.backend=='auto' else args.backend
        if not 180<=args.steps<=3000:parser.error('check requires 180..3000 steps')
        report=check(root,args.policy,backend,args.steps)
        if args.out:
            args.out.mkdir(parents=True,exist_ok=True);(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report));return
    if args.mode in ('shadow','replay'):
        manifest=json.loads((root/'deploy/actor_s100/manifest.json').read_text())
        fixture=manifest['pico_inputs_release']['offline_check']
        command=[sys.executable,str(root/'scripts/pico_inputs.py'),args.policy,args.mode,
            '--feedback','body31' if args.mode=='shadow' else 'synthetic-stand',
            '--backend',args.backend,'--threads',str(args.threads),'--seconds',str(args.seconds if args.seconds is not None else (30 if args.mode=='shadow' else 0))]
        if args.policy=='pico_smpl':command+=['--wrist-mode','stand']
        if args.mode=='replay':command+=['--input',str(root/fixture['input']['path']),'--sha256',fixture['input']['sha256'],
            '--trigger-script',str(root/fixture['triggers']['path'])]
        else:command+=['--pico-service',args.pico_service,'--group',args.group]
        if args.out:command+=['--out',str(args.out)]
        if args.record_observations:command+=['--record-observations']
        os.execv(sys.executable,command)
    require_target()
    from deployment.vita_s100_aorta_30dof.pico_aligned_runtime import AlignedRuntime
    runtime=AlignedRuntime(root,args.policy,backend='sdk',threads=args.threads)
    # Reuse the Git read-only Body31 function itself, no controller creation.
    path=root/'scripts/actor_pico.py';spec=importlib.util.spec_from_file_location('git_actor_modes',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    args.seconds=5. if args.seconds is None else args.seconds
    if not 0<args.seconds<=30:parser.error('state-check duration must be 0..30 seconds')
    report=module.state_check(runtime,args,lambda row,*_:print(json.dumps(row)))
    print(json.dumps(report))
