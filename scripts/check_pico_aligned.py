#!/usr/bin/env python3
"""Model/input-adapter check only, not live lifecycle or hardware acceptance."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def check(root,route,backend,steps=240):
    import numpy as np
    if backend=='ort':
        from deployment.vita_s100_aorta_30dof.pico_shadow_math import prepare
        prepare(root,route)
    from deployment.vita_s100_aorta_30dof.pico_smpl_workflow import prepare_guard_imports,VirtualClock
    prepare_guard_imports(root)
    from deployment.vita_s100_aorta_30dof.pico_aligned_runtime import AlignedRuntime,ManualInput,pico_reference_command
    from deployment.vita_s100_aorta_30dof.actor_human_encoder import Network
    clock=VirtualClock();runtime=AlignedRuntime(root,route,backend=backend);runtime.clock=clock
    feedback=dict(q=runtime.model.contract.default.copy(),dq=np.zeros(28),root=np.array([1.,0.,0.,0.]),gyro=np.zeros(3))
    with np.load(root/'checks/pico_historical_75frames.npz',allow_pickle=False) as z:body=z['body_poses'][0].copy()
    class Receiver:
        def poll(self,stamp):return (body,stamp,stamp),(True,stamp,0.,0.,0.),dict(state='receiving')
    manual=ManualInput(runtime,Receiver(),clock=clock);manual.connect(feedback['root'])
    full=Network(root/runtime.runtime.config['artifacts']['model']['path'],runtime.math,backend,1,
        input_name='obs_dict',input_dim=1740 if route=='pico_smpl' else 1167,output_name='action',output_dim=28)
    peak=0.;tracking=0;plan=runtime.hold_motion();plan.lead_in(feedback['q'],feedback['root'],{})
    runtime.infer_motion_command(plan,**feedback)
    for i in range(steps):
        snapshot=manual.snapshot()
        if i>=160:
            command,observation,row=pico_reference_command(runtime,snapshot,clock.monotonic(),**feedback);tracking+=1
        else:command,observation,row=runtime.infer_motion_command(plan,ref_time=i*.02,**feedback)
        if command.protocol_version!=4:raise ValueError('BodyCmd4 declaration missing')
        if i%40==0 or i==steps-1:
            expected=full.run(observation[None])[0]
            peak=max(peak,float(np.max(np.abs(expected-row['raw_action']))))
        clock.sleep(.02)
    if peak>1e-6 or not tracking:raise ValueError('Original modality model parity/tracking failed')
    return dict(status='PASS_ALIGNED_OFFLINE_CHECK',route=route,backend=backend,inference_steps=steps,
        scope='model_and_input_adapter_only',live_lifecycle_exercised=False,
        checkpoint_iteration=runtime.runtime.config['checkpoint_iteration'],
        model_sha256=runtime.runtime.config['artifacts']['model']['sha256'],
        tracking_steps=tracking,full_onnx_parity_abs_max=peak,actions_sent=0,control_handles_created=0,
        input='historical neutral body / simulated clocks',feedback='synthetic neutral',
        hardware_startup_exit_validated=False)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--route',choices=('pico_smpl','pico_teleop'),required=True)
    p.add_argument('--backend',choices=('ort','sdk'),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=240);a=p.parse_args()
    if not 180<=a.steps<=3000:p.error('Use 180..3000 explicit offline steps')
    report=check(ROOT,a.route,a.backend,a.steps);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
