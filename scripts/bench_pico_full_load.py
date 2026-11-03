#!/usr/bin/env python3
"""Repeated, non-actuating full-compute shadow; receiver is owned by the matrix.

Live or replay Pico body/trigger threads, frozen GMR bodies, live read-only Body31
subscription, production reference/observation/inference/DTO and JSON logging.
No controller or command publisher is constructed. Recorded feedback is an
explicit option when hardware has faults, and is labelled in every report.

N: subscription/queue/logging only. A: unrestricted control, GMR CPU1.
B: control CPU2-4, GMR CPU1. C: control CPU2-4, GMR CPU0-1.
Replay uses a synthetic 90Hz SDK shim; it excludes native RPC/packet parsing.
The native objective prototypes are not included in this benchmark.
"""
import argparse, copy, fcntl, hashlib, json, math, multiprocessing, os, signal, subprocess, sys, time
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'src/deployment/unitree_g1_ros2_29dof/src')]

def require_readonly_source(source):
    """Reject control-capable transports and block accidental publication."""
    if source.transport.publishers or source.transport._body_publisher is not None:
        raise RuntimeError('Full-load shadow requires a publisher-free state source')
    def forbidden(*args,**kwargs):
        raise RuntimeError('Publishing forbidden in full-load shadow')
    source.transport.publish=forbidden


def require_identical_outputs(results):
    """Only comparable, complete non-actuating trials may be reported."""
    if not results or len({r['qpos_sha256'] for r in results})!=1:
        raise RuntimeError('GMR numerical output changed between layouts')
    if len({r['gmr_frames'] for r in results})!=1:
        raise RuntimeError('GMR frame counts differ between layouts')
    if any(r['actions_sent'] or r['control_publishers'] for r in results):
        raise RuntimeError('Full-load shadow reported actuation')


def hardware():
    p=Path('/sys/devices/system/cpu/cpufreq')
    return dict(freq={x.name:{k:(x/k).read_text().strip() for k in
        ['scaling_cur_freq','scaling_max_freq','scaling_governor']} for x in p.glob('policy*')},
        temperature_mC={x.parent.name:int(x.read_text()) for x in Path('/sys/class/thermal').glob('thermal_zone*/temp')},
        loadavg=Path('/proc/loadavg').read_text().strip())

def stats(rows,key):
    import numpy as np
    x=np.array([r[key] for r in rows if key in r],dtype=float)
    if not len(x):return None
    return dict(mean=float(x.mean()),p50=float(np.median(x)),p95=float(np.percentile(x,95)),
                p99=float(np.percentile(x,99)),max=float(x.max()))

def capture_shared(root,config,contract,commands,output,stop,controls,progress,fixture_path,frames,receiver_pid,out,pico_mode):
    import numpy as np
    from deployment.vita_s100_aorta_30dof import actor_pico_worker as worker
    from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import SessionFixture
    sys.path.append(str(Path(root)/'build/pico'))
    fixture=SessionFixture(fixture_path);original=worker.ReferenceStream
    rows=[];paths=[];digest=hashlib.sha256();before=hardware()
    class Frozen(original):
        def push(self,body,metadata,anchor):
            i=len(rows);begin,cpu=time.monotonic(),time.thread_time()
            # Preserve real source timing for control-clock reference playback.
            result=super().push(fixture.frame_body[i].astype(np.float64),metadata,anchor)
            rows.append(dict(frame=i,retarget_ms=result['retarget_ms'],retarget_cpu_ms=result['retarget_cpu_ms'],
                snapshot_ms=(time.monotonic()-begin)*1000,snapshot_cpu_ms=(time.thread_time()-cpu)*1000,
                pico_stamp_ns=int(metadata['timestamp_ns'])))
            digest.update(self.retarget.last_qpos.astype('<f8',copy=False).tobytes())
            if self.source_path is not None and not paths:paths.append(str(self.source_path))
            result['benchmark_frame']=i
            with progress.get_lock():progress.value=len(rows)
            if len(rows)>=frames:stop.set()
            return result
    class BorrowedReceiver:
        # Only the matrix owns the vendor service lifecycle; this keeps the
        # headset connection intact across ABAB trials. The production frame
        # processing and scheduling code is unchanged.
        def poll(self):
            try:os.kill(receiver_pid,0);return None
            except ProcessLookupError:return 1
        def terminate(self):pass
        def wait(self,timeout=None):return 0
        def kill(self):pass
    popen=worker.subprocess.Popen
    expected=str(Path(root)/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice/RoboticsServiceProcess')
    def service_factory(args,*a,**kw):
        if list(args)==[expected]:return BorrowedReceiver()
        return popen(args,*a,**kw)
    worker.ReferenceStream=Frozen
    from contextlib import ExitStack
    with ExitStack() as resources:
        resources.enter_context(patch.object(worker.subprocess,'Popen',side_effect=service_factory))
        if pico_mode=='replay':
            from deployment.vita_s100_aorta_30dof import pico_input
            from humanoid_policy.local_smpl_source import LocalPicoSmplSource
            class ReplaySDK:
                started=time.monotonic()
                def get_body_pose_snapshot(self):
                    tick=int((time.monotonic()-self.started)*90)
                    return True,1_700_000_000_000_000_000+tick*11_111_111,fixture.frame_body[0].astype(np.float64).tolist()
                def get_controller_trigger_snapshot(self):
                    return True,1_700_000_000_000_000_000+int((time.monotonic()-self.started)*90)*11_111_111,0.,0.,0.
            def replay_source(*,logger,maximum_age,duplicate_poll_interval_s):
                return LocalPicoSmplSource(logger=logger,max_source_age=maximum_age,sdk=ReplaySDK(),
                    duplicate_poll_interval_s=duplicate_poll_interval_s)
            resources.enter_context(patch.object(pico_input,'create_pico_source',side_effect=replay_source))
        worker._capture(root,config,contract,None,None,commands,output,stop,controls)
    report=dict(frames=len(rows),expected_frames=frames,qpos_sha256=digest.hexdigest(),
        wall_ms=stats(rows,'retarget_ms'),cpu_ms=stats(rows,'retarget_cpu_ms'),
        steady_wall_ms=stats(rows[50:],'retarget_ms'),steady_cpu_ms=stats(rows[50:],'retarget_cpu_ms'),
        snapshot_ms=stats(rows,'snapshot_ms'),raw_gmr_logs=paths,rows=rows,before=before,after=hardware(),
        pico_mode=pico_mode,capture_header=json.loads(Path(paths[0]).read_text().splitlines()[0]) if paths else None)
    Path(out).write_text(json.dumps(report,indent=2)+'\n')
    if len(rows)!=frames:raise RuntimeError(f'Capture incomplete: {len(rows)}/{frames}')

def trial(options,label,index,receiver_pid):
    os.setsid()
    def interrupted(*_):raise KeyboardInterrupt('Matrix stopped')
    signal.signal(signal.SIGTERM,interrupted)
    import numpy as np
    from deployment.vita_s100_aorta_30dof.actor_pico_runtime import ActorPicoRuntime
    from deployment.vita_s100_aorta_30dof.actor_pico_session_replay import SessionFixture
    from deployment.vita_s100_aorta_30dof.actor_pico_worker import PicoReferenceProcess,thread_affinity
    from deployment.vita_s100_aorta_30dof.actor_pico_live import Inputs,pico_reference_command,next_policy_deadline
    from deployment.vita_s100_aorta_30dof.actor_pico_scheduling import control_scheduler
    from locomotion_aorta.pull import PullStateSource
    fixture=SessionFixture(options.fixture)
    runtime=ActorPicoRuntime(ROOT,backend='auto',policy='supermimic',human_height=fixture.human_height_m)
    assert fixture.meta['body_order']==list(runtime.policy.order)
    assert fixture.meta['model']['sha256']==runtime.config['artifacts']['model']['sha256']
    cfg=copy.deepcopy(runtime.config)
    if label=='C':cfg['pico_worker']['gmr_cpus']=[0,1]
    prefix=options.output/f'{index:02d}_{label}'
    ctx=multiprocessing.get_context('spawn')
    worker=PicoReferenceProcess(runtime)
    worker.process=ctx.Process(target=capture_shared,args=(str(ROOT),cfg,runtime.model.contract,
        worker.commands,worker.output,worker.stop,worker.controls,progress:=ctx.Value('i',0),
        str(options.fixture),options.frames,receiver_pid,str(prefix)+'.gmr.json',options.pico))
    original_mask=os.sched_getaffinity(0);original_scheduler=os.sched_getscheduler(0);original_priority=os.sched_getparam(0)
    rows=[];source=None;inferred=skipped=invalid_feedback=stale_feedback=0;before=hardware()
    try:
        # Start input before narrowing control: SDK/readers must inherit their
        # own worker mask, not the control thread's CPU2-4 mask.
        worker.start()
        control_mask=[] if label in ('A','N') else [2,3,4]
        applied=thread_affinity(control_mask)
        if control_mask:
            # Restrict model/IPC threads already created in this control
            # process too. The GMR child is a different process/session.
            for task in Path('/proc/self/task').iterdir():
                try:os.sched_setaffinity(int(task.name),control_mask)
                except ProcessLookupError:pass
        scheduler=control_scheduler()
        source=PullStateSource(node_name=f'gmr-shadow-{index:02d}-{label}')
        require_readonly_source(source)
        source.wait_state(3.)
        validator=Inputs(runtime,source,None,time)
        runtime.infer_motion(runtime.math.Motion.hold(runtime.model.contract.default),
            q=runtime.model.contract.default,dq=np.zeros(28),root=[1,0,0,0],gyro=np.zeros(3))
        runtime.reset_inputs();worker.connect([1.,0.,0.,0.])
        started=deadline=time.monotonic();first=None;last_notice=0.;ticks=0
        with Path(str(prefix)+'.control.jsonl').open('x',buffering=1024*1024) as log:
            log.write(json.dumps(dict(record_type='full_shadow_header',label=label,feedback=options.feedback,
                control_affinity=applied,scheduler=scheduler,pico_stream=cfg['pico_stream'],pico_worker=cfg['pico_worker'],
                observations_recorded=True,actions_sent=0,control_publishers=0))+'\n')
            while worker.process.is_alive() and progress.value<options.frames:
                time.sleep(max(0.,deadline-time.monotonic()))
                begin,cpu=time.monotonic(),time.thread_time();ticks+=1
                snapshot=worker.snapshot()
                if snapshot.get('state')=='failed':raise RuntimeError(str(snapshot.get('error')))
                state=source.wait_state(0);status=source.control_status
                age=validator.ages(state)
                try:live_q,live_dq,live_root,live_gyro=validator.validate_state(state)
                except (ValueError,TimeoutError) as error:
                    invalid_feedback+=1
                    if 'stale' in str(error).lower():stale_feedback+=1
                    if options.feedback=='live':raise
                if options.feedback=='recorded':
                    i=inferred%len(fixture.tick_q_body)
                    q=fixture.tick_q_body[i,runtime.policy.mapping];dq=fixture.tick_dq_body[i,runtime.policy.mapping]
                    root=fixture.tick_root_wxyz[i];gyro=fixture.tick_gyro[i]
                else:q,dq,root,gyro=live_q,live_dq,live_root,live_gyro
                diagnostics=dict(record_type='full_shadow_tick',tick=ticks,feedback_mode=options.feedback,
                    gmr_frames=int(progress.value),actual_state_id=int(state.state_id),actual_state_ages=age,
                    actual_joint_errors=state.error_code.tolist(),
                    actual_control_phase=None if status is None else str(status.phase),actions_sent=0,control_handles_created=0)
                if snapshot.get('state')=='ready':
                    if first is None:first=begin
                    if label=='N':
                        diagnostics.update(record_type='queue_only_tick',retarget_ms=snapshot['retarget_ms'],
                            retarget_cpu_ms=snapshot['retarget_cpu_ms'])
                    else:
                        _,obs,row=pico_reference_command(runtime,snapshot,begin,q=q,dq=dq,root=root,gyro=gyro,yaw_offset=0.)
                        diagnostics.update(row,record_type='full_shadow_tick',observation=obs.tolist(),
                            feedback_used=dict(q=q.tolist(),dq=dq.tolist(),root=root.tolist(),gyro=gyro.tolist()))
                        inferred+=1
                compute=(time.monotonic()-begin)*1000
                diagnostics.update(compute_ms=compute,thread_cpu_ms=(time.thread_time()-cpu)*1000)
                log.write(json.dumps(diagnostics,allow_nan=False,separators=(',',':'))+'\n')
                complete=(time.monotonic()-begin)*1000
                timing=dict(compute_ms=compute,complete_ms=complete,thread_cpu_ms=(time.thread_time()-cpu)*1000,
                    start_monotonic=begin,active=('inference_ms' in diagnostics))
                if 'inference_ms' in diagnostics:
                    timing.update(inference_ms=diagnostics['inference_ms'],policy_ms=diagnostics['total_ms'])
                rows.append(timing)
                deadline,missed=next_policy_deadline(deadline,time.monotonic(),.02);skipped+=missed
                now=time.monotonic()
                if now-last_notice>10:
                    print(json.dumps(dict(trial=index,label=label,gmr_frames=progress.value,inference_ticks=inferred,
                        skipped_ticks=skipped,actions_sent=0)),flush=True);last_notice=now
                if first is None and now-started>options.startup_timeout:raise TimeoutError('Waiting for Pico body data')
                if now-started>options.startup_timeout+max(120,options.frames*.2):raise TimeoutError('Trial deadline')
        elapsed=time.monotonic()-started
    finally:
        os.sched_setaffinity(0,original_mask);os.sched_setscheduler(0,original_scheduler,original_priority)
        worker.stop.set();worker.process.join(timeout=6.)
        worker.close()
        if source is not None:source.close()
        assert os.sched_getaffinity(0)==original_mask
    gmr=json.loads(Path(str(prefix)+'.gmr.json').read_text())
    assert gmr['frames']==options.frames
    assert len({r['pico_stamp_ns'] for r in gmr['rows']})==options.frames
    active_rows=[r for r in rows if r['active']]
    # Infer cadence uses timestamps, not cleanup duration.
    cadence=[]
    for a,b in zip(active_rows,active_rows[1:]):cadence.append(dict(interval_ms=(b['start_monotonic']-a['start_monotonic'])*1000))
    report=dict(label=label,index=index,feedback_mode=options.feedback,pico_mode=options.pico,control_affinity=applied,
        scheduler=scheduler,gmr_affinity=gmr['capture_header']['gmr_affinity'],worker_affinity=gmr['capture_header']['worker_affinity'],
        gmr_frames=gmr['frames'],qpos_sha256=gmr['qpos_sha256'],gmr_wall_ms=gmr['wall_ms'],gmr_cpu_ms=gmr['cpu_ms'],
        steady_gmr_wall_ms=gmr['steady_wall_ms'],steady_gmr_cpu_ms=gmr['steady_cpu_ms'],
        control_ticks=ticks,inference_ticks=inferred,elapsed_s=elapsed,skipped_ticks=skipped,
        invalid_live_feedback_ticks=invalid_feedback,stale_live_feedback_ticks=stale_feedback,
        control_complete_ms=stats(rows,'complete_ms'),control_cpu_ms=stats(rows,'thread_cpu_ms'),
        active_control_complete_ms=stats(active_rows,'complete_ms'),control_interval_ms=stats(cadence,'interval_ms'),
        policy_ms=stats(rows,'policy_ms'),inference_ms=stats(rows,'inference_ms'),
        control_overrun_ticks=sum(r['complete_ms']>20 for r in rows),actions_sent=0,control_publishers=0,
        raw_gmr_logs=gmr['raw_gmr_logs'],before=before,after=hardware())
    Path(str(prefix)+'.report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('TRIAL_COMPLETE '+json.dumps(report),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,default=ROOT/'tests/fixtures/actor_pico/robot42_pico_session_20261029.npz')
    parser.add_argument('--frames',type=int,default=1000)
    parser.add_argument('--feedback',choices=['live','recorded'],default='live')
    parser.add_argument('--pico',choices=['live','replay'],default='live')
    parser.add_argument('--sequence',default='NABABABACACAC')
    parser.add_argument('--startup-timeout',type=float,default=300.)
    parser.add_argument('--output',type=Path,required=True)
    options=parser.parse_args()
    if set(options.sequence)-set('NABC') or not options.sequence:parser.error('Unknown condition')
    if not 50<=options.frames<=5107:parser.error('Frames must be in [50,5107]')
    if not math.isfinite(options.startup_timeout) or not 0<options.startup_timeout<=300:
        parser.error('startup-timeout must lie in (0,300] seconds')
    if options.output.exists() and (not options.output.is_dir() or any(options.output.iterdir())):
        parser.error('output directory must be empty; existing test data is never overwritten')
    options.output.mkdir(parents=True,exist_ok=True)
    sdk=ROOT/'vendor_packages/xrobotoolkit-arm64/opt/apps/roboticsservice'
    os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MUJOCO_GL='disable',
        QT_PLUGIN_PATH=str(sdk/'plugins'),QT_QPA_PLATFORM='offscreen',XDG_RUNTIME_DIR=str(ROOT/'build/pico/actor_xdg'),
        ZENOH_SESSION_CONFIG_URI=str(ROOT/'outputs/actor_pico_zenoh_session.json5'))
    os.environ['LD_LIBRARY_PATH']=':'.join(map(str,[ROOT/'venv/lib/python3.10/site-packages/locomotion_aorta/_onnx/.libs',
        ROOT/'build/pico',sdk,sdk/'lib',sdk/'SDK/arm64','/usr/hobot/lib']))+':'+os.environ.get('LD_LIBRARY_PATH','')
    results=[]
    with (ROOT/'build/operator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with (options.output/'receiver.log').open('w') as log:
            service=subprocess.Popen([str(sdk/'RoboticsServiceProcess')],cwd=sdk,env=os.environ,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            (options.output/'receiver.pid').write_text(str(service.pid))
            try:
                ctx=multiprocessing.get_context('spawn');current=None
                for index,label in enumerate(options.sequence):
                    if service.poll() is not None:raise RuntimeError('Owned receiver exited')
                    # Fresh process avoids state/SDK thread-pool carryover
                    # between the alternating trials.
                    current=ctx.Process(target=trial,args=(options,label,index,service.pid))
                    current.start();deadline=time.monotonic()+options.startup_timeout+max(150,options.frames*.2)
                    while current.is_alive() and time.monotonic()<deadline:current.join(timeout=.5)
                    if current.is_alive():
                        current.terminate();current.join(timeout=8.)
                        if current.is_alive():current.kill();current.join(timeout=2.)
                        raise TimeoutError('Control shadow trial timed out')
                    if current.exitcode:raise RuntimeError(f'Trial {index}/{label} exited {current.exitcode}')
                    report_path=options.output/f'{index:02d}_{label}.report.json'
                    results.append(json.loads(report_path.read_text()))
                    (options.output/'reports.json').write_text(json.dumps(results,indent=2)+'\n')
                    time.sleep(1.)
            finally:
                if 'current' in locals() and current is not None and current.is_alive():
                    current.terminate();current.join(timeout=8.)
                    if current.is_alive():current.kill();current.join(timeout=2.)
                service.terminate()
                try:service.wait(timeout=3.)
                except subprocess.TimeoutExpired:service.kill();service.wait(timeout=2.)
                (options.output/'cleanup.json').write_text(json.dumps(dict(receiver_pid=service.pid,
                    receiver_exitcode=service.returncode,actions_sent=0,hardware=hardware()),indent=2)+'\n')
    require_identical_outputs(results)
    print('MATRIX_COMPLETE '+json.dumps(dict(trials=len(results),all_gmr_outputs_identical=True,actions_sent=0)),flush=True)

if __name__=='__main__':main()
