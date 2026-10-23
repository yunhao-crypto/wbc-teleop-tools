"""Independent original conditioning and inert wire audit for captured shadow runs."""
import ast
import hashlib
import json
from pathlib import Path
from typing import Sequence
import numpy as np


def read_commands(path):
    with path.open('rb') as stream:
        while True:
            header=stream.read(4)
            if not header:return
            if len(header)!=4:raise ValueError('Truncated LowCmd length')
            size=int.from_bytes(header,'little')
            if size>65536:raise ValueError('Unexpected LowCmd size')
            body=stream.read(size)
            if len(body)!=size:raise ValueError('Truncated LowCmd payload')
            yield body


class OriginalOutputAudit:
    def __init__(self, runtime, initial_q, *, original_source, input_file, rows, output):
        from deployment.common.policy_runtime import PolicyStep
        from deployment.vita_s100_aorta_30dof.command_guard import CommandEnvelope, command_envelope_from_contract
        from deployment.vita_s100_aorta_30dof.policy_contract import VITA_ACTION_TO_COMMAND_SLOTS,VITA_COMMAND_SLOT_COUNT
        from deployment.vita_s100_aorta_30dof.robot_adapter import VITA_CONTROLLER_JOINT_NAMES
        self.source=original_source/'deployment/vita_s100_aorta_30dof/aorta_adapter.py'
        source=self.source.read_text()
        names={'InitialPoseError','validate_gains','VitaCommandAdapter'}
        tree=ast.parse(source,filename=str(self.source))
        selected=[x for x in tree.body if isinstance(x,(ast.ClassDef,ast.FunctionDef)) and x.name in names]
        if len(selected)!=3:raise ValueError('Original conditioning symbols missing')
        namespace=dict(np=np,Sequence=Sequence,PolicyStep=PolicyStep,CommandEnvelope=CommandEnvelope,
                       VITA_ACTION_TO_COMMAND_SLOTS=VITA_ACTION_TO_COMMAND_SLOTS,VITA_COMMAND_SLOT_COUNT=VITA_COMMAND_SLOT_COUNT,
                       VITA_CONTROLLER_JOINT_NAMES=VITA_CONTROLLER_JOINT_NAMES)
        # Only the three pure definitions execute; no SDK module imports/init.
        exec(compile(ast.Module(body=selected,type_ignores=[]),str(self.source),'exec'),namespace)
        interface=runtime.bundle.artifacts['deployment_vita02a_a_sample_full_interface'].path
        base=command_envelope_from_contract(json.loads(interface.read_text()))
        self.envelopes={name:runtime.command_envelope(base,name) for name in ('velocity','motion')}
        self.adapter=namespace['VitaCommandAdapter'](self.envelopes['velocity'])
        self.hold=np.asarray(initial_q,np.float64)
        self.adapter.arm(self.hold)
        self.wirefile=input_file.with_suffix('.inert_lowcmd.bin')
        self.payloads=list(read_commands(self.wirefile))
        if len(self.payloads)!=len(rows):raise ValueError('LowCmd and capture frame counts differ')
        self.output=output;self.errors=[];self.modification_differences=[];self.command_id=0;self.frames=0
        self.expected=np.zeros((len(rows),3,42),np.float32)
        self.max_error=np.zeros(3,np.float64);self.over=np.zeros(3,np.int64)

    def tick(self, index, step, row):
        from lowlevel.LowCmd import LowCmd
        wire=self.payloads[index]
        if step is None:
            if wire or 'inert_output' in row:self.errors.append(dict(frame=index,error='Unexpected output without policy step'))
            return
        self.frames+=1
        self.command_id=self.command_id%65535+1
        self.adapter.envelope=self.envelopes['motion' if row['mode']=='motion' else 'velocity']
        q,kp,kd,conditioning=self.adapter.condition(step,hold_q=self.hold,period_s=.02)
        original=np.asarray([q,kp,kd],np.float32);self.expected[index]=original
        recorded=row.get('inert_output')
        if recorded is None:self.errors.append(dict(frame=index,error='Missing inert output'));return
        if (recorded['guard_allowed'] is not True or recorded['guard_reasons']
                or recorded['ownership_scope']!='simulated-local-only'):
            self.errors.append(dict(frame=index,error='Guard/ownership result invalid'))
        values=np.asarray([recorded[k] for k in ('target_q42','kp42','kd42')],np.float32)
        if values.shape!=(3,42) or not np.isfinite(values).all():raise ValueError('Bad output arrays')
        error=np.abs(values.astype(np.float64)-original.astype(np.float64))
        self.max_error=np.maximum(self.max_error,error.max(axis=1))
        self.over+=(error>np.asarray([1e-4,0,0])[:,None]).sum(axis=1)
        for key,value in conditioning.items():
            if value != recorded['conditioning'][key]:
                self.modification_differences.append(dict(frame=index,field=key,original=value,board=recorded['conditioning'][key]))
        if not wire:self.errors.append(dict(frame=index,error='Missing serialized payload'));return
        if len(wire)!=recorded['payload_bytes']:self.errors.append(dict(frame=index,error='Payload size mismatch'))
        msg=LowCmd.GetRootAs(wire)
        if msg.MotorCmdLength()!=42 or msg.CmdId()!=self.command_id:
            self.errors.append(dict(frame=index,error='Wire slots or command ID mismatch'));return
        if msg.AortaHeader() is not None or msg.ExternalVersion() or msg.ExternalSessionId() or msg.ExternalSequence():
            self.errors.append(dict(frame=index,error='Unexpected external metadata'))
        decoded=np.zeros((3,42),np.float32)
        for j in range(42):
            motor=msg.MotorCmd(j)
            if (motor.Mode(),motor.Dq(),motor.Tau())!=(10,0.,0.):
                self.errors.append(dict(frame=index,slot=j,error='Invalid motor mode/dq/tau'))
            decoded[:,j]=(motor.Q(),motor.Kp(),motor.Kd())
        if not np.array_equal(decoded.view(np.uint32),values.view(np.uint32)):
            self.errors.append(dict(frame=index,error='Wire values differ from recorded float32'))

    def finish(self):
        path=self.output/'original_conditioned42.npz'
        np.savez(path,target_q42=self.expected[:,0],kp42=self.expected[:,1],kd42=self.expected[:,2])
        report=dict(frames=self.frames,original_source=str(self.source),original_source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                    method='Exact three original pure conditioning definitions extracted via AST; original verified runtime envelopes and original GPU PolicyStep; no SDK initialization',
                    max_abs=dict(zip(('target_q42','kp42','kd42'),map(float,self.max_error))),
                    coordinates_exceeding_threshold=dict(zip(('target_q42_1e4','kp42_exact','kd42_exact'),map(int,self.over))),
                    wire_errors=self.errors,all_recorded_guards_allowed=not any('Guard' in x['error'] for x in self.errors),
                    conditioning_diagnostics_differences=self.modification_differences,
                    wire_roundtrip_scope='Every42slot q/kp/kd bitexact to recordedfloat32; mode10,dq/tau0,cmdidsequence,noheader/externalsessionmetadata',
                    passed=not self.errors and not self.over.any(),action_sent=False,
                    inert_lowcmd_sha256=hashlib.sha256(self.wirefile.read_bytes()).hexdigest())
        (self.output/'output_audit.json').write_text(json.dumps(report,indent=2))
        return report
