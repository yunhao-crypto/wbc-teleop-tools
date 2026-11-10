"""SONIC-v4 affine-target MorphMimic reference contract over the original SDK/WBC adapter.

The previous adapter applies a conservative rectangular ankle reference box,
which modifies mechanically valid full-V3 goals. This version uses the native
serial limits AND closed-loop motor workspace. Reference feasibility is separate from PD targets. The pinned main decoder
preserves affine PD targets; normalized action clipping remains +/-20.
No transport, inference engine or device is constructed by importing this file.
"""
from dataclasses import replace
import copy
import json
import math
import numpy as np

CONTRACT_KEY='morphmimic.tracking_contract.v1'
GUARD='native_serial_and_coupled_motor_workspace_v2'
PROFILE='vita02a_sonic_v2'
SERIAL_LIMITS=((-0.959931088596881,0.610865238198015),
               (-0.436332312998582,0.436332312998582))


def reference_ankle_class():
    from deployment.vita_s100_aorta_30dof.actor_parallel_ankle import MotorThetaAnkle,motor_angles,_links,_foot
    class NativeWorkspaceAnkle(MotorThetaAnkle):
        def __init__(self,names,block):
            super().__init__(names,block)
            self._reference_cache=None

        def project_reference(self,q):
            value=np.asarray(q,dtype=np.float64)
            if not np.isfinite(value).all():raise ValueError('Nonfinite ankle reference')
            cached=self._reference_cache
            if cached is not None and q is cached[0]and np.array_equal(value,cached[1]):
                return cached[2],cached[3]
            result=value.copy();low,high=self.motor_limits
            def feasible(pitch,roll):
                theta=motor_angles(pitch,roll,self.geometry)
                for (y,height),angle in zip(_links(self.geometry),theta):
                    anchor=np.asarray(_foot(pitch,roll,-self.geometry['lx'],y,0.))
                    crank=np.array([-self.geometry['lx']*math.cos(angle),y,height+self.geometry['lx']*math.sin(angle)])
                    if abs(np.linalg.norm(anchor-crank)-height)>1e-7:return False
                return min(theta)>=low-1e-7 and max(theta)<=high+1e-7
            for row in result.reshape(-1,result.shape[-1]):
                for p,r,sign in self.legs:
                    pitch=float(np.clip(row[p],*SERIAL_LIMITS[0]))
                    roll=sign*float(np.clip(row[r],*SERIAL_LIMITS[1]))
                    if not feasible(pitch,roll):
                        left,right=0.,1.
                        for _ in range(36):
                            middle=(left+right)/2
                            if feasible(middle*pitch,middle*roll):left=middle
                            else:right=middle
                        pitch,roll=left*pitch,left*roll
                    row[p],row[r]=pitch,sign*roll
            changed=np.any(np.abs(result-value)>1e-10,axis=-1)
            # Copy the key values as well as checking identity: mutation of a
            # live Motion.q cannot silently reuse an obsolete trajectory.
            self._reference_cache=(q,value.copy(),result,changed)
            return result,changed
    return NativeWorkspaceAnkle


def contract_from_metadata(math_api,metadata):
    from deployment.vita_s100_aorta_30dof.actor_morphmimic import MorphMimicContract
    raw=json.loads(metadata[CONTRACT_KEY])
    if raw.get('robot_profile')!=PROFILE or raw.get('reference_feasibility_guard')!=GUARD:
        raise ValueError('Expected the explicitly versioned versioned native-workspace reference guard')
    # Explicit transfer: never relabel a bounded-trained checkpoint as affine-trained.
    signed=raw.get('sonic_contract',{})
    execution=raw.get('execution_contract',{})
    transfer=raw.get('execution_transfer',{})
    if (signed.get('version')!=3 or signed.get('sha256')!='5d721a32756d4e363664c9082b40835f444a594ab24018ed28e976f2b8ed6809'
            or raw.get('training_contract')!=signed
            or execution.get('version')!=4 or execution.get('sha256')!='7b68798d018079585cf678c9ea2033e593a37aff7e31bcb2a442e7d2c5da088a'
            or transfer.get('mode')!='bounded_v3_actor_to_affine_v4'
            or transfer.get('source_checkpoint_sha256')!='7fda55aa7acc76f5c4979f77b36a12162d202a1213c4a6126c9742e850f9d063'
            or raw.get('checkpoint_sha256')!='7fda55aa7acc76f5c4979f77b36a12162d202a1213c4a6126c9742e850f9d063'
            or transfer.get('actor_weights_unchanged')is not True
            or raw.get('pd_target_semantics')!='affine_no_position_target_clip'):
        raise ValueError('Invalid or relabelled bounded-v3 to affine-v4 execution provenance')
    legacy=copy.deepcopy(raw)
    legacy['reference_feasibility_guard']='virtual_box_and_coupled_motor_v1'
    # Retain all inherited validation/observation/decode semantics. The new guard
    # deliberately makes an unmodified legacy parser fail rather than silently
    # applying its different reference box to a new model.
    validated=MorphMimicContract.from_metadata(math_api,{CONTRACT_KEY:json.dumps(legacy)})
    ankle=reference_ankle_class()(validated.names,raw['control']['ankle'])
    return replace(validated,ankle=ankle)


def create_policy(math_api,metadata,profile):
    from deployment.vita_s100_aorta_30dof.actor_morphmimic import policy_class
    return policy_class(math_api)(contract_from_metadata(math_api,metadata),profile)
