"""Offline IMU spectral/kinematic diagnostics with explicit coordinate/clock declarations."""
from pathlib import Path
import numpy as np
from .actor_control_compare import _rotation, _json
from .actor_control_shadow import sha256


def audit_imu(npz_path, metadata_path, start_s=0., end_s=None, max_gap_s=.05, min_segment_s=4.):
    source_hash=sha256(npz_path);metadata_hash=sha256(metadata_path)
    metadata=_json(Path(metadata_path).read_text())
    if not isinstance(metadata,dict):raise ValueError('Metadata must be a JSON object')
    if metadata.get('schema')!='wtt.imu_analysis.v1': raise ValueError('Need wtt.imu_analysis.v1 metadata')
    if not isinstance(metadata.get('evidence'),str) or not metadata['evidence'].strip(): raise ValueError('Missing convention/clock evidence or explicit operator declaration')
    time=metadata['time'];accel=metadata['acceleration'];quat=metadata['quaternion']
    if any(not isinstance(v,dict) for v in (time,accel,quat,metadata.get('sensor',{}))):raise ValueError('Convention declarations must be objects')
    if not isinstance(time.get('domain'),str) or not time['domain'].strip():raise ValueError('Timestamp unit/domain must be explicitly declared')
    if type(time.get('zero_is_missing',True)) is not bool:raise ValueError('zero_is_missing must be boolean')
    if time.get('unit') not in ('s','ms','us','ns') or time.get('domain') in (None,'','unknown'):
        raise ValueError('Timestamp unit/domain must be explicitly declared')
    if time.get('semantics') not in ('sample','ingested','body_state_proxy'): raise ValueError('Undeclared time semantics')
    if accel.get('unit')!='m/s^2' or accel.get('frame')!='body' or metadata.get('world_up_axis')!='z':
        raise ValueError('Need body acceleration m/s^2 and explicit world z-up')
    if quat.get('order')!='wxyz' or quat.get('direction')!='body_to_world': raise ValueError('Unsupported quaternion convention')
    for x in (start_s,max_gap_s,min_segment_s):
        if not np.isfinite(x) or x<0: raise ValueError('Invalid time bounds')
    if max_gap_s<=0 or min_segment_s<4.: raise ValueError('Use a positive gap threshold and at least 4-second spectral segments')
    if end_s is not None and (not np.isfinite(end_s) or end_s<=start_s): raise ValueError('Invalid end time')
    with np.load(npz_path,allow_pickle=False) as data:
        raw=np.asarray(data[time['key']]);a=np.asarray(data[accel['key']],dtype=float);q=np.asarray(data[quat['key']],dtype=float)
        if raw.ndim!=1 or raw.dtype.kind not in 'iuf' or a.shape!=(len(raw),3) or q.shape!=(len(raw),4): raise ValueError('Malformed IMU arrays')
        if not len(raw): raise ValueError('Empty IMU data')
        keep=np.ones(len(raw),dtype=bool)
        sensor=metadata.get('sensor',{})
        for field,value_field in [('selector_key','value'),('status_key','valid_status')]:
            if field in sensor:
                selector=np.asarray(data[sensor[field]])
                if selector.shape!=raw.shape: raise ValueError('Malformed sample selector')
                keep &= selector==sensor[value_field]
        validity=sensor.get('require_boolean_fields',{})
        if not isinstance(validity,dict):raise ValueError('Boolean sample validity declarations must be an object')
        for key,wanted in validity.items():
            values=np.asarray(data[key])
            if type(wanted) is not bool or values.dtype!=np.bool_ or values.shape!=raw.shape:
                raise ValueError('Invalid boolean sample validity declaration')
            keep &= values==wanted
        if not sensor.get('unique_samples') is True: raise ValueError('Input must declare deduplication by real sensor sample identity')
        if time.get('zero_is_missing',True): keep &= raw!=0
        selected_count=int(keep.sum())
        raw,a,q=raw[keep],a[keep],q[keep]
        if not np.isfinite(raw).all() or not np.isfinite(a).all() or not np.isfinite(q).all():raise ValueError('Nonfinite selected IMU data')
        gyro=metadata.get('gyro');g=None
        if gyro is not None:
            if gyro.get('unit')!='rad/s' or gyro.get('frame')!='body':raise ValueError('Need explicitly body-frame rad/s gyro')
            g=np.asarray(data[gyro['key']],dtype=float)
            if g.shape!=(len(keep),3):raise ValueError('Malformed gyro')
            g=g[keep]
            if not np.isfinite(g).all():raise ValueError('Nonfinite selected gyro')
    if len(raw)<2 or np.any(raw[1:]<=raw[:-1]): raise ValueError('Selected timestamps are missing, duplicated, or nonmonotonic; do not sort away clock resets')
    # Subtract integer origin BEFORE conversion to float; preserve uint64 precision.
    origin=raw[0].item();factor={'s':1.,'ms':1e-3,'us':1e-6,'ns':1e-9}[time['unit']]
    t=np.asarray(raw-raw[0],dtype=float)*factor
    end_s=float(t[-1]) if end_s is None else end_s
    window=(t>=start_s)&(t<=end_s);t,a,q=t[window],a[window],q[window]
    if g is not None:g=g[window]
    if len(t)<2:raise ValueError('No samples in requested window')
    norms=np.linalg.norm(q,axis=1)
    if np.any((norms<.8)|(norms>1.2)):raise ValueError('Quaternion norm outside diagnostic normalization allowance')
    rotation=_rotation(q);world=np.einsum('nij,nj->ni',rotation,a)
    cuts=np.r_[0,np.where(np.diff(t)>max_gap_s)[0]+1,len(t)]
    segments=[];skipped=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        tt=t[lo:hi];duration=float(tt[-1]-tt[0])
        if duration<min_segment_s or len(tt)<3:
            skipped.append({'start_s':float(tt[0]),'end_s':float(tt[-1]),'reason':'short_segment'});continue
        rate=1/float(np.median(np.diff(tt)))
        if rate<=16.:
            skipped.append({'start_s':float(tt[0]),'end_s':float(tt[-1]),'reason':'insufficient_sampling_for_8Hz'});continue
        grid=np.arange(tt[0],tt[-1],1/rate)
        signal=np.interp(grid,tt,world[lo:hi,2]);signal-=signal.mean();w=np.hanning(len(grid))
        power=abs(np.fft.rfft(signal*w))**2/(len(grid)*np.sum(w*w))
        power[1:]*=2
        if len(grid)%2==0:power[-1]/=2
        freq=np.fft.rfftfreq(len(grid),1/rate);band=(freq>=2)&(freq<=8)
        rms=float(np.sqrt(power[band].sum()))
        item=dict(start_s=float(tt[0]),end_s=float(tt[-1]),samples=len(tt),resample_hz=rate,
                  max_sample_gap_s=float(np.diff(tt).max()),frequency_resolution_hz=float(rate/len(grid)),
                  world_z_2_8hz_rms_m_s2=rms)
        if g is not None:
            from scipy.spatial.transform import Rotation
            r=Rotation.from_matrix(rotation[lo:hi]);angular=(r[:-1].inv()*r[1:]).as_rotvec()/np.diff(tt)[:,None]
            measured=(g[lo:hi-1]+g[lo+1:hi])/2
            item['quaternion_gyro_rmse_rad_s']=np.sqrt(np.mean((angular-measured)**2,axis=0)).tolist()
            item['quaternion_gyro_correlation']=[float(np.corrcoef(angular[:,i],measured[:,i])[0,1])
                if np.std(angular[:,i])>1e-10 and np.std(measured[:,i])>1e-10 else None for i in range(3)]
        segments.append(item)
    duration=sum(s['end_s']-s['start_s'] for s in segments)
    aggregate=(float(np.sqrt(sum(s['world_z_2_8hz_rms_m_s2']**2*(s['end_s']-s['start_s']) for s in segments)/duration)) if duration else None)
    if sha256(npz_path)!=source_hash or sha256(metadata_path)!=metadata_hash:raise ValueError('Input changed during analysis')
    return dict(schema='wtt.imu_audit.v1',status='DIAGNOSTIC_ONLY' if segments else 'INSUFFICIENT_COVERAGE',
        source_npz_sha256=source_hash,metadata_sha256=metadata_hash,
        declared_metadata=metadata, timestamp_origin_raw=origin,selected_samples=selected_count,window_samples=len(t),
        requested_window_s=[start_s,end_s],valid_segment_duration_s=duration,
        requested_duration_coverage_fraction=duration/(end_s-start_s),
        world_z_2_8hz_rms_m_s2=aggregate,segments=segments,excluded_segments=skipped,
        limits=['Declared clock/frame conventions are not independently calibrated by this tool.',
                'BodyState-associated or ingested timestamps are proxies, not physical sample acquisition time.' if time['semantics']!='sample' else 'Sample timestamps require the declared clock unit and semantics to be correct.',
                'No interpolation across large gaps; Hann FFT, no zero padding, no inferred displacement.',
                'HIP point acceleration differs from simulated root-origin acceleration; attitude error can leak gravity into the band.',
                'Statistics are not a stability/safety verdict or evidence of unique hardware causality.'])
