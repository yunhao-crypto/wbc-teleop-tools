#!/usr/bin/env python3
"""Bounded ControlStatus subscriber-only probe; never creates a control client.

Local schema hash + successful decoding is not proof of live advertised schema
identity: this SDK callback omits the received schema hash. Query topic metadata
separately if that evidence is needed. No ExternalController instance is made.
"""
import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import threading
import time

EXPECTED_SCHEMA='cc897baf058703077ece4cbdc1f61b7c443905dbe1e24f204451ccf629afebf6'
TOPIC='/sdk/foot_humanoid/control/status'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds',type=float,default=5)
    p.add_argument('--startup-timeout',type=float,default=10)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--imports-only',action='store_true')
    a=p.parse_args()
    if not 0<a.seconds<=30 or not 0<a.startup_timeout<=30:raise ValueError('Probe duration must be bounded to30seconds')
    import aorta
    from aorta.publisher import Publisher
    import control_status_schema_meta as schema
    from locomotion_sdk.ControlStatus import ControlStatus
    from locomotion_aorta.transport import _decode_control_status
    from locomotion_aorta import ExternalController,PROTOCOL_VERSION
    expected_versions={'aorta-sdk':'2026.9.10+humanoid.7100871','aorta-msgs':'2026.9.10+humanoid.7100871','flatbuffers':'25.9.23','locomotion-aorta':'0.1.0'}
    versions={name:importlib.metadata.version(name) for name in expected_versions}
    if versions!=expected_versions:raise ValueError('Unexpected SDK dependencies: '+repr(versions))
    bfbs_hash=hashlib.sha256(schema.SCHEMA_BFBS).hexdigest()
    if schema.SCHEMA_HASH_HEX!=EXPECTED_SCHEMA or bfbs_hash!=EXPECTED_SCHEMA:raise ValueError('Local ControlStatus schema mismatch')
    # Inspect compatibility without constructing any controller/transport.
    api={name:callable(getattr(aorta.Node,name,None)) for name in ('group','create_subscriber','create_subscriber_typed','create_publisher_typed','close')}
    api['Publisher.publish_typed']=callable(getattr(Publisher,'publish_typed',None))
    api.update({'ExternalController.'+name:callable(getattr(ExternalController,name,None)) for name in ('wait_lowstate','enter_external','exit_external','publish','close')})
    if not all(api.values()):raise ValueError('Required API absent: '+repr(api))
    def forbidden(*args,**kwargs):raise RuntimeError('Readonly probe forbids publishers/control clients/actions/services')
    blocked=[]
    for name in dir(aorta.Node):
        if name.startswith('create_') and any(part in name for part in ('publisher','client','action','service')):
            setattr(aorta.Node,name,forbidden);blocked.append(name)
    report=dict(scope='SDK imports and bounded subscriber only',versions=versions,required_api_present=api,
                local_schema_hash=EXPECTED_SCHEMA,local_bfbs_sha256=bfbs_hash,live_schema_hash_verified=False,
                schema_limitation='SDK subscriber discards received schema hash; local schema and live decode/protocol_version checked, exact advertised hash requires separate readonly topic metadata query',
                control_clients_created=0,publishers_created_by_probe=0,leases_requested=0,actions_sent=0,
                forbidden_factories=blocked,profile_path=os.environ.get('ZENOH_SESSION_CONFIG_URI'),samples=[],errors=[])
    if a.imports_only:
        report['passed']=True;report['live_connection_tested']=False
        a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0
    if os.environ.get('ZENOH_SESSION_CONFIG_URI')!='/app_param/zenoh/s100_session_peer.json5':
        raise ValueError('Use verified robot-local peer profile before probe')
    node=None;subscriber=None;first=threading.Event();stop=threading.Event();lock=threading.Lock()
    def callback(payload,context):
        if stop.is_set():return
        now=time.monotonic_ns()
        try:
            if not 8<=len(payload)<=65536:raise ValueError('Unexpected ControlStatus payload size')
            v=ControlStatus.GetRootAs(payload)
            status=_decode_control_status(v)
            if status.protocol_version!=PROTOCOL_VERSION:raise ValueError('Live protocol_version mismatch')
            if not status.current_action:raise ValueError('Empty live current_action')
            row=dataclasses.asdict(status);row['phase']=status.phase.value
            row.update(received_monotonic_ns=now,payload_bytes=len(payload))
            with lock:
                if len(report['samples'])>=10000:raise ValueError('Unexpected excessive status rate')
                report['samples'].append(row)
            first.set()
        except Exception as e:
            with lock:report['errors'].append(type(e).__name__+': '+str(e))
            first.set();stop.set()
    try:
        node=aorta.Node('vita-controlstatus-readonly-'+str(time.monotonic_ns()),'default')
        report['group']=node.group()
        if report['group']!='default':raise ValueError('Unexpected Aorta group')
        subscriber=node.create_subscriber(TOPIC,callback,qos=aorta.QoS.state_update(),
            options=aorta.SubscriberOptions(include_receive_timestamp=True,enable_header_context=True))
        if not first.wait(a.startup_timeout):raise TimeoutError('No ControlStatus received')
        stop.wait(a.seconds)
        stats=subscriber.runtime_stats();report['runtime_stats']=dataclasses.asdict(stats)
    except Exception as e:
        report['errors'].append(type(e).__name__+': '+str(e))
    finally:
        stop.set()
        for obj in (subscriber,node):
            if obj is not None:
                try:obj.close()
                except Exception as e:report['errors'].append('cleanup '+type(e).__name__+': '+str(e))
    samples=report['samples'];report['live_connection_tested']=True;report['sample_count']=len(samples)
    report['latest_status']=samples[-1] if samples else None
    report['phases']=sorted({x['phase'] for x in samples});report['actions']=sorted({x['current_action'] for x in samples})
    active={'PREPARING','ARMED','TRANSITIONING','RUNNING','TIMEOUT_HOLD'}
    report['active_external_owner_seen']=any(x['current_action']=='EXTERNAL' and x['phase'] in active for x in samples)
    report['passed']=bool(samples) and not report['errors']
    report['available_by_original_client_precheck']=bool(report['passed'] and not report['active_external_owner_seen'])
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('samples','forbidden_factories')},indent=2))
    return 0 if report['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
