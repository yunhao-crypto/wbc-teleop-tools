"""Protocol byte/schema checks without any network connection or robot API."""
import json
from pathlib import Path
import fake_pico_host_protocol as p

assert p.encode_frame(p.CONNECT,b'X|1',tail_seconds=7).hex()=='3f1903000000587c310700000000000000a5'
body=p.SyntheticBodyGenerator(7).frame(17);stamp=1700000000123456789
payload=p.body_state_payload(body,stamp);inner=json.loads(json.loads(payload)['value'])
assert inner['Body']['timeStampNs']==stamp
assert [[float(v) for v in j['p'].split(',')] for j in inner['Body']['joints']]==body
class Sink:
    def __init__(self):self.data=bytearray()
    def sendall(self,data):self.data.extend(data)
sink=Sink();client=p.FakePicoClient(sink);client.register(settle_seconds=0);client.send_body(body,timestamp_ns=stamp)
decoder=p.FrameDecoder();messages=[]
for byte in sink.data:messages.extend(decoder.feed(bytes([byte])))
assert [m['command'] for m in messages]==[0x19,0x19,0x23,0x6d]
assert messages[-1]['payload']==payload and not decoder.pending
for command in [0x5f,0x71,0x72]:
    try:p.encode_frame(command,b'')
    except ValueError:pass
    else:raise AssertionError('Control command allowed')
try:client.send_body(body,timestamp_ns=stamp)
except ValueError:pass
else:raise AssertionError('Duplicate timestamp accepted')
report=dict(exact_header_tail_golden_vector=True,byte_at_a_time_fragmentation=True,double_json_native_schema=True,body_24x7_float_roundtrip_exact=True,strict_monotonic_stamp=True,only_input_commands_allowed=True,packets=len(messages),payload_bytes=len(payload),network_connections=0,robot_actions=0)
output=Path(__file__).resolve().parents[1]/'outputs/fakeprotocol_selftest.json';output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
