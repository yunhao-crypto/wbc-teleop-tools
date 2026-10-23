"""Execute vita_wifi.sh against fake nmcli/ip; no real network tools are called."""
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts/vita_wifi.sh'
FAKE=r'''import json,os,sys,stat
from pathlib import Path
base=Path(os.environ['WIFI_MOCK_DIR']);conf=json.loads((base/'mock.json').read_text());args=sys.argv[1:];name=Path(sys.argv[0]).name
created=json.loads((base/'created').read_text()) if (base/'created').exists() else {}
with (base/'calls.jsonl').open('a') as f:f.write(json.dumps([name]+args)+'\n')
if name=='ip':
 if '-j' in args:
  via='192.168.125.1'
  if conf.get('route_changes') and (base/'connected').exists():via='192.168.125.254'
  print(json.dumps([{'dst':'default','gateway':via,'dev':'eth0','expires':999 if not (base/'connected').exists() else 998},{'dst':'192.168.124.0/24','dev':'wlan0'}] if '-4' in args else []))
 else:print('default via192.168.125.1 deveth0')
 raise SystemExit(0)
if args[:2]==['radio','wifi']:
 print(conf.get('radio','enabled'));raise SystemExit(0)
if args[:2]==['-g','GENERAL.TYPE,GENERAL.STATE']:
 print(conf.get('type','wifi'));print(conf.get('state','30 (disconnected)'));raise SystemExit(0)
if args and args[0]=='-f':print('GENERAL.DEVICE:wlan0\nGENERAL.CONNECTION:mock');raise SystemExit(0)
if args[:2]==['-g','connection.uuid']:
 if not conf.get('exists') and not (base/'created').exists():raise SystemExit(10)
 if conf.get('missing_after_create') and (base/'created').exists():raise SystemExit(10)
 print(conf.get('identifier','11111111-1111-4111-8111-111111111111'));raise SystemExit(0)
if args and args[0]=='-g':
 vals={'connection.type':'802-11-wireless','connection.interface-name':conf.get('profile_iface','wlan0'),'802-11-wireless.ssid':conf.get('ssid',created.get('ssid','ASUS-5G')),'connection.autoconnect':'no','ipv4.method':'auto','ipv4.never-default':conf.get('never_default','yes'),'ipv4.ignore-auto-routes':'yes','ipv4.ignore-auto-dns':'yes','ipv6.method':'disabled','ipv4.gateway':'','ipv4.routes':'','802-11-wireless-security.key-mgmt':conf.get('security',created.get('security','wpa-psk'))}
 for k in args[1].split(','):print(vals[k])
 raise SystemExit(0)
if args[:2]==['connection','add']:
 if 'connection.uuid' in args:print("Error: failed to modify connection.uuid: the property can't be changed.",file=sys.stderr);raise SystemExit(2)
 if conf.get('add_fail'):print('mock add rejected',file=sys.stderr);raise SystemExit(4)
 (base/'created').write_text(json.dumps({'ssid':args[args.index('ssid')+1],'security':args[args.index('wifi-sec.key-mgmt')+1] if 'wifi-sec.key-mgmt' in args else ''}))
 print('Connection added');raise SystemExit(0)
if args[:4]==['--wait','30','connection','up']:
 if args[4:6]!=['uuid',conf.get('identifier','11111111-1111-4111-8111-111111111111')]:print('Wrong activation UUID',file=sys.stderr);raise SystemExit(99)
 if not conf.get('exists') and not (base/'created').exists():raise SystemExit(10)
 if conf.get('up_fail'):print('mock activation rejected',file=sys.stderr);raise SystemExit(4)
 if 'passwd-file' in args:
  p=Path(args[args.index('passwd-file')+1]);assert stat.S_IMODE(p.stat().st_mode)==0o600
  if conf.get('echo_secret'):print(p.read_text().split(':',1)[1].strip())
 (base/'connected').write_text('yes');print('Connection active');raise SystemExit(0)
print('Unexpected mock command',file=sys.stderr);raise SystemExit(99)
'''

class WifiTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='wifi_test_',dir=ROOT/'build')
        self.base=Path(self.tmp.name);self.bin=self.base/'bin';self.bin.mkdir()
        for name in ('nmcli','ip'):
            p=self.bin/name;p.write_text('#!'+sys.executable+'\n'+FAKE);p.chmod(0o755)
        self.secret='offline-fixture-password'
        self.config=self.base/'private.json'
        value={'schema':'vita_manual_wifi_v1','interface':'wlan0','networks':{'asus':{'ssid':'ASUS-5G','profile':'vita-manual-ASUS-5G','security':'wpa-psk','password_file':'asus.passwd'},'horizon':{'ssid':'Horizon-Device','profile':'vita-manual-Horizon-Device','security':'open'}}}
        self.config.write_text(json.dumps(value));self.config.chmod(0o600)
        self.password=self.base/'asus.passwd';self.password.write_text('802-11-wireless-security.psk:'+self.secret+'\n');self.password.chmod(0o600)
        self.settings={};self.env={**os.environ,'PATH':str(self.bin)+':'+os.environ['PATH'],'VITA_WIFI_CONFIG':str(self.config),'WIFI_MOCK_DIR':str(self.base)}

    def tearDown(self):self.tmp.cleanup()

    def run_script(self,*args):
        (self.base/'mock.json').write_text(json.dumps(self.settings))
        result=subprocess.run(['bash',str(SCRIPT),*args],env=self.env,capture_output=True,text=True,timeout=10)
        self.assertNotIn(self.secret,result.stdout+result.stderr)
        return result

    def calls(self):
        p=self.base/'calls.jsonl'
        return list(map(json.loads,p.read_text().splitlines())) if p.exists() else []

    def writes(self):
        return [x for x in self.calls() if x[0]=='nmcli' and (x[1:3]==['connection','add'] or x[1:5]==['--wait','30','connection','up'])]

    def test_default_status_has_no_mutation_and_needs_no_secret(self):
        self.config.unlink();self.password.unlink()
        r=self.run_script();self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(self.writes(),[])

    def test_dry_run_plans_creation_without_mutation_or_secret_arguments(self):
        r=self.run_script('dry-run');self.assertEqual(r.returncode,0,r.stderr)
        plan=json.loads(r.stdout);self.assertTrue(plan['dry_run']);self.assertFalse(plan['wifi_default_route'])
        self.assertEqual(self.writes(),[]);self.assertEqual(plan['nmcli_arguments'][0][:2],['connection','add'])
        self.assertEqual(plan['nmcli_arguments'][1][:2],['-g','connection.uuid'])
        self.assertIn('<UUID_AFTER_CREATE>',plan['nmcli_arguments'][-1])
        self.assertNotIn('connection.uuid',plan['nmcli_arguments'][0])
        self.assertIn('passwd-file',plan['nmcli_arguments'][-1]);self.assertNotIn(self.secret,json.dumps(self.calls()))

    def test_explicit_connect_creates_wlan0_profile_preserves_wired_routes(self):
        r=self.run_script('connect-asus');self.assertEqual(r.returncode,0,r.stderr)
        writes=self.writes();self.assertEqual(len(writes),2)
        add=writes[0];self.assertEqual(add[add.index('ifname')+1],'wlan0')
        for k,v in [('connection.autoconnect','no'),('ipv4.never-default','yes'),('ipv4.ignore-auto-routes','yes'),('ipv4.ignore-auto-dns','yes')]:self.assertEqual(add[add.index(k)+1],v)
        self.assertIn('passwd-file',writes[1]);self.assertNotIn(self.secret,json.dumps(self.calls()))
        self.assertNotIn('connection.uuid',add)
        self.assertEqual(writes[1][5:7],['uuid','11111111-1111-4111-8111-111111111111'])
        self.assertFalse(any('modify' in x or 'delete' in x or 'down' in x or 'on' in x for x in self.calls()))

    def test_existing_checked_profile_reused_without_modification(self):
        self.settings['exists']=True
        r=self.run_script('connect-asus');self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(len(self.writes()),1)

    def test_second_connect_reuses_system_generated_profile(self):
        self.settings['identifier']='22222222-2222-4222-8222-222222222222'
        for _ in range(2):
            r=self.run_script('connect-asus');self.assertEqual(r.returncode,0,r.stderr)
        writes=self.writes()
        self.assertEqual(len(writes),3)
        self.assertEqual(sum(x[1:3]==['connection','add'] for x in writes),1)
        self.assertEqual(writes[-1][6],self.settings['identifier'])

    def test_missing_uuid_after_creation_never_activates(self):
        self.settings['missing_after_create']=True
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0)
        self.assertEqual(len(self.writes()),1)

    def test_invalid_or_duplicate_generated_uuid_never_activates(self):
        for identifier in ('invalid', '11111111-1111-4111-8111-111111111111\n22222222-2222-4222-8222-222222222222'):
            with self.subTest(identifier=identifier):
                (self.base/'created').unlink(missing_ok=True)
                (self.base/'calls.jsonl').unlink(missing_ok=True)
                self.settings['identifier']=identifier
                r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0)
                self.assertEqual(len(self.writes()),1)

    def test_created_profile_is_validated_before_activation(self):
        self.settings['profile_iface']='eth0'
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0)
        self.assertEqual(len(self.writes()),1)

    def test_activation_failure_keeps_profile_for_explicit_retry(self):
        self.settings['up_fail']=True
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0)
        self.assertEqual(len(self.writes()),2)
        self.settings['up_fail']=False
        r=self.run_script('connect-asus');self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len(self.writes()),3)

    def test_existing_profile_on_ethernet_or_wrong_route_policy_rejected(self):
        self.settings.update(exists=True,profile_iface='eth0')
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0);self.assertEqual(self.writes(),[])
        self.settings.update(profile_iface='wlan0',never_default='no')
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0);self.assertEqual(self.writes(),[])

    def test_horizon_open_profile_needs_no_password(self):
        self.password.unlink()
        r=self.run_script('connect-horizon');self.assertEqual(r.returncode,0,r.stderr)
        self.assertNotIn('passwd-file',self.writes()[-1]);self.assertIn('Horizon-Device',self.writes()[0])

    def test_insecure_credentials_rejected_before_network_commands(self):
        self.password.chmod(0o644)
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0);self.assertEqual(self.calls(),[])

    def test_unmanaged_or_disabled_wifi_not_enabled_implicitly(self):
        self.settings['state']='10 (unmanaged)'
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0);self.assertEqual(self.writes(),[])
        self.settings.update(state='30 (disconnected)',radio='disabled')
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0);self.assertEqual(self.writes(),[])

    def test_creation_failure_does_not_attempt_connection(self):
        self.settings['add_fail']=True
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0);self.assertEqual(len(self.writes()),1)

    def test_route_change_detected_without_automatic_route_rewrite(self):
        self.settings['route_changes']=True
        r=self.run_script('connect-asus');self.assertNotEqual(r.returncode,0)
        self.assertTrue(all(x[1:] in (['-j','-4','route','show'],['-j','-6','route','show']) for x in self.calls() if x[0]=='ip'))

    def test_secret_echo_from_nmcli_is_redacted(self):
        self.settings['echo_secret']=True
        r=self.run_script('connect-asus');self.assertEqual(r.returncode,0,r.stderr);self.assertIn('[REDACTED]',r.stdout)

if __name__=='__main__':unittest.main(verbosity=2)
