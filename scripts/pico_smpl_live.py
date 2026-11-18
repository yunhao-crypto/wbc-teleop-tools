#!/usr/bin/env python3
"""Compatibility entry; same explicit pico_smpl selection and Git state machine."""
import sys
from pico_actor_aligned import main
if __name__=='__main__':
    if any(a=='--policy' or a.startswith('--policy=') for a in sys.argv[1:]):
        raise SystemExit('This entry fixes the policy to pico_smpl')
    sys.argv.extend(['--policy','pico_smpl'])
    main()
