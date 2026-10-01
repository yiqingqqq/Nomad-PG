"""Stop only the designated training process after its third durable epoch."""
import json
import os
from pathlib import Path
import signal
import time

PID = 1565
ROOT = Path('/root/autodl-tmp/RVN-Bench-interactive-runs/ppo_v2_frozen_bs256')
PROC = Path(f'/proc/{PID}')

def identity():
    return (PROC / 'stat').read_text().split(') ', 1)[1].split()[19]

started = identity()
cmd = (PROC / 'cmdline').read_bytes()
assert b'train_vint_pg_ppo_v2.py' in cmd and str(ROOT).encode() in cmd
print('ARMED: stop PID 1565 after epoch index 2 is saved', flush=True)
while PROC.exists() and identity() == started:
    path = ROOT / 'history.json'
    history = json.loads(path.read_text())['history'] if path.exists() else []
    if history and history[-1]['epoch'] >= 2:
        # history is atomically published only AFTER latest_train.pth is saved.
        assert (ROOT / 'latest_train.pth').stat().st_size > 0
        os.kill(PID, signal.SIGTERM)
        for _ in range(100):
            if not PROC.exists():
                break
            time.sleep(0.1)
        report = dict(stopped=not PROC.exists(), pid=PID,
                      saved_epoch=history[-1]['epoch'], timestamp=time.time(),
                      resume_checkpoint=str(ROOT / 'latest_train.pth'))
        temporary = ROOT / 'stop_after_round3.json.tmp'
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(ROOT / 'stop_after_round3.json')
        print(json.dumps(report), flush=True)
        break
    time.sleep(0.5)
else:
    print('Target exited or changed before requested boundary', flush=True)
