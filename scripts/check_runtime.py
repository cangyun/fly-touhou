import json
from pathlib import Path
from fly_touhou.background import BackgroundTouhouEnv

root = Path(__file__).resolve().parents[1]
env = BackgroundTouhouEnv(root/'games/th14_trial/extracted/app/th14.exe',
                         root/'profiles/th14_trial_0.01b.json')
evidence = {'source': 'native-process', 'snapshots': []}
try:
    for episode in range(2):
        observation, info = env.reset()
        print('RESET', episode, env.pid, env.snapshot.stage_frame, flush=True)
        evidence['snapshots'].append(env.snapshot.to_dict())
        for step in range(450):
            observation, reward, done, truncated, info = env.step(0)
            if step % 60 == 0 or done:
                print('STEP', step, env.snapshot.stage_frame, len(env.snapshot.bullets), done, flush=True)
                evidence['snapshots'].append(env.snapshot.to_dict())
            if done or truncated:
                break
    evidence['window_audit'] = env.controller.audit_windows()
    evidence['passed'] = True
finally:
    env.close()
    (root/'artifacts/runtime_verified.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
