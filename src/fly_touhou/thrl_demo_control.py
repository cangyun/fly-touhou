"""Local controls for one owned, visible, frozen neural-policy demonstration."""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import uuid

from .thrl_control import alive, controller_lock, process_identity
from .thrl_durable_records import write_json as write
from .thrl_study_io import read_shared_json as read, utc_now

CATALOGUE_V1 = 'B1-real-neural-demo-catalogue-v1'
CATALOGUE_V2 = 'B1-real-neural-demo-catalogue-v2'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            result.update(block)
    return result.hexdigest()


def validate_selection(catalogue, key):
    schema = catalogue.get('schema')
    if schema not in (CATALOGUE_V1, CATALOGUE_V2):
        raise ValueError('Unrecognized demonstration catalogue')
    selected = catalogue['models'][key]
    if not selected.get('qualified_neural_policy'):
        raise ValueError('A qualified frozen neural policy is required')
    if schema == CATALOGUE_V1:
        if (selected.get('formal_test_episodes') != 20 or selected.get('formal_test_clears', 0) <= 0
                or selected.get('known_clear_native_seed') != catalogue['known_native_seed']):
            raise ValueError('The demonstration seed needs an actual recorded neural clear')
    else:
        _validate_model_evidence(selected)
    if digest(selected['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('The selected checkpoint has changed')
    return selected


def _validate_model_evidence(selected):
    scope = selected.get('evidence_scope')
    episodes, clears = selected.get('evaluation_episodes'), selected.get('evaluation_clears')
    seed, environment_seed = selected.get('known_clear_native_seed'), selected.get('known_clear_environment_seed')
    if (selected.get('model_family') != 'B1' or selected.get('stable_completion_claimed') is not False
            or scope not in ('formal', 'development') or type(episodes) is not int or episodes != 20
            or type(clears) is not int or not 0 < clears <= episodes
            or type(seed) is not int or not 0 <= seed <= 65535
            or type(environment_seed) is not int or not 0 <= environment_seed < 2**32):
        raise ValueError('A B1 policy needs explicit20-game evidence and its own recorded clear seed')
    from .thrl_training_mode import seed_split, draw_native_seed
    import numpy as np
    split = 'development' if scope == 'development' else 'test'
    if (selected.get('native_seed_split') != split or seed_split(seed) != split
            or draw_native_seed(np.random.default_rng(environment_seed), split) != seed):
        raise ValueError('The model-specific environment/native seed partition differs')
    if not all(selected.get(name) for name in ('source_runtime', 'source_manifest_sha256',
                                               'report', 'report_sha256')):
        raise ValueError('Model-specific evidence and frozen runtime are required')
    if digest(selected['report']) != selected['report_sha256']:
        raise ValueError('The model evidence report has changed')
    report = read(selected['report'])
    rows = report.get('episode_results', [])
    if (report.get('episodes') != episodes or report.get('clears') != clears or len(rows) != episodes
            or sum(row.get('outcome') == 'clear' for row in rows) != clears
            or any(row.get('outcome') not in ('clear', 'gameover') or row.get('native_frames', 0) <= 0 for row in rows)):
        raise ValueError('The model evidence is not a completed20-game report')
    known = [row for row in rows if row.get('native_seed') == seed]
    if (len(known) != 1 or known[0].get('outcome') != 'clear'
            or known[0].get('evaluation_seed') != environment_seed
            or known[0].get('stage_history') != [1, 2, 3, 4, 5, 6]):
        raise ValueError('The model-specific demonstration seed needs a recorded six-stage neural clear')
    if scope == 'development':
        if selected.get('formal_test_episodes', 0) or selected.get('formal_test_clears', 0):
            raise ValueError('Development evidence cannot be labelled as a formal test')
        if not selected.get('evaluation_plan') or digest(selected['evaluation_plan']) != selected.get('evaluation_plan_sha256'):
            raise ValueError('Development evidence needs its frozen evaluation plan')
        plan = read(selected['evaluation_plan'])
        policy = plan.get('policies', {}).get(selected.get('evaluation_policy_key'), {})
        if (plan.get('native_seed_split') != 'development' or plan.get('episodes_each') != 20
                or policy.get('sha256') != selected['checkpoint_sha256']
                or plan.get('source_runtime') != selected.get('policy_source_runtime')
                or plan.get('source_manifest_sha256') != selected.get('policy_source_manifest_sha256')):
            raise ValueError('Development policy/source provenance differs from the selected B1')


def selection_description(catalogue, key):
    """Display evidence without touching a process, runtime or checkpoint."""
    selected = catalogue['models'][key]
    if catalogue.get('schema') != CATALOGUE_V2:
        return catalogue.get('selection_note', '已知通关开局展示，不计入新的独立测试。')
    scope = '开发评估' if selected.get('evidence_scope') == 'development' else '原固定独立测试'
    return (f"{scope}：{selected['evaluation_clears']}/{selected['evaluation_episodes']} 通关；"
            f"展示开局 native {selected['known_clear_native_seed']}。"
            + ('开发成绩不代表正式稳定通关率。' if selected.get('evidence_scope') == 'development' else '')
            + '本次为已知开局实时演示，不计入新的测试成绩。')


def make_launch_plan(catalogue, key, output):
    """Resolve one model's seeds/runtime; constructing a plan launches nothing."""
    selected = validate_selection(catalogue, key)
    modern = catalogue.get('schema') == CATALOGUE_V2
    runtime = Path(selected['source_runtime'] if modern else catalogue['source_runtime'])
    manifest_sha = selected['source_manifest_sha256'] if modern else catalogue['source_manifest_sha256']
    if digest(runtime/'source_manifest.json') != manifest_sha:
        raise ValueError('演示运行版本校验失败')
    native_seed = selected['known_clear_native_seed'] if modern else catalogue['known_native_seed']
    environment_seed = selected['known_clear_environment_seed'] if modern else catalogue['known_environment_seed']
    launch_schema = selected.get('launch_catalogue_schema', CATALOGUE_V2) if modern else CATALOGUE_V1
    if launch_schema not in (CATALOGUE_V1, CATALOGUE_V2):
        raise ValueError('Unsupported frozen demonstration launch schema')
    if modern and selected['evidence_scope'] == 'development' and launch_schema != CATALOGUE_V2:
        raise ValueError('Development B1 needs a runtime that understands model-specific evidence')
    return {**catalogue, 'schema': launch_schema, 'catalogue_schema': catalogue['schema'],
        'source_runtime': str(runtime), 'source_manifest_sha256': manifest_sha,
        'selected_model': key, 'selected': selected, 'output': str(output),
        'known_native_seed': native_seed, 'known_environment_seed': environment_seed,
        'environment_seed': environment_seed,
        'environment_role': 'development' if modern and selected['evidence_scope'] == 'development' else 'test',
        'evidence_scope': selected.get('evidence_scope', 'formal'),
        'known_seed_demonstration': True, 'formal_test': False,
        'optimizer_updates': 0, 'teacher_actions': 0, 'safety_takeovers': 0}


class DemoControl:
    def __init__(self, root):
        self.root = Path(root)

    def catalogue(self):
        return read(self.root/'catalogue.json')

    def status(self):
        try:
            active = read(self.root/'active.json')
        except FileNotFoundError:
            return dict(status='idle', running=False)
        try:
            state = read(Path(active['output'])/'status.json')
        except (FileNotFoundError, ValueError):
            state = dict(status='starting')
        running = alive(state.get('worker_identity')) or alive(active.get('launcher_identity'))
        if not running and state.get('status') in ('starting', 'running', 'paused', 'menu_resume'):
            state = {**state, 'status': 'interrupted'}
        return {**active, **state, 'running': running}

    def start(self, key=None):
        with controller_lock(self.root):
            if self.status()['running']:
                raise RuntimeError('已有实机演示在运行，请先停止或等待结束')
            catalogue = self.catalogue()
            key = key or catalogue['default_model']
            selected = validate_selection(catalogue, key)
            import psutil
            worlds = sum((p.info['name'] or '').lower() == 'th14.exe'
                for p in psutil.process_iter(['name']))
            if worlds >= 30:
                raise RuntimeError('当前已有30个原作实例；请等评估结束后启动演示')
            output = Path(catalogue['output_root'])/(utc_now().replace(':', '').replace('+', '_')+'-'+uuid.uuid4().hex[:6])
            plan = make_launch_plan(catalogue, key, output)
            runtime = Path(plan['source_runtime'])
            output.mkdir(parents=True, exist_ok=False)
            write(output/'plan.json', plan)
            command = [str(catalogue.get('python') or Path(sys.executable).parent/'python.exe'), '-B',
                str(runtime/'scripts/run_b1_real_demonstration.py'), '--plan', str(output/'plan.json')]
            environment = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
            for name in ('TEMP', 'TMP', 'TMPDIR'):
                environment[name] = catalogue['temporary_directory']
            environment.pop('FLY_TOUHOU_DESKTOP_NAME', None)
            with (output/'stdout.log').open('ab') as stdout, (output/'stderr.log').open('ab') as stderr:
                process = subprocess.Popen(command, cwd=runtime, env=environment,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            active = dict(output=str(output), model=key, model_label=selected['label'],
                launcher_identity=process_identity(process.pid), started_utc=utc_now(),
                known_seed_demonstration=True, formal_test=False)
            write(self.root/'active.json', active)
            return active

    def stop(self):
        with controller_lock(self.root):
            state = self.status()
            if state['running']:
                (Path(state['output'])/'STOP').touch(exist_ok=True)
            return state

    def pause(self):
        with controller_lock(self.root):
            state = self.status()
            if not state['running']:
                return state
            path = Path(state['output'])/'PAUSE'
            if path.exists():
                path.unlink()
            else:
                path.touch(exist_ok=True)
            return state
