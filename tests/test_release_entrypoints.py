"""CPU-only checks of the public setup and frozen-evaluation contract."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseTests(unittest.TestCase):
    def test_cuda_compilation_sources_are_present(self):
        for name in ('deterministic_csr_spmm.cu', 'fused_edge_gradient.cu'):
            self.assertTrue((ROOT/'src/fly_touhou/csrc'/name).is_file())

    def test_frozen_command_uses_branch_assets_and_seeds(self):
        module = load_script('run_release')
        args = argparse.Namespace(game_exe=Path('games/with spaces/th14.exe'), checkpoint=None,
                                  connectome=None, output=None, cache=None, opening_seeds=None)
        plan, env = module.make_plan(args)
        release = json.loads((ROOT/'configs/release.json').read_text())
        self.assertEqual(plan['command'][-len(release['opening_seeds']):], list(map(str, release['opening_seeds'])))
        self.assertEqual(plan['command'][plan['command'].index('--action-selection')+1], 'family_mode')
        self.assertEqual(plan['command'][plan['command'].index('--episodes')+1], str(len(release['opening_seeds'])))
        self.assertTrue(Path(plan['environment_config']['profile']).is_absolute())
        self.assertIn('with spaces', plan['environment_config']['game_exe'])
        self.assertNotIn('--execute', plan['command'])
        self.assertIn('FLY_TOUHOU_CACHE_ROOT', env)

    def test_explicit_opening_seeds_replace_default_plan(self):
        args = argparse.Namespace(game_exe=Path('th14.exe'), checkpoint=None, connectome=None,
                                  output=Path('outputs/custom'), cache=None, opening_seeds=[1, 2, 3])
        plan, _ = load_script('run_release').make_plan(args)
        self.assertEqual(plan['command'][-3:], ['1', '2', '3'])

    def test_native_installer_rejects_unexpected_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with zipfile.ZipFile(root/'bad.zip', 'w') as archive:
                archive.writestr('../outside.txt', 'bad')
            with self.assertRaises(ValueError):
                load_script('setup_release').install_native(root/'bad.zip', root)

    def test_release_assets_agree_with_report(self):
        release = json.loads((ROOT/'configs/release.json').read_text())
        if release['branch'] == 'main':
            report = json.loads((ROOT/'results/clear-v6.json').read_text(encoding='utf-8'))
            self.assertEqual(report['clears'], 1)
            self.assertEqual(release['opening_seeds'], [report['episode_results'][0]['evaluation_seed']])
            self.assertEqual(release['original_cycle'], 87)
        else:
            report = json.loads((ROOT/'results/cycle101-frozen20.json').read_text(encoding='utf-8'))
            self.assertEqual(release['opening_seeds'], [e['evaluation_seed'] for e in report['episode_results']])
            self.assertEqual(report['clears'], 0)
        self.assertEqual(report['teacher_actions'], 0)
        self.assertEqual(report['optimizer_updates'], 0)


if __name__ == '__main__':
    unittest.main()
