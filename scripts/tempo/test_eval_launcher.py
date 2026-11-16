"""Logger compatibility is checked before GPU work; simulator results survive failures."""
import contextlib
import ast
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
import evaluate_smolvla as runner


class EvalLauncherTests(unittest.TestCase):
    def test_websocket_disabled_keepalive_uses_python_none_on_both_endpoints(self):
        with TemporaryDirectory() as d:
            root=Path(d);argv=self.fixture(root)+['--dry-run']
            config=root/'eval.yaml';recipe=yaml.safe_load(config.read_text())
            recipe['launcher']['websocket']={'ws_ping_interval_s':None,'ws_ping_timeout_s':None,'request_timeout_s':120}
            recipe['evaluation']['seed_start']=100001
            config.write_text(yaml.safe_dump(recipe))
            with patch('sys.argv',argv),patch.object(runner.socket,'socket') as socket,\
                 contextlib.redirect_stdout(io.StringIO()) as output:
                socket.return_value.__enter__.return_value.getsockname.return_value=('127.0.0.1',1234)
                runner.main()
            commands=json.loads(output.getvalue())
            value=next(v.split('=',1)[1] for v in commands['server'] if v.startswith('ws_ping_interval_s='))
            self.assertIsNone(ast.literal_eval(value))
            additional=commands['simulator'][commands['simulator'].index('--additional_info')+1]
            self.assertIn('ws_ping_interval_s=None',additional)
            self.assertIn('seed_start=100001',additional)

    def test_existing_eval_run_can_be_resumed_by_explicit_id(self):
        command=runner.logger_command(Path('/root'),{'policy_python':'python','policy_name':'TEMPO_SmolVLA'},
                                      {'task':'cover_blocks','env_cfg':'aloha_agilex','seed':0},
                                      {'run_name':'eval','group':'group','training_run_id':'train',
                                       'resume_run_id':'existing-eval'},Path('/100000'),Path('/output'))
        self.assertEqual(command[-2:],['--wandb-run-id','existing-eval'])

    def fixture(self, root):
        checkpoint = root/'100000';checkpoint.mkdir();(checkpoint/'model.safetensors').write_bytes(b'model')
        recipe = {'launcher': {'policy_name': 'TEMPO_SmolVLA', 'policy_python': 'policy-python',
                  'simulator_python': 'sim-python', 'checkpoint': str(checkpoint), 'output_root': str(root/'eval'),
                  'wandb_logger_script': str(root/'bundled/logger.py'), 'wandb_metadata_helper': str(root/'bundled/tempo_wandb.py')},
                  'evaluation': {'task': 'battery_try', 'env_cfg': 'aloha_agilex', 'seed': 0, 'episodes': 100,
                  'num_workers': 4, 'action_type': 'joint', 'task_config': 'demo_clean', 'instruction_type': 'seen',
                  'expert_check': True, 'frequency': 30, 'max_seed_attempts': 5000},
                  'wandb': {'enabled': True, 'entity': 'test', 'project': 'test', 'group': 'group',
                            'run_name': 'train_eval', 'training_run_id': 'train'}}
        config = root/'eval.yaml';config.write_text(yaml.safe_dump(recipe))
        return ['eval', '--root', str(root), '--config', str(config)]

    def test_incompatible_logger_fails_before_server_or_simulator_start(self):
        with TemporaryDirectory() as d:
            root = Path(d);argv = self.fixture(root)
            with patch('sys.argv', argv), patch.object(runner.socket, 'socket') as socket, patch.object(runner.subprocess, 'run',
                    side_effect=subprocess.CalledProcessError(2, 'old logger')) as run,\
                 patch.object(runner.subprocess, 'Popen') as launch:
                socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 1234)
                with self.assertRaises(subprocess.CalledProcessError):runner.main()
            launch.assert_not_called()
            command = run.call_args.args[0]
            self.assertEqual(command[1], str(root/'bundled/logger.py'))
            self.assertIn('--check-config', command);self.assertIn('--metadata-helper', command)

    def test_finished_simulator_result_retained_if_logger_later_fails(self):
        with TemporaryDirectory() as d:
            root = Path(d);argv = self.fixture(root)
            server = SimpleNamespace(pid=10, poll=lambda: 0)
            simulator = SimpleNamespace(pid=11, wait=lambda: 0, poll=lambda: 0)
            logger = SimpleNamespace(wait=lambda **kwargs: 1, returncode=1, poll=lambda: 1)
            def launch(command, **kwargs):
                if command[0] == 'sim-python':
                    kwargs['stdout'].write('Final batch success rate: 2/100 = 2.0%\n');kwargs['stdout'].flush()
                    return simulator
                return logger if 'logger.py' in command[1] else server
            with patch('sys.argv', argv), patch.object(runner.socket, 'socket') as socket, patch.object(runner.subprocess, 'run'),\
                 patch.object(runner.subprocess, 'Popen', side_effect=launch), contextlib.redirect_stdout(io.StringIO()):
                socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 1234)
                with self.assertRaisesRegex(RuntimeError, 'simulator result retained'):runner.main()
            outputs = list((root/'eval').glob('*/simulator_completed.json'))
            self.assertEqual(len(outputs), 1)
            self.assertEqual(json.loads(outputs[0].read_text())['success_rate'], .02)
            self.assertFalse((outputs[0].parent/'completed.json').exists())


if __name__ == '__main__':unittest.main()
