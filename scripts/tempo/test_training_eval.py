"""Smoke evaluation must pass before production training is started."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml

spec=importlib.util.spec_from_file_location('training_eval',Path(__file__).with_name('run_training_eval.py'))
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class TrainingEvalTests(unittest.TestCase):
    def run_case(self, root, *, fail_smoke_eval=False, local_eval=False):
        output=root/'checkpoints'
        (root/'train.yaml').write_text(yaml.safe_dump({'training':{'output_dir':str(output),
              'steps':100000,'batch_size':8,'wandb_run_id':'train-id'}}))
        (root/'eval.yaml').write_text(yaml.safe_dump({'launcher':{'checkpoint':str((root/'local')/'100000') if local_eval else str(output/'100000'),
                                                    'execution':'local' if local_eval else 'remote'},
                                                    'wandb':{'training_run_id':'train-id'}}))
        argv=['runner','--root',str(root),'--training-config',str(root/'train.yaml'),
              '--eval-config',str(root/'eval.yaml'),'--state',str(root/'status.json'),
              '--smoke-eval-episodes','2']
        def fake_run(command, **kwargs):
            if '--no-wandb' in command and fail_smoke_eval:
                raise subprocess.CalledProcessError(1,command)
            if '-m' in command and '--smoke' not in command:
                (output/'100000').mkdir(parents=True)
                (output/'100000/tempo_run.json').write_text(json.dumps({'step':100000}))
        with patch('sys.argv',argv),patch.object(runner.subprocess,'run',side_effect=fake_run) as run,\
             contextlib.redirect_stdout(io.StringIO()):
            if fail_smoke_eval:
                with self.assertRaises(subprocess.CalledProcessError):runner.main()
                self.assertEqual(len(run.call_args_list),3)
                self.assertEqual(json.loads((root/'status.json').read_text())['stage'],'failed')
            else:
                runner.main()
                commands=[call.args[0] for call in run.call_args_list]
                if local_eval:
                    self.assertEqual(len(commands),2)
                    self.assertIn('--smoke',commands[0])
                    self.assertNotIn('--smoke',commands[1])
                    state=json.loads((root/'status.json').read_text())
                    self.assertEqual(state['stage'],'awaiting_local_download')
                    self.assertEqual(state['instance_action'],'none')
                    return
                self.assertEqual(len(commands),6)
                self.assertIn('--smoke',commands[0])
                self.assertIn('--no-wandb',commands[2])
                self.assertIn(str(output)+'_smoke/000003',commands[2])
                self.assertNotIn('--smoke',commands[3])
                self.assertNotIn('--no-wandb',commands[5])
                self.assertEqual(json.loads((root/'status.json').read_text())['stage'],'complete')

    def test_smoke_then_production_then_final_eval(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory))

    def test_failed_smoke_eval_prevents_production(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory),fail_smoke_eval=True)

    def test_local_eval_never_provisions_or_runs_remote_simulator(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory),local_eval=True)


if __name__=='__main__':unittest.main()
