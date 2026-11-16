"""Queue ordering and fail-closed checks, without cloud/GPU processes."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml

spec=importlib.util.spec_from_file_location('followup_queue',Path(__file__).with_name('queue_gla_eval_baseline.py'))
queue=importlib.util.module_from_spec(spec);spec.loader.exec_module(queue)


class QueueTests(unittest.TestCase):
    def test_matching_rejects_changed_learning_rate(self):
        with self.assertRaises(ValueError):
            queue.validate_match({'learning_rate':1.,'steps':100000},
                                 {'learning_rate':2.,'steps':100000,'policy_variant':'smolvla'})

    def run_case(self, root, complete):
        reference={'output_dir':str(root/'gla'),'steps':100000,'learning_rate':.000025}
        baseline={**reference,'output_dir':str(root/'baseline'),'policy_variant':'smolvla'}
        checkpoint=root/'gla/100000'
        checkpoint.mkdir(parents=True)
        if complete:
            (checkpoint/'tempo_run.json').write_text(json.dumps({'step':100000}))
            for name in ('model.safetensors','config.json','optimizer.pt'):
                (checkpoint/name).touch()
        for name,values in [('reference',{'training':reference}),('baseline',{'training':baseline}),
                            ('eval',{'launcher':{'checkpoint':str(checkpoint)}})]:
            (root/f'{name}.yaml').write_text(yaml.safe_dump(values))
        argv=['queue','--root',str(root),'--training-pid','0',
              '--reference-config',str(root/'reference.yaml'),'--baseline-config',str(root/'baseline.yaml'),
              '--eval-config',str(root/'eval.yaml'),'--state',str(root/'state.json')]
        with patch('sys.argv',argv),patch.object(queue.subprocess,'run') as run,contextlib.redirect_stdout(io.StringIO()):
            if complete:
                queue.main()
                commands=[c.args[0] for c in run.call_args_list]
                self.assertTrue(commands[0][1].endswith('setup_smolvla_eval.sh'))
                self.assertTrue(commands[1][1].endswith('evaluate_smolvla.py'))
                self.assertIn('--smoke',commands[2])
                self.assertNotIn('--smoke',commands[3])
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'complete')
            else:
                with self.assertRaises(RuntimeError): queue.main()
                run.assert_not_called()
                self.assertEqual(json.loads((root/'state.json').read_text())['stage'],'failed')

    def test_eval_then_smoke_then_baseline(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory),True)

    def test_incomplete_training_prevents_evaluation_and_baseline(self):
        with TemporaryDirectory() as directory:self.run_case(Path(directory),False)


if __name__=='__main__':unittest.main()
