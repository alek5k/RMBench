import copy
import json
from pathlib import Path
import signal
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml

import run_local_reference_sanity_eval as runner


class ReferenceSanityEvalTests(unittest.TestCase):
    def test_resume_supervisor_even_if_reference_fails(self):
        original = {'pid': 123, 'start': '456', 'state': 'S',
                    'command': ['python', '/scripts/run_local_checkpoint_evals.py', '--config', '/sweep.yaml']}
        stopped = dict(original, state='T')
        with patch.object(runner, 'same_process', return_value=True), \
                patch.object(runner, 'process_info', return_value=stopped), patch.object(runner.os, 'kill') as kill:
            with self.assertRaisesRegex(RuntimeError, 'reference failed'):
                with runner.paused_supervisor(original, '/sweep.yaml'):
                    raise RuntimeError('reference failed')
        self.assertEqual(kill.call_args_list[0].args, (123, signal.SIGSTOP))
        self.assertEqual(kill.call_args_list[1].args, (123, signal.SIGCONT))

    def test_unrelated_process_never_signalled(self):
        original = {'pid': 123, 'start': '456', 'command': ['python', '/other/training.py']}
        with patch.object(runner.os, 'kill') as kill:
            with self.assertRaises(ValueError):
                with runner.paused_supervisor(original, '/sweep.yaml'):pass
        kill.assert_not_called()

    def test_hash_mismatch_fails(self):
        with TemporaryDirectory() as d:
            source = Path(d);(source/'model').write_bytes(b'not the checkpoint')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                runner.verify_checkpoint(source, {'model': '0'*64})

    def test_same_local_evaluation_except_reference_identity(self):
        config = Path(__file__).parents[2]/'config/tempo'
        reference = yaml.safe_load((config/'eval_battery_try_tempo_reference_100k_local_sanity.yaml').read_text())
        sweep = yaml.safe_load((config/'eval_battery_try_tempo_horizon10_local_all_checkpoints.yaml').read_text())
        self.assertEqual(reference['evaluation'], sweep['evaluation'])
        self.assertEqual(reference['launcher'], sweep['launcher'])
        for key in ('source_root', 'vlm_assets', 'smovla_root', 'cuda_visible_devices'):
            self.assertEqual(reference[key], sweep[key])
        self.assertEqual(reference['expected_horizon'], 50)
        self.assertEqual(reference['training_run_id'], 'e84cae63')
        for digest in reference['reference_source']['sha256'].values():
            self.assertEqual(len(digest), 64)
        before = copy.deepcopy(reference)
        settings = runner.evaluation_recipe(reference, Path('/portable/100000'))
        self.assertEqual(reference, before)
        self.assertEqual(settings['wandb']['training_run_id'], 'e84cae63')
        self.assertEqual(settings['launcher']['checkpoint'], '/portable/100000')
        self.assertEqual(settings['evaluation']['episodes'], 100)


if __name__ == '__main__':unittest.main()
