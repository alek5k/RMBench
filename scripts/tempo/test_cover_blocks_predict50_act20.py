"""Cover Blocks smoke must gate fresh production; no eval or lifecycle actions."""
import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import yaml
import run_cover_blocks_predict50_act20 as runner


class CoverBlocksRunTests(unittest.TestCase):
    def test_smoke_gates_production_and_all_checkpoints_are_verified(self):
        configs = Path(__file__).resolve().parents[2]/'config/tempo'
        for smoke_pass in (True, False):
            with self.subTest(smoke_pass=smoke_pass), TemporaryDirectory() as directory:
                root = Path(directory)
                reference = yaml.safe_load((configs/'battery_try_tempo_predict50_act20_100k_seed0.yaml').read_text())
                target = yaml.safe_load((configs/'cover_blocks_tempo_predict50_act20_100k_seed0.yaml').read_text())
                reference['training']['output_dir'] = str(root/'battery')
                target['training']['output_dir'] = str(root/'cover')
                (root/'battery/100000').mkdir(parents=True)
                resolved = {'training': reference['training'], 'policy': {'memory': reference['memory']}}
                (root/'reference.json').write_text(json.dumps(resolved))
                (root/'training.yaml').write_text(yaml.safe_dump(target))
                (root/'receipt.json').write_text('{}')
                (root/'SOURCE_BUNDLE.json').write_text('{"bundle_id":"test-source"}')
                argv = ['runner', '--root', str(root), '--config', str(root/'training.yaml'),
                        '--reference', str(root/'reference.json'), '--receipt', str(root/'receipt.json'),
                        '--state', str(root/'state.json')]
                with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=100*1024**3)), \
                     patch.object(runner, 'validate_dataset', return_value={'total_frames': 51077}), \
                     patch.object(runner, 'verify_checkpoint') as checkpoints, \
                     patch.object(runner, 'check_execution_slice', return_value={'executed_actions': 20}), \
                     patch.object(runner, 'check_smoke', return_value={'parameter_counts': {}},
                                  side_effect=None if smoke_pass else ValueError('Wrong normalization')), \
                     patch.object(runner.subprocess, 'check_output', return_value=''), \
                     patch.object(runner.subprocess, 'run') as execute:
                    if smoke_pass:runner.main()
                    else:
                        with self.assertRaisesRegex(ValueError, 'normalization'):runner.main()
                self.assertEqual(execute.call_count, 3 if smoke_pass else 2)
                self.assertIn('--smoke', execute.call_args_list[1].args[0])
                if smoke_pass:self.assertNotIn('--smoke', execute.call_args_list[2].args[0])
                self.assertEqual(checkpoints.call_count, 11 if smoke_pass else 1)
                state = json.loads((root/'state.json').read_text())
                self.assertEqual(state['stage'], 'complete' if smoke_pass else 'failed')
                self.assertFalse(state['evaluation_scheduled'])
                self.assertEqual(state['instance_action'], 'none')


if __name__ == '__main__':unittest.main()
