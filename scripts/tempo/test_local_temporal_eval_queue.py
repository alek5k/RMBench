import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import yaml
from queue_local_temporal_evals import load_plan, predecessor_ready, training_ready


class LocalTemporalEvalQueueTests(unittest.TestCase):
    def test_plan_uses_same_protocol_and_only_eval_gpu(self):
        directory = Path(__file__).resolve().parents[2] / 'config/tempo'
        queue = yaml.safe_load((directory / 'queue_liftqa_smolvla_horizon10_local_evals.yaml').read_text())
        plan = load_plan(queue, directory)
        self.assertEqual([p['seed'] for p in plan], [1, 2])
        for p in plan:
            self.assertEqual(p['evaluation']['launcher']['policy_kind'], 'smolvla')
            self.assertEqual(p['evaluation']['evaluation']['episodes'], 100)
            self.assertEqual(p['evaluation']['wandb']['training_run_id'], p['run_id'])

    def test_seed1_can_eval_while_seed2_trains_but_seed2_must_wait(self):
        state = {'stage': 'training', 'plan': [{'seed': 1, 'run_id': 'one'}, {'seed': 2, 'run_id': 'two'}],
                 'completed': [{'seed': 1, 'run_id': 'one'}]}
        self.assertTrue(training_ready(state, 1, 'one'))
        self.assertFalse(training_ready(state, 2, 'two'))
        with self.assertRaises(ValueError):
            training_ready(state, 2, 'wrong')
        state['stage'] = 'failed'
        with self.assertRaises(RuntimeError):
            training_ready(state, 2, 'two')

    def test_predecessor_requires_exact_completed_full_eval(self):
        with TemporaryDirectory() as d:
            path = Path(d) / 'results.json'
            result = {'training_run_id': 'tempo', 'checkpoint_step': 20000, 'episodes': 100, 'seed': 0}
            path.write_text(json.dumps(result))
            state = {'training_run_id': 'tempo', 'checkpoint_step': 20000, 'stage': 'evaluating', 'result': str(path)}
            self.assertFalse(predecessor_ready(state, 'tempo'))
            state['stage'] = 'complete'
            self.assertTrue(predecessor_ready(state, 'tempo'))
            path.write_text(json.dumps({**result, 'episodes': 99}))
            with self.assertRaises(ValueError):
                predecessor_ready(state, 'tempo')


if __name__ == '__main__':
    unittest.main()
