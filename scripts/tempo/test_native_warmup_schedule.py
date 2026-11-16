"""Matched native warmup/decay for the train/eval/train/eval comparison."""
from pathlib import Path
import unittest
import yaml
from queue_training_after_run import validate_match, validate_gla_reference
from queue_temporal_suite import validate_evaluation


class NativeWarmupScheduleTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.native=self.load('waitatgoal_smolvla_mild_decay_warmup_10k.yaml')
        self.tempo=self.load('waitatgoal_smolvla_gla_mild_decay_warmup_10k.yaml')

    def load(self,name):return yaml.safe_load((self.directory/name).read_text())

    def test_shared_parameters_and_native_schedule_match(self):
        changes=validate_match(self.native['training'],self.tempo['training'])
        self.assertFalse({'learning_rate','minimum_lr','warmup_steps','scheduler','scheduler_decay_steps'} & set(changes))
        for recipe in (self.native,self.tempo):
            t=recipe['training']
            self.assertEqual((t['scheduler'],t['warmup_steps'],t['scheduler_decay_steps'],t['steps']),
                             ('smolvla',1000,30000,10000))
            self.assertEqual((t['learning_rate'],t['minimum_lr']),(1e-4,7.5e-5))
            self.assertFalse(t['wandb_log_checkpoints'])
        self.assertNotIn('memory',self.native)
        validate_gla_reference(self.load('waitatgoal_smolvla_gla_reference_10k.yaml'),self.tempo,
                               {'minimum_lr':7.5e-5})

    def test_both_evals_and_completion_order(self):
        names=[('smolvla',self.native,'91f4a6b2'),('gla',self.tempo,'c74d109e')]
        for name,recipe,rid in names:
            evaluation=self.load(f'eval_waitatgoal_{name}_mild_decay_warmup_10k_100ep.yaml')
            validate_evaluation(recipe['training'],evaluation)
            self.assertEqual(evaluation['wandb']['group'],recipe['training']['wandb_group'])
            self.assertEqual(evaluation['wandb']['training_run_id'],rid)
            self.assertEqual(evaluation['evaluation']['seed'],0)
            self.assertEqual(evaluation['evaluation']['max_steps'],400)
        queue=self.load('queue_waitatgoal_smolvla_eval_then_gla_mild_decay_warmup_10k.yaml')
        self.assertEqual(queue['predecessor_training_run_id'],'91f4a6b2')
        self.assertTrue(queue['require_native_warmup'])
        self.assertEqual(queue['predecessor_eval_state'],
                         '/data/RMBench-state/waitatgoal-smolvla-mild-decay-warmup-10k-eval-queue.json')

    def test_actual_native_schedules_identical_for_all_updates(self):
        import torch
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        def build(training,constant=False):
            optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=training['learning_rate'])
            return CosineDecayWithWarmupSchedulerConfig(
                num_warmup_steps=training['warmup_steps'],num_decay_steps=training['scheduler_decay_steps'],
                peak_lr=training['learning_rate'],
                decay_lr=training['learning_rate'] if constant else training['minimum_lr'],
            ).build(optimizer,num_training_steps=training['steps'])
        first,second=build(self.native['training']),build(self.tempo['training'])
        rates=[1e-4*first.lr_lambdas[0](s) for s in range(10001)]
        self.assertEqual(rates,[1e-4*second.lr_lambdas[0](s) for s in range(10001)])
        self.assertAlmostEqual(rates[0],1e-4/334)
        self.assertTrue(all(a<b for a,b in zip(rates[:332],rates[1:333])))
        self.assertAlmostEqual(rates[5000],8.75e-5)
        self.assertAlmostEqual(rates[10000],7.5e-5)
        self.assertTrue(all(a>=b for a,b in zip(rates[333:],rates[334:])))
        constant=build(self.native['training'],constant=True)
        self.assertAlmostEqual(1e-4*constant.lr_lambdas[0](0),1e-4/334)
        self.assertLess(1e-4*constant.lr_lambdas[0](100),1e-4)
        for s in (333,1000,5000,10000):
            self.assertAlmostEqual(1e-4*constant.lr_lambdas[0](s),1e-4)


if __name__=='__main__':unittest.main()
