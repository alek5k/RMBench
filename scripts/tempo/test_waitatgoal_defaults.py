"""Default mild-decay schedule and matched fresh 12k train/eval recipes."""
from pathlib import Path
import unittest
import yaml
from queue_training_after_run import validate_match, validate_gla_reference, validate_defaults
from queue_temporal_suite import validate_evaluation


class WaitAtGoalDefaultTests(unittest.TestCase):
    def setUp(self):
        self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
        self.defaults=self.load('waitatgoal_defaults.yaml')
        self.native=self.load('waitatgoal_smolvla_mild_decay_warmup_12k.yaml')
        self.tempo=self.load('waitatgoal_smolvla_gla_mild_decay_warmup_12k.yaml')

    def load(self,name):return yaml.safe_load((self.directory/name).read_text())

    def test_default_is_enforced_and_shared_parameters_match(self):
        for recipe in (self.native,self.tempo):
            t=recipe['training'];validate_defaults(t,self.defaults)
            self.assertEqual((t['steps'],t['save_interval']),(12000,12000))
            self.assertFalse(t['wandb_log_checkpoints'])
            for key,value in [('scheduler','legacy'),('warmup_steps',0),('minimum_lr',2.5e-6)]:
                with self.assertRaises(ValueError):validate_defaults({**t,key:value},self.defaults)
        changes=validate_match(self.native['training'],self.tempo['training'])
        self.assertFalse(set(self.defaults['training']) & set(changes))
        self.assertNotIn('memory',self.native)

    def test_duration_changes_are_explicit_and_do_not_waive_other_settings(self):
        previous=self.load('waitatgoal_smolvla_mild_decay_warmup_10k.yaml')['training']
        target=self.native['training'];duration={'steps':12000,'save_interval':12000}
        with self.assertRaises(ValueError):validate_match(previous,target)
        validate_match(previous,target,duration_changes=duration)
        with self.assertRaises(ValueError):validate_match(previous,{**target,'batch_size':16},duration_changes=duration)
        with self.assertRaises(ValueError):validate_match(previous,target,duration_changes={'batch_size':8})
        with self.assertRaises(ValueError):validate_match(previous,target,duration_changes={'steps':10000})
        validate_gla_reference(self.load('waitatgoal_smolvla_gla_mild_decay_warmup_10k.yaml'),self.tempo,
                               duration_changes=duration)

    def test_eval_checkpoint_and_order_match_12k(self):
        for name,recipe in [('smolvla',self.native),('gla',self.tempo)]:
            evaluation=self.load(f'eval_waitatgoal_{name}_mild_decay_warmup_12k_100ep.yaml')
            validate_evaluation(recipe['training'],evaluation)
            self.assertTrue(evaluation['launcher']['checkpoint'].endswith('/012000'))
            self.assertEqual(evaluation['wandb']['group'],recipe['training']['wandb_group'])
        queue=self.load('queue_waitatgoal_smolvla_eval_then_gla_mild_decay_warmup_12k.yaml')
        self.assertEqual(queue['predecessor_training_run_id'],self.native['training']['wandb_run_id'])
        self.assertTrue(queue['predecessor_eval_state'].endswith('smolvla-mild-decay-warmup-12k-eval-queue.json'))
        self.assertEqual(queue['defaults_config'],'waitatgoal_defaults.yaml')

    def test_pinned_native_scheduler_identical_and_has_400_update_warmup(self):
        import torch
        from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
        def rates(training):
            optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))],lr=training['learning_rate'])
            scheduler=CosineDecayWithWarmupSchedulerConfig(
                num_warmup_steps=training['warmup_steps'],num_decay_steps=training['scheduler_decay_steps'],
                peak_lr=training['learning_rate'],decay_lr=training['minimum_lr'],
            ).build(optimizer,num_training_steps=training['steps'])
            return [training['learning_rate']*scheduler.lr_lambdas[0](step) for step in range(12001)]
        native,tempo=rates(self.native['training']),rates(self.tempo['training'])
        self.assertEqual(native,tempo)
        self.assertAlmostEqual(native[0],1e-4/401)
        self.assertTrue(all(a<b for a,b in zip(native[:399],native[1:400])))
        self.assertTrue(all(a>=b for a,b in zip(native[400:],native[401:])))
        self.assertAlmostEqual(native[6000],8.75e-5)
        self.assertAlmostEqual(native[-1],7.5e-5)


if __name__=='__main__':unittest.main()
