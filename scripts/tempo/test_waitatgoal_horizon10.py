"""Ten-action-horizon study retains mild schedule and full-data seed-repeat settings."""
from pathlib import Path
import unittest
import yaml
from queue_training_after_run import validate_defaults
from queue_temporal_suite import validate_evaluation


class HorizonStudyTests(unittest.TestCase):
    def test_seed1_repeat_changes_only_seed_and_identity(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        load=lambda name: yaml.safe_load((root/name).read_text())
        a=load('waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2.yaml')
        b=load('waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed1.yaml')
        self.assertEqual(a['memory'],b['memory'])
        changed={k for k in a['training'].keys()|b['training'].keys()
                 if a['training'].get(k)!=b['training'].get(k)}
        self.assertEqual(changed,{'seed','reference_wandb_run','output_dir','wandb_run_id','wandb_group','run_name'})
        t=b['training']
        self.assertEqual((t['seed'],t['steps'],t['chunk_size'],t['n_action_steps']),(1,20000,10,10))
        self.assertEqual(t['reference_wandb_run'],'uts_robot_lab/RMBench/1dd44cee')
        validate_defaults(t,load('waitatgoal_defaults.yaml'))
        ev=load('eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed1_100ep.yaml')
        validate_evaluation(t,ev)
        self.assertEqual(ev['evaluation'],load('eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2_100ep.yaml')['evaluation'])
        self.assertEqual(ev['wandb']['group'],t['wandb_group'])
        self.assertEqual(ev['wandb']['run_name'],t['run_name']+'_eval')
        q=load('queue_eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed1.yaml')
        self.assertEqual(q['training_run_id'],t['wandb_run_id'])
        self.assertEqual(q['predecessor_state'],'/data/RMBench-state/gla-training-'+t['wandb_run_id']+'.json')

    def test_only_seed_horizons_and_identity_change(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        load=lambda name: yaml.safe_load((root/name).read_text())
        a=load('waitatgoal_tempo_mild_lr_decay_20k_seed1.yaml')
        b=load('waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2.yaml')
        self.assertEqual(a['memory'],b['memory'])
        changed={k for k in a['training'].keys()|b['training'].keys()
                 if a['training'].get(k)!=b['training'].get(k)}
        self.assertEqual(changed,{'seed','chunk_size','n_action_steps','reference_wandb_run',
                                 'output_dir','wandb_run_id','wandb_group','run_name'})
        t=b['training']
        self.assertEqual((t['seed'],t['steps'],t['batch_size'],t['save_interval']), (2,20000,8,10000))
        self.assertEqual((t['chunk_size'],t['n_action_steps']),(10,10))
        validate_defaults(t,load('waitatgoal_defaults.yaml'))
        ev=load('eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2_100ep.yaml')
        validate_evaluation(t,ev)
        self.assertEqual(ev['evaluation'],load('eval_waitatgoal_tempo_mild_lr_decay_20k_seed1_100ep.yaml')['evaluation'])
        self.assertEqual(ev['wandb']['group'],t['wandb_group'])
        self.assertEqual(ev['wandb']['run_name'],t['run_name']+'_eval')
        q=load('queue_eval_waitatgoal_tempo_mild_lr_decay_horizon10_20k_seed2.yaml')
        self.assertEqual(q['training_run_id'],t['wandb_run_id'])
        self.assertEqual(q['predecessor_state'],'/data/RMBench-state/gla-training-'+t['wandb_run_id']+'.json')


if __name__=='__main__':unittest.main()
