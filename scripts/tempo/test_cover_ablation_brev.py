from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import os
import subprocess
import yaml
import run_cover_ablation_brev as runner
from defer_cover_ablation_brev import ready
from stop_l40s_after_queues import validate_recipe


class CoverAblationTests(unittest.TestCase):
    def test_setup_uses_configured_policy_environment_on_a_new_username(self):
        script=Path(__file__).with_name('setup_smolvla_training.sh').read_text()
        assignment=next(line for line in script.splitlines() if line.startswith('PYTHON='))
        actual=subprocess.check_output(['bash','-c',assignment+'; printf %s "$PYTHON"'],
                                       env={**os.environ,'TEMPO_SMOLVLA_ENV':'/task/pinned-policy'},text=True)
        self.assertEqual(actual,'/task/pinned-policy/bin/python')

    def test_deferred_launch_requires_exact_healthy_existing_instance(self):
        row={'name':'calm-crimson-urial','id':'zcb67bfcs','gpu':'L40S',
             'status':'FAILURE','shell_status':'NOT READY','health_status':'UNHEALTHY'}
        self.assertFalse(ready(row))
        row.update(status='RUNNING',shell_status='READY',health_status='HEALTHY')
        self.assertTrue(ready(row))
        with self.assertRaises(ValueError):ready({**row,'id':'replacement-not-authorized'})

    def test_legacy_receipt_length_is_derived_from_real_episode_metadata(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with TemporaryDirectory() as directory:
            root=Path(directory);p=root/'meta/episodes/chunk-000/file-000.parquet';p.parent.mkdir(parents=True)
            pq.write_table(pa.table({'length':[6,9]}),p)
            self.assertEqual(runner.maximum_episode_frames(root,{'total_episodes':2,'total_frames':15}),9)
            with self.assertRaises(ValueError):runner.maximum_episode_frames(root,{'total_episodes':2,'total_frames':16})

    def setUp(self):self.directory=Path(__file__).resolve().parents[2]/'config/tempo'
    def load(self,name):return yaml.safe_load((self.directory/name).read_text())
    def study(self,variant):
        names={'native64':'native64_100k','balanced200k':'balanced_200k','reference1m':'reference_1m'}
        queue=self.load('queue_cover_blocks_'+names[variant]+'_brev.yaml')
        return self.load(queue['training_config']),self.load(queue['reference_config']),self.load(queue['eval_config'])
    def test_each_study_only_changes_its_declared_ablation(self):
        for variant in ('native64','balanced200k','reference1m'):
            recipe,reference,evaluation=self.study(variant)
            runner.validate(recipe,reference,evaluation,variant)
            self.assertEqual(recipe['memory']['history_queries'],16)
            self.assertEqual(recipe['memory']['num_tokens'],4)
            self.assertFalse(recipe['training']['wandb_log_checkpoints'])
    def test_unrelated_lr_horizon_or_loss_changes_rejected(self):
        for variant in ('native64','balanced200k','reference1m'):
            recipe,reference,evaluation=self.study(variant)
            for section,key,value in [('training','learning_rate',1e-3),('training','n_action_steps',10),
                                      ('training','temporal_objective','reconstruction_only'),('memory','history_queries',128)]:
                changed=deepcopy(recipe);changed[section][key]=value
                with self.assertRaises(ValueError):runner.validate(changed,reference,evaluation,variant)
        recipe,reference,evaluation=self.study('native64')
        recipe['training']['history_loss_balance']={'window':100}
        with self.assertRaises(ValueError):runner.validate(recipe,reference,evaluation,'native64')
        recipe,reference,evaluation=self.study('reference1m')
        recipe['training']['history_loss_balance']={'window':100}
        with self.assertRaises(ValueError):runner.validate(recipe,reference,evaluation,'reference1m')
    def test_wrong_final_eval_or_keepalive_settings_rejected(self):
        for variant in ('native64','balanced200k','reference1m'):
            recipe,reference,evaluation=self.study(variant)
            changed=deepcopy(evaluation);changed['launcher']['checkpoint']='/wrong/100000'
            with self.assertRaises(ValueError):runner.validate(recipe,reference,changed,variant)
            changed=deepcopy(evaluation);changed['launcher']['websocket']['ws_ping_interval_s']=20
            with self.assertRaises(ValueError):runner.validate(recipe,reference,changed,variant)
    def test_shutdown_tracks_only_these_two_new_final_evals(self):
        recipe=self.load('stop_cover_blocks_native64_balanced_261113.yaml');validate_recipe(recipe)
        self.assertEqual(recipe['barrier'],'per_instance_queues_complete')
        runs=[r for host in recipe['instances'] for q in host['queues'] for r in q['expected_runs']]
        self.assertEqual([(r['id'],r['step']) for r in runs],[('718d9127',100000),('db47b024',200000)])
        replacement=self.load('stop_cover_blocks_native64_balanced_sound_261113.yaml');validate_recipe(replacement)
        self.assertEqual([i['instance'] for i in replacement['instances']],['normal-scarlet-vole','sound-coral-echidna'])


if __name__=='__main__':unittest.main()
