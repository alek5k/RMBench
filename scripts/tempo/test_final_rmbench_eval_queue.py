"""Final eval must wait for successful training and reject identity/count drift."""
import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import yaml
import queue_final_rmbench_eval as queue
from validate_copied_eval_runtime import fingerprint


class FinalEvalTests(unittest.TestCase):
    def setUp(self):
        configs=Path(__file__).resolve().parents[2]/'config/tempo'
        self.training=yaml.safe_load((configs/'swap_T_tempo_predict50_act50_100k_seed0.yaml').read_text())['training']
        self.evaluation=yaml.safe_load((configs/'eval_swap_T_tempo_predict50_act50_100k_seed0.yaml').read_text())

    def test_final_config_alignment(self):
        queue.validate_evaluation(self.training,self.evaluation)
        for section,key,value in [('evaluation','task','cover_blocks'),('evaluation','episodes',10),
                                   ('evaluation','num_workers',1),('wandb','training_run_id','wrong'),
                                   ('launcher','checkpoint',self.training['output_dir']+'/090000'),
                                   ('wandb','enabled',False)]:
            changed=copy.deepcopy(self.evaluation);changed[section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):queue.validate_evaluation(self.training,changed)

    def test_cover_blocks_final_100k_uses_same_benchmark_as_quick_checks(self):
        configs=Path(__file__).resolve().parents[2]/'config/tempo'
        training=yaml.safe_load((configs/'cover_blocks_tempo_predict50_act20_100k_seed0.yaml').read_text())['training']
        evaluation=yaml.safe_load((configs/'eval_cover_blocks_tempo_predict50_act20_100k_seed0.yaml').read_text())
        quick=yaml.safe_load((configs/'eval_cover_blocks_tempo_predict50_act20_70k_quick.yaml').read_text())
        queue.validate_evaluation(training,evaluation)
        self.assertEqual(evaluation['evaluation'],{**quick['evaluation'],'episodes':100,'num_workers':4})
        self.assertEqual(evaluation['wandb']['training_run_id'],'9b4a762e')
        self.assertEqual(evaluation['wandb']['group'],quick['wandb']['group'])
        self.assertTrue(evaluation['launcher']['checkpoint'].endswith('/100000'))

    def test_queue_never_launches_until_training_completes(self):
        for success in (True,False):
            with self.subTest(success=success),TemporaryDirectory() as directory:
                root=Path(directory);training={**self.training,'output_dir':str(root/'training')}
                evaluation=copy.deepcopy(self.evaluation)
                evaluation['launcher']['checkpoint']=str(root/'training/100000')
                evaluation['launcher']['output_root']=str(root/'evaluations')
                checkpoint=root/'training/100000';checkpoint.mkdir(parents=True)
                (checkpoint/'tempo_run.json').write_text(json.dumps({'training':training}))
                (checkpoint/'config.json').write_text(json.dumps({'type':'tempo_smolvla','memory_backend':'gla',
                    'chunk_size':50,'n_action_steps':50}))
                (root/'training.yaml').write_text(yaml.safe_dump({'training':training}))
                (root/'eval.yaml').write_text(yaml.safe_dump(evaluation))
                source={'stage':'training','training_run_id':'b871c40f','source_root':str(root)}
                (root/'previous.json').write_text(json.dumps(source))
                (root/'ready.json').write_text(json.dumps({'ready':True,'source_root':str(root)}))
                recipe={'training_config':str(root/'training.yaml'),'training_run_id':'b871c40f',
                        'eval_config':'eval.yaml','predecessor_state':str(root/'previous.json'),
                        'runtime_receipt':str(root/'ready.json'),'poll_seconds':0}
                (root/'queue.yaml').write_text(yaml.safe_dump(recipe))
                def finish(_):
                    execute.assert_not_called();verify.assert_not_called();gpu.assert_not_called()
                    source['stage']='complete' if success else 'failed'
                    (root/'previous.json').write_text(json.dumps(source))
                def evaluate(*args,**kwargs):
                    result=root/'evaluations/result';result.mkdir(parents=True)
                    (result/'completed.json').write_text(json.dumps({'training_run_id':'b871c40f',
                        'checkpoint_step':100000,'checkpoint':str(checkpoint),'episodes':100,'successes':50}))
                argv=['queue','--root',str(root),'--config',str(root/'queue.yaml'),'--state',str(root/'state.json')]
                with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()), \
                     patch.object(queue.time,'sleep',side_effect=finish), \
                     patch.object(queue,'verify_checkpoint',return_value=checkpoint) as verify, \
                     patch.object(queue.subprocess,'check_output',return_value='') as gpu, \
                     patch.object(queue.subprocess,'run',side_effect=evaluate) as execute:
                    if success:queue.main()
                    else:
                        with self.assertRaises(RuntimeError):queue.main()
                state=json.loads((root/'state.json').read_text())
                self.assertEqual(state['stage'],'complete' if success else 'failed')
                self.assertEqual(execute.call_count,1 if success else 0)
                self.assertEqual(state['instance_action'],'none')

    def test_result_requires_exact_checkpoint_and_100_episodes(self):
        result={'training_run_id':'b871c40f','checkpoint_step':100000,'episodes':100,'successes':0,
                'checkpoint':self.training['output_dir']+'/100000'}
        queue.validate_result(result,self.training)
        for key,value in [('episodes',99),('checkpoint_step',90000),('training_run_id','wrong'),('successes',101)]:
            with self.subTest(key=key),self.assertRaises(ValueError):queue.validate_result({**result,key:value},self.training)

    def test_runtime_fingerprint_ignores_bytecode_only(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);(root/'binary').write_bytes(b'1234')
            (root/'link').symlink_to('binary');(root/'__pycache__').mkdir()
            (root/'__pycache__/test.pyc').write_bytes(b'ignored')
            self.assertEqual(fingerprint(root),{'files':1,'bytes':4,'symlinks':1})


if __name__=='__main__':unittest.main()
