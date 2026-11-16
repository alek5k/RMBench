"""Parallel episode completion must not drop W&B records."""
import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock,patch

spec=importlib.util.spec_from_file_location('eval_logging',Path(__file__).parents[1]/'lte/log_eval_wandb.py')
logger=importlib.util.module_from_spec(spec);spec.loader.exec_module(logger)


class ParallelLoggingTests(unittest.TestCase):
    def test_resume_id_skips_already_logged_steps_but_finishes_summary(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'eval.log'
            path.write_text('episode=3 success=True batch success: 1/1\n'
                            'episode=1 success=False batch success: 1/2\n')
            args=SimpleNamespace(log_file=path,mode='smolvla',task='cover_blocks',env_cfg='aloha_agilex',
                                 checkpoint='/missing/100000',seed=0,run_name='test',group='test',
                                 training_run_id='training',once=True,check_config=False,tag=[],wandb_run_id='eval-id')
            run=MagicMock();run.summary={};run.step=2
            with patch.object(logger,'parse_args',return_value=args),patch.object(logger.wandb,'init',return_value=run) as init,\
                 patch.dict(os.environ,{'WANDB_API_KEY':'test','WANDB_ENTITY':'test','WANDB_PROJECT':'test'}):
                logger.main()
            self.assertEqual(init.call_args.kwargs['id'],'eval-id')
            self.assertEqual(init.call_args.kwargs['resume'],'must')
            self.assertEqual([c.kwargs['step'] for c in run.log.call_args_list],[2])
            self.assertEqual(run.summary['eval/final_episodes'],2)

    def setUp(self):
        bundled = Path(__file__).with_name('tempo_wandb.py')
        self.helper = bundled if bundled.exists() else Path(__file__).resolve().parents[3]/'XPolicyLab/utils/tempo_wandb.py'

    def test_cli_accepts_bundled_helper_preflight_and_backfill(self):
        argv = ['logger', '--log-file', '/tmp/eval.log', '--mode', 'tempo', '--task', 'battery_try',
                '--env-cfg', 'aloha_agilex', '--checkpoint', '/tmp/100000', '--run-name', 'eval',
                '--policy-name', 'TEMPO_SmolVLA', '--metadata-helper', '/tmp/tempo_wandb.py', '--check-config', '--once']
        with patch('sys.argv', argv):args = logger.parse_args()
        self.assertTrue(args.check_config);self.assertTrue(args.once)
        self.assertEqual(args.metadata_helper, Path('/tmp/tempo_wandb.py'))

    def test_preflight_does_not_create_a_wandb_run(self):
        args=SimpleNamespace(checkpoint='/missing/100000',mode='smolvla',task='battery_try',env_cfg='aloha_agilex',
                             run_name='eval',group='group',seed=0,training_run_id='train',check_config=True,tag=[])
        with patch.object(logger,'parse_args',return_value=args),patch.object(logger.wandb,'init') as init,\
             patch.dict(os.environ,{'WANDB_API_KEY':'test','WANDB_ENTITY':'test','WANDB_PROJECT':'test'}),\
             contextlib.redirect_stdout(io.StringIO()) as output:
            logger.main()
        init.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['config']['checkpoint_step'],100000)

    def test_out_of_order_episode_ids_use_monotonic_completion_steps(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/'eval.log'
            path.write_text('episode=3 success=True batch success: 1/1\n'
                            'episode=1 success=False batch success: 1/2\n')
            args=SimpleNamespace(log_file=path,mode='tempo',task='battery_try',env_cfg='aloha_agilex',
                                 checkpoint='/checkpoints/run/100000',seed=0,run_name='test',group='test',
                                 training_run_id='test-run',screen_session='test-screen',process_id=None,poll_seconds=0,
                                 metadata_helper=self.helper,once=True,checkpoint_in_name=True)
            run=MagicMock();run.summary={}
            with patch.object(logger,'parse_args',return_value=args),patch.object(logger.wandb,'init',return_value=run) as init,\
                 patch.object(logger,'screen_is_running',side_effect=AssertionError('Backfill must not wait for a process')) as screen,\
                 patch.dict(os.environ,{'WANDB_API_KEY':'test','WANDB_ENTITY':'test','WANDB_PROJECT':'test'}):
                logger.main()
            screen.assert_not_called()
            self.assertEqual([c.kwargs['step'] for c in run.log.call_args_list],[1,2])
            self.assertEqual([c.args[0]['eval/episode'] for c in run.log.call_args_list],[3,1])
            self.assertEqual(run.summary['eval/final_success_rate'],.5)
            self.assertEqual(run.summary['eval/checkpoint_step'],100000)
            self.assertEqual(init.call_args.kwargs['name'],'test_checkpoint_100000_eval')

    def test_tempo_tags_from_checkpoint_and_user_tags(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            log=root/'eval.log';log.write_text('episode=0 success=True batch success: 1/1\n')
            (root/'tempo_run.json').write_text(json.dumps({'training':{
                'memory_backend':'gla','run_name':'battery_try_tempo_smolvla_gla_100k'}}))
            (root/'config.json').write_text(json.dumps({'type':'tempo_smolvla','train_expert_only':True}))
            args=SimpleNamespace(log_file=log,mode='tempo',task='battery_try',env_cfg='aloha_agilex',
                                 checkpoint=str(root),seed=0,run_name='test',group='test',policy_name='TEMPO_SmolVLA',
                                 training_run_id='train',screen_session='test-screen',process_id=None,
                                 poll_seconds=0,tag=['custom'],metadata_helper=self.helper)
            run=MagicMock();run.summary={}
            with patch.object(logger,'parse_args',return_value=args),patch.object(logger.wandb,'init',return_value=run) as init,\
                 patch.object(logger,'screen_is_running',return_value=False),\
                 patch.dict(os.environ,{'WANDB_API_KEY':'test','WANDB_ENTITY':'test','WANDB_PROJECT':'test',
                                        'WANDB_TAGS':'from-env,custom'}):
                logger.main()
            tags=init.call_args.kwargs['tags']
            self.assertTrue({'gla','TEMPO_SmolVLA','battery_try','action-expert-only','from-env','custom'}<=set(tags))
            self.assertEqual(tags.count('custom'),1)
            self.assertEqual(init.call_args.kwargs['config']['policy'],'TEMPO_SmolVLA')
            self.assertEqual(init.call_args.kwargs['name'],'battery_try_tempo_100k_eval')

    def test_native_control_checkpoint_gets_matching_tags_without_tempo(self):
        with TemporaryDirectory() as directory:
            root=Path(directory)
            log=root/'eval.log';log.write_text('episode=0 success=True batch success: 1/1\n')
            (root/'tempo_run.json').write_text(json.dumps({'training':{'policy_variant':'smolvla'}}))
            (root/'config.json').write_text(json.dumps({'type':'smolvla','train_expert_only':True}))
            args=SimpleNamespace(log_file=log,mode='smolvla',task='battery_try',env_cfg='aloha_agilex',
                                 checkpoint=str(root),seed=0,run_name='test',group='test',
                                 training_run_id='train',screen_session='test-screen',process_id=None,poll_seconds=0,
                                 metadata_helper=self.helper)
            run=MagicMock();run.summary={}
            with patch.object(logger,'parse_args',return_value=args),patch.object(logger.wandb,'init',return_value=run) as init,\
                 patch.object(logger,'screen_is_running',return_value=False),\
                 patch.dict(os.environ,{'WANDB_API_KEY':'test','WANDB_ENTITY':'test','WANDB_PROJECT':'test'}):
                logger.main()
            tags=init.call_args.kwargs['tags']
            self.assertTrue({'SmolVLA','smolvla','frozen-backbone','action-expert-only'}<=set(tags))
            self.assertFalse({'tempo','rmt','gla'}&set(tags))


if __name__=='__main__':unittest.main()
