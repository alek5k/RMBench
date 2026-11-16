"""Checkpoint completion/hash gates and per-frame temporal-memory updates."""
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
import yaml

sys.path.insert(0,str(Path(__file__).parent))
import export_checkpoint_manifest as exporter
from followup_local_eval import verify_tree
spec=importlib.util.spec_from_file_location('temporal_eval',Path(__file__).parents[1]/'lte/eval_lte_smolvla_temporal.py')
evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)


class LocalFollowupTests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source/RMBench'
        config=source/'config/tempo/train.yaml';config.parent.mkdir(parents=True)
        checkpoints=root/'checkpoints';checkpoints.mkdir()
        vlm=root/'revision';vlm.mkdir();(vlm/'config.json').write_text('{}')
        training={'wandb_run_id':'run','steps':20000,'save_interval':10000,
                  'output_dir':str(checkpoints),'vlm_revision':'revision'}
        config.write_text(yaml.safe_dump({'training':training}))
        (source/'SOURCE_BUNDLE.json').write_text(json.dumps({'bundle_id':'a'*20}))
        bootstrap=root/'bootstrap';bootstrap.mkdir()
        archive=bootstrap/f"rmbench-source-{'a'*20}.tar.zst";archive.write_bytes(b'archive')
        Path(str(archive)+'.sha256').write_text(exporter.digest(archive)+' archive\n')
        for step in (10000,20000):
            path=checkpoints/f'{step:06d}';path.mkdir()
            for name in ('model.safetensors','optimizer.pt','policy_preprocessor.json','policy_postprocessor.json'):
                (path/name).write_bytes(b'checkpoint')
            (path/'tempo_run.json').write_text(json.dumps({'step':step,'training':training}))
            (path/'config.json').write_text(json.dumps({'vlm_model_name':str(vlm)}))
        state=root/'state.json'
        state.write_text(json.dumps({'stage':'awaiting_local_download','training_run_id':'run',
                                     'training_config':str(config)}))
        return state,checkpoints,bootstrap

    def test_complete_inventory_and_checksums(self):
        with TemporaryDirectory() as directory:
            state,checkpoints,bootstrap=self.fixture(Path(directory))
            report=exporter.export(state,'run',bootstrap)
            self.assertEqual(report['steps'],[10000,20000])
            verify_tree(checkpoints,report['checkpoints']['files'])

    def test_missing_checkpoint_prevents_export(self):
        with TemporaryDirectory() as directory:
            state,checkpoints,bootstrap=self.fixture(Path(directory))
            (checkpoints/'010000/model.safetensors').unlink()
            with self.assertRaises(FileNotFoundError):exporter.export(state,'run',bootstrap)

    def test_running_job_prevents_export(self):
        with TemporaryDirectory() as directory:
            state,_,bootstrap=self.fixture(Path(directory))
            data=json.loads(state.read_text());data['stage']='training';state.write_text(json.dumps(data))
            with self.assertRaises(ValueError):exporter.export(state,'run',bootstrap)

    def test_corruption_and_path_escape_fail_verification(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);path=root/'model';path.write_bytes(b'original')
            entry={'path':'model','size':path.stat().st_size,'sha256':exporter.digest(path)}
            path.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'Checksum mismatch'):verify_tree(root,[entry])
            with self.assertRaisesRegex(ValueError,'Unsafe'):verify_tree(root,[{**entry,'path':'../outside'}])

    def test_memory_observes_every_frame_and_chunks_do_not_advance_it_again(self):
        policy=SimpleNamespace(config=SimpleNamespace(n_action_steps=2),
                               observe=MagicMock(side_effect=lambda batch,memory:(memory or 0)+1),
                               act=MagicMock(return_value=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])))
        queue=evaluator.TempoActionQueue(policy,lambda action:action)
        actions=[queue.next_action({'frame':i}) for i in range(5)]
        self.assertEqual(policy.observe.call_count,5)
        self.assertEqual(policy.act.call_count,3)
        self.assertEqual(queue.memory,5)
        np.testing.assert_equal(actions,[[1.,2.],[3.,4.],[1.,2.],[3.,4.],[1.,2.]])
        self.assertIsNone(evaluator.TempoActionQueue(policy,lambda x:x).memory)


if __name__=='__main__':unittest.main()
