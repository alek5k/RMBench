"""No-GPU archive identity, partial-save, integrity and publication checks."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import archive_temporal_checkpoints as archive


class CheckpointArchiveTests(unittest.TestCase):
    def fixture(self,root):
        checkpoint=root/'remote/010000';checkpoint.mkdir(parents=True)
        assets=root/'revision';assets.mkdir();(assets/'config.json').write_text('{}')
        for name in archive.REQUIRED:(checkpoint/name).write_text('checkpoint fixture')
        (checkpoint/'tempo_run.json').write_text(json.dumps({'step':10000,'training':{'wandb_run_id':'run1'}}))
        (checkpoint/'config.json').write_text(json.dumps({'vlm_model_name':str(assets)}))
        jobs=[{'run_id':'run1','remote_root':str(root/'remote'),'expected_steps':[10000,20000],
               'local_name':'baseline_run1','vlm_revision':'revision'}]
        return checkpoint,jobs

    def test_running_run_exports_completed_checkpoint_but_not_partial_save(self):
        with TemporaryDirectory() as d:
            root=Path(d);checkpoint,jobs=self.fixture(root)
            (checkpoint.parent/'020000').mkdir()
            report=archive.remote_inventory(jobs)
            self.assertEqual([c['step'] for c in report['runs'][0]['checkpoints']],[10000])
            self.assertEqual(report['runs'][0]['expected_steps'],[10000,20000])
            archive.verify(checkpoint,report['runs'][0]['checkpoints'][0]['files'])

    def test_wrong_run_checkpoint_refused(self):
        with TemporaryDirectory() as d:
            root=Path(d);checkpoint,jobs=self.fixture(root)
            jobs[0]['run_id']='other'
            with self.assertRaisesRegex(ValueError,'identity mismatch'):archive.remote_inventory(jobs)

    def test_corruption_and_path_escape_refused(self):
        with TemporaryDirectory() as d:
            root=Path(d);checkpoint,_=self.fixture(root)
            entries=archive.inventory(checkpoint)
            (checkpoint/'optimizer.pt').write_text('corrupt')
            with self.assertRaisesRegex(ValueError,'Checksum mismatch'):archive.verify(checkpoint,entries)
            with self.assertRaisesRegex(ValueError,'Unsafe inventory'):
                archive.verify(root,[{'path':'../outside','size':0,'sha256':''}])

    def test_transfer_is_published_only_after_hash_verification(self):
        with TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'model').write_bytes(b'weights')
            entries=archive.inventory(source);dest=root/'local/010000'
            def rsync(command,**kw):
                staging=Path(command[-1]);self.assertFalse(dest.exists())
                (staging/'model').write_bytes(b'weights')
            with patch.object(archive.subprocess,'run',side_effect=rsync) as execute:
                archive.transfer('instance','/remote/010000',dest,entries)
                self.assertTrue(json.loads((dest/'archive_verified.json').read_text())['verified'])
                archive.transfer('instance','/remote/010000',dest,entries)
                self.assertEqual(execute.call_count,1)
            self.assertFalse(dest.with_name('.010000.incoming').exists())

    def test_failed_transfer_does_not_publish_a_checkpoint(self):
        with TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();(source/'model').write_bytes(b'weights')
            entries=archive.inventory(source);dest=root/'010000'
            def rsync(command,**kw):(Path(command[-1])/'model').write_bytes(b'corrupt')
            with patch.object(archive.subprocess,'run',side_effect=rsync):
                with self.assertRaisesRegex(ValueError,'Checksum mismatch'):
                    archive.transfer('instance','/remote/010000',dest,entries)
            self.assertFalse(dest.exists())
            self.assertTrue(dest.with_name('.010000.incoming').exists())


if __name__=='__main__':unittest.main()
