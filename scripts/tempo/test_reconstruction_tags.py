from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from tag_reconstruction_runs import append_tag, synchronize


def run(run_id, config=None):
    return SimpleNamespace(id=run_id, name='display-name', tags=['existing'], config=config or {}, update=Mock())


class ReconstructionTagTests(unittest.TestCase):
    def test_append_preserves_tags_and_is_idempotent(self):
        item=run('a'); append_tag(item,'study'); append_tag(item,'study')
        self.assertEqual(item.tags,['existing','study']);item.update.assert_called_once()

    def test_matches_source_ids_and_waits_for_both_final_evals(self):
        training=[run('a'),run('b')]
        evaluation=run('eval-a',{'training_run_id':'a','checkpoint_step':10000,'episodes':100})
        wrong_step=run('eval-b',{'training_run_id':'b','checkpoint_step':3,'episodes':100})
        def runs(project,filters):
            if 'name' in filters:return [r for r in training if r.id==filters['name']]
            return [r for r in (evaluation,wrong_step) if r.config['training_run_id']==filters['config.training_run_id']]
        report=synchronize(SimpleNamespace(runs=runs),'entity/project',['a','b'],'study')
        self.assertFalse(report['complete']);self.assertEqual(report['evaluation_sources_found'],['a'])
        self.assertEqual(wrong_step.tags,['existing'])
        wrong_step.config['checkpoint_step']=10000
        self.assertTrue(synchronize(SimpleNamespace(runs=runs),'entity/project',['a','b'],'study')['complete'])


if __name__=='__main__':unittest.main()
