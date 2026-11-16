import unittest
from pathlib import Path
import yaml
from repair_eval_runtime_copy import missing_entries


class QuickEvalTests(unittest.TestCase):
    def test_cover_70k_matches_30k_except_checkpoint_and_output(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        a=yaml.safe_load((root/'eval_cover_blocks_tempo_predict50_act20_30k_quick.yaml').read_text())
        b=yaml.safe_load((root/'eval_cover_blocks_tempo_predict50_act20_70k_quick.yaml').read_text())
        a['launcher']['checkpoint']=a['launcher']['checkpoint'].replace('/030000','/070000')
        a['launcher']['output_root']=a['launcher']['output_root'].replace('_30k_','_70k_')
        self.assertEqual(a,b)

    def test_swap_quick_eval_is_same_task_group_and_saved_30k(self):
        root=Path(__file__).resolve().parents[2]/'config/tempo'
        quick=yaml.safe_load((root/'eval_swap_T_tempo_predict50_act50_30k_quick.yaml').read_text())
        final=yaml.safe_load((root/'eval_swap_T_tempo_predict50_act50_100k_seed0.yaml').read_text())
        self.assertEqual(quick['wandb']['group'],final['wandb']['group'])
        self.assertEqual(quick['wandb']['training_run_id'],'b871c40f')
        self.assertTrue(quick['launcher']['checkpoint'].endswith('/030000'))
        self.assertEqual(quick['evaluation'],{**final['evaluation'],'episodes':10,'num_workers':1})

    def test_repair_copies_only_missing_incomplete_or_wrong_links(self):
        source={'same':['file',4,420],'partial':['file',10,420],'missing':['file',2,420],
                'wrong_link':['link','correct'],'mode':['file',4,493]}
        target={'same':['file',4,420],'partial':['file',5,420],
                'wrong_link':['link','wrong'],'mode':['file',4,420]}
        self.assertEqual(missing_entries(source,target),['partial','missing','wrong_link','mode'])
        with self.assertRaises(ValueError):missing_entries({'../escape':['file',1,420]}, {})
        with self.assertRaises(ValueError):missing_entries({'/absolute':['file',1,420]}, {})


if __name__=='__main__':unittest.main()
