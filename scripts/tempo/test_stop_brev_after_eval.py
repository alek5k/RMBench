import unittest
from stop_brev_after_eval import validate_completed


class StopAfterEvalTests(unittest.TestCase):
    def setUp(self):
        self.recipe={'training_run_id':'b871c40f','checkpoint':'/trained/100000'}
        self.remote={'state':{'training_run_id':'b871c40f','stage':'complete'},'result':{
            'training_run_id':'b871c40f','checkpoint_step':100000,'episodes':100,'checkpoint':'/trained/100000'}}

    def test_only_full_final_result_releases_stop(self):
        self.assertTrue(validate_completed(self.recipe,self.remote))
        self.assertFalse(validate_completed(self.recipe,{**self.remote,'state':{**self.remote['state'],'stage':'evaluating'}}))
        for key,value in [('episodes',99),('checkpoint_step',30000),('training_run_id','wrong'),('checkpoint','/other/100000')]:
            with self.subTest(key=key),self.assertRaises(ValueError):
                validate_completed(self.recipe,{**self.remote,'result':{**self.remote['result'],key:value}})

    def test_failed_eval_never_releases_stop(self):
        with self.assertRaises(RuntimeError):
            validate_completed(self.recipe,{**self.remote,'state':{**self.remote['state'],'stage':'failed'}})


if __name__=='__main__':unittest.main()
