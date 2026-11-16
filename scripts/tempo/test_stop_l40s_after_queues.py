from copy import deepcopy
from pathlib import Path
import unittest
import yaml
from stop_l40s_after_queues import validate_recipe,validate_queue,barrier_ready,stop_arguments,AUTHORIZED,validate_instances


def fixture(run_id):
    spec={'kind':'suite','state':'/data/RMBench-state/queue.json','expected_runs':[
        {'id':run_id,'step':100000,'checkpoint':'/data/RMBench-state/checkpoints/'+run_id+'/100000'}]}
    snapshot={'state':{'stage':'complete','run_ids':[run_id],'completed':[{'training_run_id':run_id}]},
              'results':[{'training_run_id':run_id,'checkpoint_step':100000,'episodes':100,
                          'checkpoint':spec['expected_runs'][0]['checkpoint'],'successes':13,'success_rate':0.13}],
              'checkpoint_metadata':[{'step':100000,'training':{'wandb_run_id':run_id,'steps':100000}}]}
    return spec,snapshot


class StopL40STests(unittest.TestCase):
    def test_one_million_study_stops_only_its_exact_instance_after_final_eval(self):
        path=Path(__file__).resolve().parents[2]/'config/tempo/stop_cover_blocks_reference_1m_261114.yaml'
        recipe=yaml.safe_load(path.read_text());validate_recipe(recipe)
        self.assertEqual([(i['instance'],i['instance_id']) for i in recipe['instances']],
                         [('normal-scarlet-vole','pk47azbb5')])
        run=recipe['instances'][0]['queues'][0]['expected_runs'][0]
        self.assertEqual((run['id'],run['step']),('f2659c83',1000000))
        changed=deepcopy(recipe);changed['instances'][0]['instance']='sound-coral-echidna'
        with self.assertRaises(ValueError):validate_recipe(changed)

    def test_config_covers_nine_cloud_evals_excluding_moved_local_study(self):
        path=Path(__file__).resolve().parents[2]/'config/tempo/stop_l40s_after_all_queues.yaml'
        recipe=yaml.safe_load(path.read_text());validate_recipe(recipe)
        self.assertEqual(sum(len(q['expected_runs']) for i in recipe['instances'] for q in i['queues']),9)
        ids={r['id'] for i in recipe['instances'] for q in i['queues'] for r in q['expected_runs']}
        self.assertIn('b016a9c4',ids)
        self.assertFalse({'a4f73c21','c8e164b0'} & ids)
        with self.assertRaises(ValueError):validate_recipe({**recipe,'action':'delete'})
        with self.assertRaises(ValueError):validate_recipe({**recipe,'delete':True})
    def test_only_exact_full_final_results_release_barrier(self):
        spec,snapshot=fixture('a');self.assertTrue(validate_queue(spec,snapshot))
        for key,value in (('episodes',99),('checkpoint_step',20000),('training_run_id','wrong'),('checkpoint','/wrong/100000')):
            changed=deepcopy(snapshot);changed['results'][0][key]=value
            with self.assertRaises(ValueError):validate_queue(spec,changed)
        changed=deepcopy(snapshot);changed['checkpoint_metadata'][0]['training']['steps']=200000
        with self.assertRaises(ValueError):validate_queue(spec,changed)
        changed=deepcopy(snapshot);changed['state']['stage']='evaluating'
        self.assertFalse(validate_queue(spec,changed))
        changed['state']['stage']='failed'
        with self.assertRaises(RuntimeError):validate_queue(spec,changed)
    def test_both_instances_all_queues_and_idle_are_required(self):
        a,sa=fixture('a');b,sb=fixture('b')
        recipe={'instances':[{'instance':'first','queues':[a]},{'instance':'second','queues':[b]}]}
        snapshots={'first':{'queues':[sa],'gpu_processes':'','gpu_models':['NVIDIA L40S'],'active_jobs':[]},
                   'second':{'queues':[sb],'gpu_processes':'','gpu_models':['NVIDIA L40S'],'active_jobs':[]}}
        self.assertTrue(barrier_ready(recipe,snapshots))
        snapshots['second']['gpu_processes']='123'
        self.assertFalse(barrier_ready(recipe,snapshots));snapshots['second']['gpu_processes']=''
        snapshots['second']['active_jobs']=[{'pid':123}]
        self.assertFalse(barrier_ready(recipe,snapshots));snapshots['second']['active_jobs']=[]
        snapshots['second']['queues'][0]['state']['stage']='not_created_yet'
        self.assertFalse(barrier_ready(recipe,snapshots))
    def test_stop_commands_are_single_name_never_delete(self):
        for name in AUTHORIZED:self.assertEqual(stop_arguments(name),['brev','stop',name])
        for name in ('--all','*','other-instance'):
            with self.assertRaises(ValueError):stop_arguments(name)
        rows=[{'name':name,'id':id,'gpu':'L40S'} for name,id in AUTHORIZED.items()]
        validate_instances(rows)
        rows[0]['id']='wrong'
        with self.assertRaises(ValueError):validate_instances(rows)

    def test_incomplete_metadata_is_fail_closed_with_identifiable_evidence(self):
        rows=[{'name':name,'id':id,'gpu':'L40S'} for name,id in AUTHORIZED.items()]
        rows[0].pop('gpu')
        with self.assertRaisesRegex(ValueError, 'normal-scarlet-vole'):
            validate_instances(rows)

    def test_exact_sku_fills_missing_display_field_but_not_a_conflicting_gpu(self):
        rows=[{'name':name,'id':id,'gpu':'-','instance_type':'l40s-48gb.1x'} for name,id in AUTHORIZED.items()]
        validate_instances(rows)
        rows[0]['gpu']='A100'
        with self.assertRaises(ValueError):validate_instances(rows)

    def test_direct_probe_is_mandatory_even_when_all_queues_are_complete(self):
        spec,snapshot=fixture('a');recipe={'instances':[{'instance':'host','queues':[spec]}]}
        remote={'queues':[snapshot],'gpu_processes':'','active_jobs':[]}
        with self.assertRaises(ValueError):barrier_ready(recipe,{'host':remote})
        remote['gpu_models']=['NVIDIA A100']
        with self.assertRaises(ValueError):barrier_ready(recipe,{'host':remote})


if __name__=='__main__':unittest.main()
