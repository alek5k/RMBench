import unittest
import repair_cover_checkpoint_evals_local as repair


class CoverEvalRepairTests(unittest.TestCase):
    def previous(self):
        return [{'episode':i,'worker':0,'seed':seed,'success':i % 7 == 0}
                for i,seed in zip([i for i in range(100) if i not in (2,3)],
                                  [s for s in range(100000,100100) if s not in (100001,100002)],strict=True)]

    def test_merge_preserves_every_record_and_recovers_only_missing_seeds(self):
        previous=self.previous()
        fresh=[{'episode':0,'worker':0,'seed':100001,'success':True},
               {'episode':1,'worker':0,'seed':100002,'success':False}]
        result=repair.merge_records(previous,fresh)
        self.assertEqual(result[:98],previous)
        self.assertEqual([r['episode'] for r in result[-2:]],[2,3])
        self.assertEqual(len({r['seed'] for r in result}),100)
        self.assertEqual(repair.episode_records(repair.canonical_log(result)),result)
        self.assertIn(f"Final batch success rate: {sum(r['success'] for r in result)}/100",repair.canonical_log(result))

    def test_wrong_duplicate_or_incomplete_seed_evidence_is_rejected(self):
        previous=self.previous()
        with self.assertRaises(ValueError):repair.missing_seeds(previous[:-1])
        with self.assertRaises(ValueError):repair.merge_records(previous,[{'episode':0,'worker':0,'seed':100003,'success':True}])
        text=repair.canonical_log(previous)
        with self.assertRaises(ValueError):repair.episode_records(text+text)


if __name__=='__main__':unittest.main()
