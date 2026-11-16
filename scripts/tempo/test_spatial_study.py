from copy import deepcopy
from pathlib import Path
import unittest
import yaml
from spatial_study import validate_recipe
from queue_reconstruction_baselines import validate_eval


class SpatialStudyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[2] / 'config/tempo'
        self.reference = self.load('waitatgoal_tempo_mild_lr_decay_horizon10_10k_seed1.yaml')
        self.defaults = self.load('waitatgoal_defaults.yaml')
    def load(self, name): return yaml.safe_load((self.root / name).read_text())
    def test_exact_two_opt_in_spatial_recipes(self):
        for mode in ('spatial4', 'native64'):
            name = f'waitatgoal_tempo_{mode}_horizon10_10k_seed1.yaml'
            recipe = self.load(name)
            validate_recipe(recipe, self.reference, mode, self.defaults)
            validate_eval(recipe['training'], self.load('eval_' + name))
    def test_drift_is_rejected(self):
        for mode in ('spatial4', 'native64'):
            recipe = self.load(f'waitatgoal_tempo_{mode}_horizon10_10k_seed1.yaml')
            for key, value in (('batch_size', 16), ('seed', 0), ('minimum_lr', 1e-6),
                               ('temporal_objective', 'reconstruction_only'), ('history_reconstruction_layout', 'mean'),
                               ('temporal_visual_pooling', 'invalid'), ('history_loss_balance', {})):
                changed = deepcopy(recipe); changed['training'][key] = value
                with self.assertRaises(ValueError): validate_recipe(changed, self.reference, mode, self.defaults)
            changed = deepcopy(recipe); changed['memory']['history_queries'] = 64
            with self.assertRaises(ValueError): validate_recipe(changed, self.reference, mode, self.defaults)


if __name__ == '__main__': unittest.main()
