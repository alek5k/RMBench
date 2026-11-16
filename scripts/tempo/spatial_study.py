"""Strict guards for the two opt-in WaitAtGoal spatial reconstruction tests."""
from copy import deepcopy
import json
from pathlib import Path
from queue_reconstruction_baselines import validate_recipe as validate_target_recipe


def validate_recipe(recipe, reference, mode, defaults):
    if mode not in ('spatial4', 'native64'):
        raise ValueError('Unknown spatial ablation')
    t = recipe['training']; native = mode == 'native64'
    if (t.get('history_reconstruction_layout'), t.get('temporal_visual_pooling'),
            t.get('history_reconstruction_target'), t.get('history_reconstruction_diagnostics', False),
            recipe['memory']['spatial_pool_grid']) != ('patches', 'native' if native else 'adaptive', 'history', False, 8 if native else 2):
        raise ValueError('Incorrect spatial reconstruction or encoder geometry')
    if t.get('history_loss_balance') is not None:
        raise ValueError('Spatial study retains fixed history loss weight')
    if 'reconstruction-effectiveness' not in t.get('wandb_tags', []):
        raise ValueError('Missing study tag')
    matched = deepcopy(recipe)
    for key in ('history_reconstruction_layout', 'temporal_visual_pooling'):
        matched['training'].pop(key)
    # Reuse the exact seed-1, 10k schedule/provenance/identity guard without
    # weakening it for either historical target tests or the new spatial tests.
    matched['training']['history_reconstruction_target'] = 'history_mean'
    matched['training']['history_reconstruction_diagnostics'] = True
    matched['memory']['spatial_pool_grid'] = 2
    expected_cache = '/data/RMBench-state/embedding_cache/smolvla_waitatgoal_camera1_native64'
    if native:
        if t['embedding_cache_dir'] != expected_cache:
            raise ValueError('Native visual geometry needs its own cache')
        matched['training']['embedding_cache_dir'] = reference['training']['embedding_cache_dir']
    validate_target_recipe(matched, reference, 'history_mean', defaults)


def check_smoke(recipe):
    t = recipe['training']; root = Path(t['output_dir'] + '_smoke')
    marker = json.loads((root / 'smoke_passed.json').read_text())
    resolved = json.loads((root / 'resolved_config.json').read_text()); policy = resolved['policy']
    if not marker['finite'] or marker['actions_shape'] != [10, 2] or marker['memory_shape'] != [1, 4, 64]:
        raise ValueError('Smoke checkpoint reload/action/memory validation failed')
    if resolved['dataset_frames'] != 14008 or resolved['parameter_counts']['trainable_vlm'] != 0:
        raise ValueError('Unexpected dataset or unfrozen backbone')
    for key in ('history_reconstruction_layout', 'temporal_visual_pooling', 'history_reconstruction_target', 'temporal_objective'):
        if policy[key] != t[key]: raise ValueError('Actual smoke policy differs: ' + key)
    if (policy['memory']['spatial_pool_grid'] != recipe['memory']['spatial_pool_grid']
            or not policy['freeze_vision_encoder'] or not policy['train_expert_only']):
        raise ValueError('Actual spatial geometry/freezing differs')
    if resolved['pretrained_sha256'] != '7cd549ac2351fb069c0ddb3c34ad2d09cfc92b56a15dccdfc2e41467aaca01eb':
        raise ValueError('Reference pretrained weights differ')
    return resolved
