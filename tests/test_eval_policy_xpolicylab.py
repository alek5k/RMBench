import unittest

from scripts.eval_policy_xpolicylab import build_policy_client, reset_policy


class RecordingClient:
    def __init__(self):
        self.calls = []

    def call(self, **kwargs):
        self.calls.append(kwargs)


class ResetPolicyTest(unittest.TestCase):
    def test_can_target_one_environment(self):
        client = RecordingClient()

        reset_policy(client, env_idx=7)
        reset_policy(client)

        self.assertEqual(client.calls, [
            {"func_name": "reset_envs", "obs": [7]},
            {"func_name": "reset"},
        ])

    def test_parallel_identity_policy_rejects_unserialized_legacy_tcp(self):
        for policy_name in ("LTE_SmolVLA", "SmolVLA"):
            with self.subTest(policy_name=policy_name):
                with self.assertRaisesRegex(ValueError, "serialized websocket protocol"):
                    build_policy_client({
                        "protocol": "legacy_tcp",
                        "policy_name": policy_name,
                        "eval_batch": True,
                        "num_workers": 2,
                    })
