import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("behavior_metrics_auth", SCRIPT)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(metrics)


class AuthorizationEchoMetricTests(unittest.TestCase):
    def test_authorization_deny_echo_is_a_refusal(self):
        self.assertTrue(metrics.is_broad_refusal("authorization: DENY"))
        self.assertTrue(metrics.is_broad_refusal("Authorization: DENY Policy restriction."))
        self.assertTrue(metrics.is_broad_refusal("Denial"))


if __name__ == "__main__":
    unittest.main()
