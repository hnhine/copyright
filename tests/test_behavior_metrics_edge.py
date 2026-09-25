import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "metrics.py"
SPEC = importlib.util.spec_from_file_location("behavior_metrics_edge", SCRIPT)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(metrics)


class BehaviorMetricEdgeTests(unittest.TestCase):
    def test_short_deny_is_a_refusal(self):
        self.assertTrue(metrics.is_broad_refusal("Deny"))
        self.assertTrue(metrics.is_broad_refusal("Denied: policy restriction."))


if __name__ == "__main__":
    unittest.main()
