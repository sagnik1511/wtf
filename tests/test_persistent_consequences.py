"""General knowledge survives bounded context eviction and exact restoration."""

# Descriptive test names carry the assertions without redundant method docstrings.
# pylint: disable=missing-function-docstring

import copy
import unittest
from tests.test_social_learning import config
from wtf.config import validate
from wtf.learning.consequences import Consequences


def context(index, action=0):
    """Distinct valid bounded contexts, sharing one action."""
    return (
        action,
        index % 4,
        (index // 4) % 4,
        (index // 16) % 4,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
    )


def persistent_config(schedule="continuous"):
    """Small cache makes forgetting and staging boundaries observable."""
    c = config()
    c["consequence_learning"].update(
        persistent_action_totals=True, update_schedule=schedule
    )
    return validate(c)


class PersistentConsequencesTests(unittest.TestCase):
    """Fixed-width totals have independent ownership from the LRU."""

    def test_eviction_preserves_general_counts_and_unknown_context(self):
        model = Consequences(persistent_config())
        for i in range(40):
            model.learn("prey", context(i), [i < 20, 0, 0])
        self.assertEqual(len(model.tables["prey"]), 16)
        self.assertEqual(model.predict("prey", context(0)), [0.0] * 4)
        self.assertEqual(model.predict_action("prey", 0), [40 / 44, 0.5, 0, 0])
        self.assertEqual(model.predict_action("predator", 0), [0.0] * 4)

    def test_episode_staging_and_checkpoint_restore_do_not_publish_early(self):
        c = persistent_config("episode")
        model = Consequences(c)
        model.learn("prey", context(0), [1, 0, 0])
        self.assertEqual(model.predict_action("prey", 0), [0.0] * 4)
        state = model.checkpoint()
        restored = Consequences.restore(c, state)
        self.assertEqual(restored.checkpoint(), state)
        restored.publish()
        self.assertEqual(restored.predict_action("prey", 0), [0.2, 1, 0, 0])
        self.assertEqual(model.predict_action("prey", 0), [0.0] * 4)
        restored.learn("prey", context(1), [0, 1, 0])
        self.assertEqual(restored.predict_action("prey", 0), [0.2, 1, 0, 0])
        export = restored.export_state()
        warm = Consequences.restore(c, export)
        self.assertEqual(warm.predict_action("prey", 0), [2 / 6, 0.5, 0.5, 0])

    def test_legacy_import_bootstraps_available_rows_without_inventing_history(self):
        old = config()
        model = Consequences(old)
        model.learn("prey", context(0), [1, 0, 0])
        model.learn("prey", context(1), [0, 1, 0])
        restored = Consequences.restore(persistent_config(), model.export_state())
        self.assertEqual(restored.predict_action("prey", 0), [2 / 6, 0.5, 0.5, 0])
        for i in range(2, 40):
            restored.learn("prey", context(i), [0, 0, 0])
        self.assertEqual(restored.predict_action("prey", 0)[1:3], [1 / 40, 1 / 40])

    def test_opt_out_and_malformed_state_are_explicit(self):
        c = persistent_config()
        model = Consequences(c)
        model.learn("prey", context(0), [0, 1, 0])
        state = model.checkpoint()
        with self.assertRaises(ValueError):
            Consequences.restore(config(), state)
        for value in (True, float("nan"), -1):
            bad = copy.deepcopy(state)
            bad["action_totals"]["prey"][0][0] = value
            with self.assertRaises(ValueError):
                Consequences.restore(c, bad)
        bad = persistent_config()
        bad["consequence_learning"]["persistent_action_totals"] = 1
        with self.assertRaises(ValueError):
            validate(bad)
