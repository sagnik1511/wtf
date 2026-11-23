"""Social work reuse must preserve visibility boundaries and detached recovery."""

# Regression fixtures inspect private state to verify ownership, caches and exact
# continuation without expanding the public API.
# pylint: disable=protected-access


# Test/fixture names and module narratives document the expected behavior.
# pylint: disable=missing-class-docstring,missing-function-docstring

import copy
import unittest
from unittest.mock import patch

from tests.test_social_learning import config, controlled
from tests.test_consequence_schedule import episode_config, KEY
from wtf.learning.consequences import Consequences
from wtf.simulation import social


class SocialEfficiencyTests(unittest.TestCase):
    def test_stationary_witness_checks_one_position_but_moving_checks_both(self):
        for moved in (False, True):
            with self.subTest(moved=moved):
                world = controlled(config())
                state = world._social
                state.training = False
                start = world.animals[1].position
                if moved:
                    world.animals[1].x += 1
                state.before = ({1: start}, {1: "rest"}, {0: [1]})
                # A moving subject is unseen at its old position. A stationary
                # subject has only one position and is visible there.
                with patch.object(
                    social, "_can_see", side_effect=[True, False]
                ) as visible:
                    state.finish(world, [], {})
                self.assertEqual(visible.call_count, 2 if moved else 1)
                counts = state.sightings[0][-1][1]
                self.assertEqual(sum(counts), 0 if moved else 1)

    def test_episode_checkpoint_is_detached_from_both_live_table_versions(self):
        model = Consequences(episode_config())
        model.learn("prey", KEY, [1, 0, 0])
        model.publish()
        model.learn("prey", KEY, [0, 1, 0])
        original = copy.deepcopy(model.checkpoint())
        state = model.checkpoint()
        state["tables"]["prey"][0][1][0] = 999
        state["staged_tables"]["prey"][0][1][0] = 888
        state["contracts"]["prey"]["diet"] = "changed"
        self.assertEqual(model.checkpoint(), original)
