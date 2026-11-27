"""Phase projections preserve arbitrary species, ties and private sensory state."""

# The optional compiled API is required to compare native and Python observations.
# pylint: disable=missing-class-docstring,missing-function-docstring,protected-access

import copy
import random
import unittest

from tests import test_cpp_sensing as reference
from wtf.simulation.world import World


@unittest.skipUnless(
    reference._engine is not None, "Build the optional native PPO engine"
)
class NativeSensingPhaseTests(unittest.TestCase):
    def test_phase_projection_matches_multispecies_dynamic_world(self):
        generator = random.Random(417)
        for case in range(9):
            config = reference.configuration()
            for species in config["species"].values():
                species["initial_population"] = 4
                species["body"]["field_of_view_degrees"] = (150, 180, 360)[case % 3]
            if case % 3 == 0:
                config["species"]["grazer"] = copy.deepcopy(config["species"]["prey"])
            for template in config["brains"].values():
                template["observation_top_k"] = (1, 3, 10)[case % 3]
                template["movement_history_steps"] = 0 if case % 2 else 3
                template["scent_observation"] = (
                    {"mode": "blended", "radius": 2}
                    if case % 2
                    else {
                        "mode": "separated",
                        "species": list(config["species"]),
                        "separate_sexes": case % 4 == 0,
                        "radius": 3,
                    }
                )
            world = World(config, case + 31)
            for animal in world.animals.values():
                animal.x = generator.randrange(world.width)
                animal.y = generator.randrange(world.height)
                animal.heading = generator.randrange(4)
                animal.age = generator.randrange(700)
                animal.last_reproduced_age = generator.choice([-1, 10, 100])
                animal.reproduction_due_age = generator.choice([-1, 50, 300])
                animal.mating_intent_until_tick = generator.choice([-1, 2, 10])
            for tick in range(5):
                with self.subTest(case=case, tick=tick):
                    reference.NativeSensingTests.assert_observations(
                        self, world.checkpoint()
                    )
                world.step(
                    {
                        animal.id: generator.choice(
                            config["species"][animal.species]["allowed_actions"]
                        )
                        for animal in world.animals.values()
                    }
                )


if __name__ == "__main__":
    unittest.main()
