"""Native private workspaces must not retain inputs or alter pending policies."""

# Optional compiled probes deliberately inspect exact continuation state.
# pylint: disable=protected-access,missing-class-docstring,missing-function-docstring
# pylint: disable=c-extension-no-member


import math
import unittest

from tests.native_core_support import _engine, encoded
from tests.test_cpp_ppo import observation, settings
from wtf.learning.brain import Experience
from wtf.learning.ppo import PPOBrain


@unittest.skipUnless(_engine is not None, "Build the optional native PPO engine")
class NativePPOScratchTests(unittest.TestCase):
    def test_repeated_inputs_invalidate_hidden_cache_after_every_weight_update(self):
        brain = PPOBrain(settings([5, 4, 3]), ["rest", "eat", "move"], 17, 901)
        handle = _engine.ppo_create(brain.checkpoint())
        for step in range(96):
            # Alternating signed zeros must not be mistaken for an identical
            # cached bit pattern, even though Python numeric equality accepts it.
            current = tuple(-0.0 if step % 2 else 0.0 for _ in range(20))
            action = brain.choose(current)
            self.assertEqual(action, _engine.ppo_choose(handle, current, True))
            self.assertEqual(
                encoded(brain.checkpoint()), encoded(_engine.ppo_checkpoint(handle))
            )
            terminal = step == 95
            next_input = None if terminal else current
            reward = math.sin(step * 0.7)
            brain.learn(
                Experience(current, action, reward, next_input, terminal, False)
            )
            _engine.ppo_learn(
                handle, current, action, reward, next_input, terminal, False
            )
            self.assertEqual(
                encoded(brain.checkpoint()), encoded(_engine.ppo_checkpoint(handle))
            )
            self.assertEqual(
                encoded(brain.diagnostics(current, include_activations=True)),
                encoded(_engine.ppo_diagnostics(handle, current, True)),
            )

    def test_interleaved_value_only_and_diagnostic_tapes_preserve_every_parameter(self):
        for hidden in ([], [5, 4, 3]):
            for target_kl in (0, 1e-5):
                with self.subTest(hidden=hidden, target_kl=target_kl):
                    brain = PPOBrain(
                        settings(hidden, target_kl), ["rest", "eat", "move"], 17, 901
                    )
                    handle = _engine.ppo_create(brain.checkpoint())
                    for step in range(48):
                        current, following = observation(step), observation(step + 1)
                        action = brain.choose(current)
                        self.assertEqual(
                            action, _engine.ppo_choose(handle, current, True)
                        )
                        # Unrelated diagnostics/evaluation reuse native forward
                        # storage but must not overwrite the selected sample.
                        for offset in (11, 37):
                            other = observation(step + offset)
                            self.assertEqual(
                                encoded(
                                    brain.diagnostics(other, include_activations=True)
                                ),
                                encoded(_engine.ppo_diagnostics(handle, other, True)),
                            )
                            self.assertEqual(
                                brain.choose(other, explore=False),
                                _engine.ppo_choose(handle, other, False),
                            )
                        self.assertEqual(
                            encoded(brain.checkpoint()),
                            encoded(_engine.ppo_checkpoint(handle)),
                        )
                        if step % 7 == 3:
                            current = observation(step + 101)
                        if step % 9 == 2:
                            action = brain.actions[
                                (brain.action_indices[action] + 1) % 3
                            ]
                        terminal, truncated = step in (9, 47), step in (17, 33)
                        next_input = None if terminal else following
                        reward = math.cos(step * 0.31) - 0.2
                        brain.learn(
                            Experience(
                                current, action, reward, next_input, terminal, truncated
                            )
                        )
                        _engine.ppo_learn(
                            handle,
                            current,
                            action,
                            reward,
                            next_input,
                            terminal,
                            truncated,
                        )
                        self.assertEqual(
                            encoded(brain.checkpoint()),
                            encoded(_engine.ppo_checkpoint(handle)),
                        )
                        if step in (4, 19, 34):
                            # Scratch is ephemeral; rebuilding it after restore
                            # must produce the same learning continuation.
                            brain = PPOBrain.restore(brain.checkpoint())
                            handle = _engine.ppo_create(_engine.ppo_checkpoint(handle))


if __name__ == "__main__":
    unittest.main()
