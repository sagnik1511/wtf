"""Algorithm dispatch; neural architecture belongs to each brain template."""
from .brain import Brain


def create_brain(config, actions, base_seed, individual_seed):
    if config['type'] == 'q_network':
        return Brain(config, actions, base_seed, individual_seed)
    if config['type'] == 'ppo':
        from .ppo import PPOBrain
        return PPOBrain(config, actions, base_seed, individual_seed)
    if config['type'] in ('a2c', 'reinforce', 'grpo'):
        from .policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain
        return {'a2c': A2CBrain, 'reinforce': ReinforceBrain, 'grpo': GRPOBrain}[config['type']](config, actions, base_seed, individual_seed)
    raise ValueError(f'Unsupported brain type: {config["type"]}')


def restore_brain(state):
    kind = state['config']['type']
    if kind == 'q_network':
        return Brain.restore(state)
    if kind == 'ppo':
        from .ppo import PPOBrain
        return PPOBrain.restore(state)
    if kind in ('a2c', 'reinforce', 'grpo'):
        from .policy_gradients import A2CBrain, ReinforceBrain, GRPOBrain
        return {'a2c': A2CBrain, 'reinforce': ReinforceBrain, 'grpo': GRPOBrain}[kind].restore(state)
    raise ValueError(f'Unsupported checkpoint brain type: {kind}')
