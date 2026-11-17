"""Controlled neural choices for tests; never bypass physical reproduction gates."""
import copy
from ecosystem.library import ChampionBook


def force_action(b,action):
    index=b.action_indices[action]
    if hasattr(b,'epsilon'):
        b.epsilon=0.0;b.config['exploration']['minimum']=0.0
    if b.config['type']=='q_network':
        b.weights['b2'][:]=type(b.weights['b2'])([0.0]*len(b.actions));b.weights['b2'][index]=60.0
        b.target=copy.deepcopy(b.weights)
    else:
        network=b.network.actor if b.config['type']=='mappo' else b.network
        bias=network.layers[-1]['bias']
        count=len(b.actions)*b.quantiles if b.config['type']=='qrdqn' else len(b.actions)
        bias[:count]=type(bias)([0.0]*count)
        if b.config['type']=='qrdqn':
            for i in range(index*b.quantiles,(index+1)*b.quantiles):bias[i]=60.0
        else:bias[index]=60.0
        if hasattr(b,'target_network'):b.target_network.layers=copy.deepcopy(b.network.layers)
        if hasattr(b,'reference_network'):b.reference_network.layers=copy.deepcopy(b.network.layers)
    return b


def force_reproduction(experiment):
    for b in experiment.brains.values():force_action(b,'reproduce')
    experiment.champions=ChampionBook(experiment.config)
    if experiment.mode=='train':experiment.champions.consider(experiment.world.animals.values(),experiment.brains,experiment._champion_source())
    experiment._refresh_context()
    return experiment
