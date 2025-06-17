''' An example of evluating the trained models in RLCard
'''
import os
import argparse

import rlcard
from rlcard.agents import (
    DQNAgent,
    RandomAgent,
)
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
)

def load_model(model_path, env=None, position=None, device=None):
    if os.path.isfile(model_path):  # Torch model
        import torch
        agent = torch.load(model_path, map_location=device, weights_only=False)
        agent.set_device(device)
        # dummy_input = (torch.randn(1, 1, 32, 32),)
        # onnx_program = torch.onnx.export(agent, dummy_input, "test_export.onnx")
        # onnx_program.optimize()
        # onnx_program.save(model_path.replace('.pth', '.onnx'))
    elif os.path.isdir(model_path):  # CFR model
        from rlcard.agents import CFRAgent
        agent = CFRAgent(env, model_path)
        agent.load()
    elif model_path == 'random':  # Random model
        from rlcard.agents import RandomAgent
        agent = RandomAgent(num_actions=env.num_actions)
    else:  # A model in the model zoo
        from rlcard import models
        agent = models.load(model_path).agents[position]
    
    return agent

def evaluate(args):

    # Check whether gpu is available
    device = get_device()
        
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Load models
    agents = []
    for position, model_path in enumerate(args.models):
        agents.append(load_model(model_path, env, position, device))
    env.set_agents(agents)

    # Evaluate
    rewards = tournament(env, args.num_games)
    for position, reward in enumerate(rewards):
        print(position, args.models[position], reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='okey',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'okey'
        ],
    )
    parser.add_argument(
        '--models',
        nargs='*',
        default=[
            # rlcard\\examples\\
            # "experiments\\dmc_result\\10_mio\\okey\\0_10004480.pth" # results: Average score 273.485 >=300: 0.4606 >=400: 0.0223
            # "experiments\\dmc_result\\num_buffers_200\\okey\\0_10001920.pth" # results: Average score 269.361 >=300: 0.4948 >=400: 0.0361
            # "experiments\\dmc_result\\num_buffers_200_copy\\okey\\0_15002176.pth" # results: Average score 250.245 Times >=300: 0.4981 Times >=400: 0.033
            "rlcard\\examples\\experiments\\dmc_result\\num_buffers_200_copy\\okey\\0_20001856.pth" # results: Average score 244.542 Times >=300: 0.5076 Times >=400: 0.0346
            # "experiments\\dmc_result\\num_buffers_300_20mio\\okey\\0_20001920.pth" # results: Average score 242.445 >=300: 0.4965 >=400: 0.0243
            # "experiments\\dmc_result\\adjusted_hyperparams\\okey\\0_10001920.pth" # results: Average score 259.131 >=300: 0.4863 >=400: 0.0326
            # "experiments\\dmc_result\\num_buffers_200_exp_005\\okey\\0_10001920.pth" # results: Average score 246.42 Times >=300: 0.4539 Times >=400: 0.0326
            # "experiments\\dmc_result\\num_buffers_220\\okey\\0_20001920.pth" # results: Average score 248.71 Times >=300: 0.4872 Times >=400: 0.0326
            # "experiments\\dmc_result\\num_buffers_190\\okey\\0_10001920.pth" # results: Average score 252.646 Times >=300: 0.4712 Times >=400: 0.0281
            # "experiments\\dmc_result\\reward_300_only\\okey\\0_10002240.pth" # results: Average score 242.138 Times >=300: 0.4575 Times >=400: 0.0272
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )
    parser.add_argument(
        '--num_games',
        type=int,
        default=10000,
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    evaluate(args)

