import torch
import argparse
from unityagents import UnityEnvironment
import numpy as np
from agents import Agent_DDPG, agent_train, Agent_A2C, Agent_TD3, Agent_TD3_4
from utils import load_env


def main():
    # Command line Arguments
    parser = argparse.ArgumentParser("DQN")
    parser.add_argument("--mode", type=str, help="training , play, compare, complare_play, plot, hp_tuning",
                        required=True)
    parser.add_argument("--type", type=str, help="type 1-->DDQN , type 2--> A2C, type 3--> Dueling"
                                                 "DQN, no PBR, type 4-->categorical DQN, type 5--> Duelling DQN"
                                                 " with Noisy layer and PBR, Type 6--> DQN n-steps, type 7 --> "
                                                 "Rainbow DQN", required=True)
    parser.add_argument("--agent", type=str, help="type of agent: 1--> 1 or 2--> 20 arms",
                        required=True)
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    algo = args.type  # <-- type of algo to use for training
    agent_type= args.agent
    # load environment
    if args.mode != "compare" and args.mode != "compare_play" and args.mode != "plot" and args.mode != "hp_tuning":
        if args.mode == "training" and agent_type == "2":
            file_name = "../Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                        base_port,file_name,True, True)
        elif args.mode == "training" and agent_type == "1":
            file_name = "../Reacher_Windows_x86_One/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  True, True)
        elif args.mode == "play" and agent_type == "2":
            file_name = "../Reacher_Windows_x86_64_20/Reacher_Windows_x86_64/Reacher.exe"
            worker_id = 1
            base_port = 5005
            env, brain_name, brain, action_size, env_info, state, state_size, n_agents = load_env(worker_id,
                                                                                                  base_port, file_name,
                                                                                                  False, False)

    print('Device:' , DEVICE)
    CHECKPOINT_FOLDER = './models/'

    if args.mode == "training" and algo =="1": #DDQN

        # hyperparameters
        BUFFER_SIZE = int(1e5)  # replay buffer size
        BATCH_SIZE = 128        # minibatch size
        GAMMA = 0.99            # discount factor
        TAU = 1e-3              # for soft update of target parameters
        LR_ACTOR = 1e-4         # learning rate of the actor
        LR_CRITIC = 1e-4        # learning rate of the critic
        WEIGHT_DECAY = 0        # L2 weight decay




        agent = Agent_DDPG(
                        DEVICE,
                        state_size, n_agents, action_size, 4,
                        BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                        algo, CHECKPOINT_FOLDER, True,
                )


        agent_train(env,brain_name, agent, n_agents, algo )
        env.close()
    elif args.mode == "training" and algo == "2":  # TD3 4 Critics
        # hyperparameters
        BUFFER_SIZE = int(1e5)  # replay buffer size
        BATCH_SIZE = 128  # minibatch size
        GAMMA = 0.99  # discount factor
        TAU = 5e-3  # for soft update of target parameters
        LR_ACTOR = 3e-4  # learning rate of the actor
        LR_CRITIC = 1e-3  # learning rate of the critic
        WEIGHT_DECAY = 0  # L2 weight decay
        exploration_noise = 0.1
        target_policy_noise = 0.2
        target_policy_noise_clip = 0.5

        agent = Agent_TD3_4(
            DEVICE,
            state_size, n_agents, action_size, 4,
            BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
            algo, exploration_noise, target_policy_noise, target_policy_noise_clip,
            True, int(1e4), CHECKPOINT_FOLDER, True,
        )

        agent_train(env, brain_name, agent, n_agents, algo)
        env.close()
    elif args.mode == "training" and algo =="3": #TD3

        # hyperparameters
        BUFFER_SIZE = int(1e5)  # replay buffer size
        BATCH_SIZE = 128        # minibatch size
        GAMMA = 0.99            # discount factor
        TAU = 5e-3              # for soft update of target parameters
        LR_ACTOR = 3e-4         # learning rate of the actor
        LR_CRITIC = 1e-3        # learning rate of the critic
        WEIGHT_DECAY = 0        # L2 weight decay
        exploration_noise = 0.1
        target_policy_noise = 0.2
        target_policy_noise_clip = 0.5




        agent = Agent_TD3(
                        DEVICE,
                        state_size, n_agents, action_size, 4,
                        BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                        algo, exploration_noise, target_policy_noise, target_policy_noise_clip,
                        True,  int(1e4),CHECKPOINT_FOLDER, True,
                )


        agent_train(env,brain_name, agent, n_agents, algo )
        env.close()
    elif args.mode == "play" :

        if algo == "1":#DDPG
            # hyperparameters
            BUFFER_SIZE = int(1e5)  # replay buffer size
            BATCH_SIZE = 128  # minibatch size
            GAMMA = 0.99  # discount factor
            TAU = 1e-3  # for soft update of target parameters
            LR_ACTOR = 1e-4  # learning rate of the actor
            LR_CRITIC = 1e-4  # learning rate of the critic
            WEIGHT_DECAY = 0  # L2 weight decay

            # test the trained agent
            agent = Agent_DDPG(
                DEVICE, state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, CHECKPOINT_FOLDER, False
            )
        elif algo == "3":
            # hyperparameters
            BUFFER_SIZE = int(1e5)  # replay buffer size
            BATCH_SIZE = 128  # minibatch size
            GAMMA = 0.99  # discount factor
            TAU = 5e-3  # for soft update of target parameters
            LR_ACTOR = 3e-4  # learning rate of the actor
            LR_CRITIC = 1e-3  # learning rate of the critic
            WEIGHT_DECAY = 0  # L2 weight decay
            exploration_noise = 0.1
            target_policy_noise = 0.2
            target_policy_noise_clip = 0.5

            agent = Agent_TD3(DEVICE,
                state_size, n_agents, action_size, 4,
                BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC, WEIGHT_DECAY,
                algo, exploration_noise, target_policy_noise, target_policy_noise_clip,
                True, int(1e4), CHECKPOINT_FOLDER, False)

        #agent.actor_local.load_state_dict(torch.load("./checkpoints/checkpoint_actor.pth"))
        for episode in range(3):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            score = np.zeros(n_agents)

            while True:
                actions = agent.act(states, 10000,add_noise=False)

                env_info = env.step(actions)[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done
                score += rewards
                states = next_states

                if np.any(dones):
                    break

            print('Episode: \t{} \tScore: \t{:.2f}'.format(episode, np.mean(score)))

        env.close()

if __name__ == '__main__':
    main()
