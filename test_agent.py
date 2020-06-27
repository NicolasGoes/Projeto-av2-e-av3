# Pacotes e modulos terceiros
from collections import deque
from datetime import datetime
import numpy as np
import gym, os, json
# Meus pacotes e módulos:
from agent import Agent
import utils

def run_episode(env, agent, rendering=True, max_timesteps=1000):
    # Redefinir acumulador de recompensa
    episode_reward = 0
    # Informe o ambiente e o agente que um novo episódio está prestes a começar:
    env_state = env.reset()
    agent.begin_new_episode(state0=env_state)

    for _ in range(max_timesteps):
        # Solicitar ação do agente:
        agent_action = agent.get_action(env_state)
        # Dada essa ação, obtenha o próximo estado do ambiente e recompense:
        env_state, r, done, info = env.step(agent_action)   
        # Renderize a tela de estado:
        if rendering:
            env.render()
        # Acumular recompensa
        episode_reward += r
        # Verifique se o ambiente sinalizou o final do episódio:
        if done: break

    return episode_reward

def save_performance_results(episode_rewards, directory):
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{directory}results_bc_agent-{time_stamp}.json"
    with open(fname, "w") as fh:
        json.dump(results, fh)


if __name__ == "__main__":
    # Número de episódios a serem testados:
    n_test_episodes = 50

    # Inicialize o ambiente e o agente:
    env = gym.make('CarRacing-v0').unwrapped
    agent = Agent.from_file('saved_models/')

    # Episodes loop:
    episode_rewards = []
    for i in range(n_test_episodes):
        episode_reward = run_episode(env, agent, rendering=True)
        episode_rewards.append(episode_reward)
        print(f'Episode {i+1} reward:{episode_reward:.2f}')
    env.close()

    # salvar estatísticas de recompensa em um arquivo .json
    save_performance_results(episode_rewards, 'performance_results/')
    print('... finished')
