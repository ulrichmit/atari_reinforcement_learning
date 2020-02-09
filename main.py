import numpy as np
from utils import create_environment, plot_metrics
from dqn_agent import DQNAgent


if __name__ == '__main__':
    env = create_environment('PongNoFrameskip-v4')
    # setting the score so that the first model is saved
    max_score = -np.inf # +np.inf if the score is positive

    num_games = 1 # 500
    testing = False # True for testing

    agent_dqn = DQNAgent(gamma = 0.99, epsilon = 1.0,learning_rate=0.0001,
                        num_actions=env.action_space.n,
                        input_dim=(env.observation_space.shape), 
                        memory_size=50000,
                        eps_min=0.1, eps_dec=1e-5, batch_size=32, update=1000,
                        chkpt_dir='/git_workspace_mac/atari_reinforcement_learning/models',
                        algorithm='DQN', env_name='PongNoFrameskip-v4')

    if testing:
        agent_dqn.load_network_checkpoints()

# building a figure file-name
figure_file_name = 'figures/' + agent_dqn.env_name + '_' + agent_dqn.algorithm + '_' \
    + str(num_games) + '_' + str(agent_dqn.learning_rate) + '.png'

# variables for plotting
score_history = []
eps_history = []
steps_arr = []
num_steps = 0

for i in range(num_games):
    score = 0
    done = False
    observation = env.reset()

    while not done:
        action = agent_dqn.choose_action(observation) # choosing an action
        new_observation, reward, done, info = env.step(action) # taking action
        score += reward

        if not testing:
            agent_dqn.store_transition(observation, action, reward, new_observation, int(done))
            agent_dqn.learn_step()
        
        observation = new_observation
        num_steps += 1

        steps_arr.append(num_steps)    
        score_history.append(score)
        eps_history.append(agent_dqn.epsilon)
        
        avg_score = np.mean(score_history[-100:]) # average over last 100 games
        print('game-episode: ', i, 'score: ', score, 'average-score %.2f' % avg_score,
        'max-score %.2f' % max_score, 'epsiolon %.2f' % agent_dqn.epsilon, 'num_steps', num_steps)

        if avg_score > max_score:
            max_score = avg_score

            if not testing:
                agent_dqn.save_network_checkpoints()

plot_metrics(steps_arr, score_history, eps_history, figure_file_name)
        

