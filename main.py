import cv2
import numpy as np
import math
from utils import create_environment, plot_metrics
from dqn_agent import DQNAgent


if __name__ == '__main__':
        
    #### Execution variables ####
    max_score = -math.inf # Intial max_score depends on game.(Pong --> - inf | Breakout --> 0))
    train = True
    load_model = False
    create_capture = False
    num_games = 5 # training-episodes
    ####

    env = create_environment('PongNoFrameskip-v4')
    agent_dqn = DQNAgent(gamma = 0.99, epsilon = 1.0, learning_rate=0.0001,
                        num_actions=env.action_space.n,
                        input_dim=(env.observation_space.shape), 
                        memory_size=50000,
                        eps_min=0.1, eps_dec=1e-5, batch_size=32, update=1000,
                        chkpt_dir='models/',
                        algorithm='DQN', env_name='PongNoFrameskip-v4')

    if load_model:
        agent_dqn.load_network_checkpoints()

    # training the agent    
    if train:
        figure_file_name = 'figures/' + agent_dqn.env_name + '_' + agent_dqn.algorithm + '_' \
            + str(num_games) + '_' + str(agent_dqn.learning_rate) + '.png'

        score_history = []
        eps_history = []
        steps_arr = []
        num_steps = 0

        for i in range(num_games):
            score = 0
            done = False
            observation = env.reset()

            while not done:
                action = agent_dqn.choose_action(observation)
                new_observation, reward, done, info = env.step(action)
                score += reward
                
                agent_dqn.store_transition(observation, action, reward, new_observation, int(done))
                agent_dqn.learn_step()
                
                observation = new_observation
                num_steps += 1

            steps_arr.append(num_steps)    
            score_history.append(score)
            eps_history.append(agent_dqn.epsilon)    
            avg_score = np.mean(score_history[-100:]) # moving average over last 100 games

            print('game-episode: ', i, 'episode-score: ', score, ' || average-score %.2f' % avg_score,
            'max-score %.2f' % max_score, ' || epsilon %.2f' % agent_dqn.epsilon, 'num_steps', num_steps)

            if score > max_score:
                max_score = score                
                agent_dqn.save_network_checkpoints()
        
        print("Training done!")
        plot_metrics(steps_arr, score_history, eps_history, figure_file_name)        
        
    # Capturing the agent playing
    if create_capture:        
        img_arr = []
        agent_dqn.epsilon = 0.0001
        observation = env.reset()

        for i in range(1000):
            frame = np.uint8(observation[0]*255)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            img_arr.append(frame)
            
            action = agent_dqn.choose_action(observation)
            observation, _, _, _ = env.step(action)
              
        out = cv2.VideoWriter('captures/'+agent_dqn.env_name+'.avi', cv2.VideoWriter_fourcc(*'DIVX'), 25, (84, 84))

        for i in range(len(img_arr)):
            out.write(img_arr[i])
        
        out.release
        print("Capturing done")