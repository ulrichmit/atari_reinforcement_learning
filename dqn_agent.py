import torch as T
import numpy as np
from dqn_network import DeepQCNN
from experience_buffer import ExperienceReplayBuffer

class DQNAgent(object):
    '''
    Implementation of a reinforcement deep Q learning agent employing identical convolutional neural networks and updating weights between them.
    One is used for learning the best parameters and the second one for predicting traget values for solving the bellman equation.
    A Experience Replay buffer to sample from when learning is also used.
    '''
    def __init__(self, gamma, epsilon, learning_rate, num_actions, input_dim,
                 memory_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 update=1000, algorithm=None, env_name=None,
                 chkpt_dir='/models'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_actions = num_actions
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.update_target_count = update
        self.algorithm = algorithm
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(num_actions)] # for choosing actions
        self.learn_step_count = 0 # to schedule updating the network weights
        
        self.memory_buffer = ExperienceReplayBuffer(memory_size, input_dim, num_actions)

        self.q_net_eval = DeepQCNN(self.learning_rate, self.num_actions,
                                    input_dim=self.input_dim,
                                    name=self.algorithm+'_'+self.env_name+'_q_net_eval',
                                    checkpt_dir=self.chkpt_dir)

        self.q_net_next = DeepQCNN(self.learning_rate, self.num_actions,
                                    input_dim=self.input_dim,
                                    name=self.algorithm+'_'+self.env_name+'_q_net_next',
                                    checkpt_dir=self.chkpt_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon: # Greedy
            current_state = T.tensor([observation], dtype=T.float).to(self.q_net_eval.device) # Add extra dimension for batch size before sending tensor to the device
            actions = self.q_net_eval.forward(current_state)
            action = T.argmax(actions).item()
        else: # Exploratory
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory_buffer.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory_buffer.get_sample(self.batch_size)
        # Converting numpy arrays into pytorch tensors and sending them to device
        states = T.tensor(state).to(self.q_net_eval.device)        
        actions = T.tensor(action).to(self.q_net_eval.device)
        rewards = T.tensor(reward).to(self.q_net_eval.device)
        new_states = T.tensor(new_state).to(self.q_net_eval.device)        
        dones = T.tensor(done).to(self.q_net_eval.device)

        return states, actions, rewards, new_states, dones

    def save_network_checkpoints(self):
        print("Saving model checkpoints...")
        self.q_net_eval.save_checkpoint()
        self.q_net_next.save_checkpoint()

    def load_network_checkpoints(self):
        print("Loading model checkpoints...")
        self.q_net_eval.load_checkpoint()
        self.q_net_next.load_checkpoint()

    def epsilon_decrement(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def update_target_network(self):
        if self.learn_step_count % self.update_target_count == 0:
            self.q_net_next.load_state_dict(self.q_net_eval.state_dict()) # copy weights

    def learn_step(self):
        if self.memory_buffer.memory_count < self.batch_size:
            return

        self.q_net_eval.optimizer.zero_grad()
        self.update_target_network() # Get newest parameters into the target network

        states, actions, rewards, new_states, dones = self.sample_memory()
        
        # Array of numbers in range of 0 - 31. Used for indexing the actions from the q_net_eval ouput
        indices = np.arange(self.batch_size) 

        q_pred = self.q_net_eval.forward(states)[indices, actions]
        q_next = self.q_net_next.forward(new_states).max(dim=1)[0] 

        # calculating the  target value
        # Using the done flag for a mask to make conditions if the next state was terminal
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        # Network learning step
        loss = self.q_net_eval.loss(q_target, q_pred).to(self.q_net_eval.device)
        loss.backward()
        self.q_net_eval.optimizer.step()
        
        self.epsilon_decrement()
        self.learn_step_count += 1