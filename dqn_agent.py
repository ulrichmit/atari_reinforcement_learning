import torch as T
import numpy as np
from dqn_network import DeepQCNN
from experience_buffer import ExperienceReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, learning_rate, num_actions, input_dim,
                 memory_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 update=1000, algorithm=None, env_name=None,
                 chkpt_dir='/plots'):
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
        self.action_space = [i for i in range(num_actions)] # for choosing
        self.learn_step_count = 0 # to schedule updating the network weights

        # Instatinating the memory and CNN's
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
        # check greedy or exploratory
        # random?????
        if np.random.random() > self.epsilon:
            current_state = T.tensor([observation], dtype=T.float).to(self.q_net_eval.device) # Add extra dim for batch size. Sending tensor to the device
            actions = self.q_net_eval.forward(current_state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory_buffer.store_transition(state, action, reward, new_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory_buffer.get_sample(self.batch_size)
        # Converting numpy arrays into pytroch tensors and sending them to device
        states = T.tensor(state).to(self.q_net_eval.device)        
        actions = T.tensor(action).to(self.q_net_eval.device)
        rewards = T.tensor(reward).to(self.q_net_eval.device)
        new_states = T.tensor(new_state).to(self.q_net_eval.device)        
        dones = T.tensor(done).to(self.q_net_eval.device)

        return states, rewards, dones, actions, new_states

    def save_network_checkpoints(self):
        self.q_net_eval.save_checkpoint()
        self.q_net_next.save_checkpoint()

    def load_network_checkpoints(self):
        self.q_net_eval.load_checkpoint()
        self.q_net_next.load_checkpoint()

    def epsilon_decrement(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def update_target_network(self):
        if self.learn_step_count % self.update_target_count == 0:
            self.q_net_next.load_state_dict(self.q_net_eval.state_dict()) # copy weights

    def learn_step(self):
        # Wait until the number of samples in the buffer is bigger than bs
        if self.learn_step_count < self.batch_size:
            return
        
        self.update_target_network() # To have the newset parameters in the target network
        self.q_net_eval.optimizer.zero_grad()

        # sample from the memory
        states, actions, rewards, new_states, dones = self.sample_memory()

        # Calculation of the Q_pred and q_target (q_next) valuess
        # We don't just want to knwo the action values for the batch of states
        # We want the value value of the action the agent took in those states. Just the one value! Thus we have to index it correct out from the nn output
        
        # used for indexing the actions from the nn ouput
        # Array of numbers in ragne of 0 - 31
        indices = np.arange(self.batch_size) 
        # Getting just an array of shape batch_size with the action values taken
        q_pred = self.q_net_eval.forward(states)[indices, actions]

        # Getting the values for the maximum actions for the set of new_states
        q_next = self.q_net_next.forward(new_states).max(dim=1)[0] # 0 for values

        # calculating the  target value
        # using the done flag for a mask to make condition if the next state was terminal and thus the q_next should be 0. Thus the target value should just be equal to rewards
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        # Network learning step
        loss = self.q_net_eval.loss(q_target, q_pred).to(self.q_net_eval.device)
        loss.backward()
        self.q_net_eval.optimizer.step()
        
        self.epsilon_decrement()
        self.learn_step_count += 1