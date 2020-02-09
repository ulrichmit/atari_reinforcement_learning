import numpy as np

class ExperienceReplayBuffer(object):
    '''
    Experience replay buffer which accepst flexible image sizes.
    Stores: states, actions, rewards, new states, done
    
    Should not repeat
    Sample uniformly
    '''
    def __init__(self, max_size, input_shape, num_actions):
        self.memory_size = max_size
        self.memory_count = 0 # Counting the entries

        # Custom memory types with np arrays
        self.state_memory = np.zeros((self.memory_size, *input_shape),
                                        dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape),
                                        dtype=np.float32)
        # int64 due to pytorch compatability
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.uint8) # bool pytroch?

    def store_transition(self, state, action, reward ,new_state, done):
        '''
        Storing the given transition information.
        If the counter is greater then memory size, storing will start again at the beginning
        and overwrite old entries.
        '''
        # Position of the first unocuppied position
        ind = self.memory_count % self.memory_size 
        self.state_memory[ind] = state
        self.new_state_memory[ind] = new_state
        self.action_memory[ind] = action
        self.reward_memory[ind] = reward
        self.terminal_memory[ind] = done
        self.memory_count += 1

    def get_sample(self, batch_size):
        '''
        Sampling uniformly from the buffer storage.
        '''
        upper_bound = min(self.memory_count, self.memory_size) # setting upper bound
        # replace=False --> unique sampling
        batch = np.random.choice(upper_bound, batch_size, replace=False) 

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]
        
        return states, actions, rewards, new_states, terminals
