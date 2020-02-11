import numpy as np
import matplotlib.pyplot as plt
import collections
import cv2
import gym

class FrameMaxRepeatAction(gym.Wrapper):
    '''
    Atari library renders certain objects only on even or odd frames.
    Thus this class takes the max of the two last frames.
    The parameter repeat(=4) is not the parameter for frame max. The frame max repeat is always 2.
    '''
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super(FrameMaxRepeatAction, self).__init__(env)
        self.repeat = repeat 
        self.imshape = env.observation_space.low.shape # 210x260x3 from base environment - low or high .shape return same 
        self.frame_buffer = np.zeros_like((2,self.imshape)) # Frame buffer - Shape 2 * obs_space_shape
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        '''
        Returns the same as base step method but the max frame of the two last frames.
        '''
        done = False
        total_reward = 0.0
        for i in range(self.repeat):
            observ, current_reward, done, info = self.env.step(action)
            if self.clip_reward:
                current_reward = np.clip(np.array([current_reward]),-1,1)[0] 
            total_reward += current_reward
            # saving the current observation in even or odd position
            pos = i % 2
            self.frame_buffer[pos] = observ
            if done:
                break
        maximum_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return maximum_frame, total_reward, done, info 

    def reset(self):
        observ = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset() 
        
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            observ, _, _, _ = self.env.step(1)

        self.frame_buffer = np.zeros_like((2,self.imshape))
        self.frame_buffer[0] = observ
        return observ

class FramePreprocessor(gym.ObservationWrapper):
    '''
    Class for preprocessing single images obtained by the environment. At this point the maxframe wrapper
    should already be applied on the environment.
    '''
    def __init__(self, imshape, env=None):
        super(FramePreprocessor, self).__init__(env)
        self.imshape = (imshape[2], imshape[0], imshape[1])# Reshaping from gym to pytorch require channel format
        
        # Setting the observation_space to a new custom shape. (Low and high are uint8 color values)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.imshape,
                                                dtype = np.float32)
        
    def observation(self, observ):
        '''
        Override observation function.
        '''
        screen = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(screen, self.imshape[1:], interpolation=cv2.INTER_AREA) 
        # convert the cv2 object to a np array and reshape it
        final_screen = np.array(resized_screen, dtype=np.uint8).reshape(self.imshape)
        final_screen = final_screen / 255.0
        return final_screen


class FrameStacker(gym.ObservationWrapper):
    '''
    Class for stacking images in the observation.
    '''
    def __init__(self, env, repeat):
        super(FrameStacker, self).__init__(env)        
        self.observation_space = gym.spaces.Box(
                                env.observation_space.low.repeat(repeat, axis=0),
                                env.observation_space.high.repeat(repeat, axis=0),
                                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat) # Empty frame stack

    def reset(self):
        self.stack.clear()
        observ = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observ)

        return np.array(self.stack).reshape(self.observation_space.low.shape) 

    def observation(self, observ):
        '''
        Overriding for returning the stacked images.
        '''
        self.stack.append(observ)
        return np.array(self.stack).reshape(self.observation_space.low.shape) # Convert deque to np array and reshape


def create_environment(env_name, imshape=(84,84,1), repeat=4, clip_rewards=False, no_ops=0,
                        fire_first=False):
    '''
    Function for creating the fully modified environment.
    All wrappers are applied.
    '''
    env = gym.make(env_name)
    env = FrameMaxRepeatAction(env, repeat, clip_rewards, no_ops, fire_first)
    env = FramePreprocessor(imshape, env)
    env = FrameStacker(env, repeat)
    return env

def plot_metrics(x, scores, epsilons, filename, lines=None):
    '''
    Function for plotting the training metrics and saving the plot to figure.
    '''
    print("Plotting metrics figure")
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C2")
    ax.set_xlabel("Training Steps", color="C2")
    ax.set_ylabel("Epsilon", color="C2")
    ax.tick_params(axis='x', colors="C2")
    ax.tick_params(axis='y', colors="C2")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C0")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C0")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C0")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)