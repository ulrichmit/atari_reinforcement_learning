import numpy as np
import matplotlib.pyplot as plt
import collections
import cv2
import gym


def create_environment(env_name, imshape=(84, 84, 1), repeat=4):
    # Stacking the changes on the environment
    env = gym.make(env_name)
    env = FrameMax(env, repeat)
    env = FramePreprocessor(imshape, env)
    env = FrameStacker(env, repeat)
    return env


class FrameMax(gym.Wrapper):
    def __init__(self, env=None, repeat=4):
        super(FrameMax, self).__init__(env)
        self.repeat = repeat
        self.imshape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.imshape))

    def step(self, action):
        done = False
        total_reward = 0.0
        for i in range(self.repeat):
            observ, current_reward, done, info = self.env.step(action)
            total_reward += current_reward
            # saving the current observation in even or odd position
            pos = i % 2
            self.frame_buffer[pos] = observ
            if done:
                break
        maximum_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return total_reward, done, info, maximum_frame

    def reset(self):
        observ = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.imshape))
        self.frame_buffer[0] = observ
        return observ


class FramePreprocessor(gym.ObservationWrapper):
    def __init__(self, imshape, env=None):
        super(FramePreprocessor, self).__init__(env)
        # fix gym/pytorch channels difference
        self.imshape = (imshape[2], imshape[0], imshape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.imshape,
                                                dtype = np.float32)
        
    def observation(self, observ):
        screen = cv2.cvtColor(observ, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(screen, self.imshape[1:], interpolation=cv2.INTER_AREA) 
        # convert to np array and changin axis
        final_screen = np.array(resized_screen, dtype=np.uint8).reshape(self.imshape)
        final_screen = final_screen / 255.0
        return final_screen



class FrameStacker(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(FrameStacker, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                env.observation_space.low.repeat(repeat, axis=0),
                                env.observation_space.high.repeat(repeat, axis=0),
                                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        observ = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observ)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observ):
        self.stack.append(observ)
        return np.array(self.stack).reshape(self.observation_space.low.shape)
                                            