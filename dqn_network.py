import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer
import os

class DeepQCNN(nn.Module):
    '''
    Convolutional neural network for a deep Q learning agent.
    Accepts different image sizes for generic use.
    '''
    def __init__(self, learning_rate, num_actions, name, input_dim, checkpt_dir):
        super(DeepQCNN, self).__init__()
        self.checkpoint_dir = checkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_dim[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        
        # Due to the generic image-shape 
        # --> The input shape of the first fully connected layer has to be determined        
        fc_input_dim = self.calc_fc_input_dim(input_dim)
        
        # FullyConnected layers
        self.fc1 = nn.Linear(fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        # Optimizer
        self.optimizer = optimizer.RMSprop(self.parameters(), lr=learning_rate)

        # Loss function
        self.loss = nn.MSELoss()

        # Determine cuda device and assign tensors accordingly
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def calc_fc_input_dim(self, input_dim):
        '''
        Method used for determining the input shape of the first fully connected layer
        '''
        dummy_state = T.zeros(1, *input_dim) # batch_size=1
        dim = self.conv1(dummy_state)
        dim = self.conv2(dim)
        dim = self.conv3(dim)        
        return int(np.prod(dim.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        
        # Reshaping/flattening conv3 data for fc1 layer
        # Shape conv3: batch_size * num_filters * H * W (H*W of the convolved image)
        conv_reshaped = conv3.view(conv3.size()[0], -1) # view works similar to np.reshape
        # Shape conv_reshaped: batch_size * (num_filters * H * W) - Flattened tensor

        fc1 = F.relu(self.fc1(conv_reshaped))
        actions = self.fc2(fc1)
        return actions

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):        
        self.load_state_dict(T.load(self.checkpoint_file))
