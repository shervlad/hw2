import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

class CategoricalMLP(torch.nn.Module):
    def __init__(self,dimensions,activation = nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        layers = []

        for j in range(len(dimensions)-1):
            act = activation if j < len(dimensions)-2 else output_activation
            layers += [nn.Linear(dimensions[j], dimensions[j+1]), act()]

        self.perceptron =  nn.Sequential(*layers)

    def forward(self, state):
        return torch.squeeze(self.perceptron(state),-1)

class ForwardModel():
    def __init__(self):

        # load forward model
        input_dims = (6,)
        hidden_dims = (64,64,64)
        output_dims = (2,)
        self.forward_model = CategoricalMLP(input_dims + hidden_dims + output_dims)
        self.forward_model.load_state_dict(torch.load("./models/forward_model.pt"))
        self.forward_model.eval()

        # push config: push_len by default [0.06-0.1]
        self.push_len_min = 0.06 # 0.06 ensures no contact with box empiracally
        self.push_len_range = 0.04
        # task space x ~ (0.4, 0.8); y ~ (-0.3, 0.3)
        self.max_arm_reach = 0.91
        self.workspace_max_x = 0.75 # 0.8 discouraged, as box can move past max arm reach
        self.workspace_min_x = 0.4
        self.workspace_max_y = 0.3
        self.workspace_min_y = -0.3

    def sample_push(self, obj_x, obj_y, push_ang, push_len):
        # calc starting push location and ending push loaction
        start_x = obj_x - self.push_len_min * np.cos(push_ang)
        start_y = obj_y - self.push_len_min * np.sin(push_ang)
        end_x = obj_x + push_len * np.cos(push_ang)
        end_y = obj_y + push_len * np.sin(push_ang)
        start_radius = np.sqrt(start_x**2 + start_y**2)
        end_radius = np.sqrt(end_x**2 + end_y**2)
        # find valid push that does not lock the arm
        if start_radius < self.max_arm_reach \
            and end_radius + self.push_len_min < self.max_arm_reach \
            and end_x > self.workspace_min_x and end_x < self.workspace_max_x \
            and end_y > self.workspace_min_y and end_y < self.workspace_max_y:
            # find push that does not push obj out of workspace (camera view)
            return start_x, start_y, end_x, end_y
        else:
            return None

    def sample_error(self,ang_len,obj, obj2):
        obj = obj.squeeze().detach().numpy()
        push =  self.sample_push(obj[0],obj[1],ang_len[0],ang_len[1])
        if push is None:
            return 100
        inp = torch.as_tensor(np.concatenate((obj,push)),dtype=torch.float32)
        error =  np.linalg.norm(self.forward_model(inp).detach() - obj2)
        return error

    def infer(self,init_obj,goal_obj):
        init_obj = torch.FloatTensor(init_obj).unsqueeze(0)
        goal_obj = torch.FloatTensor(goal_obj).unsqueeze(0)

        #Cross Entropy Method to find optimal action
        mean = np.array([0.0,self.push_len_min + self.push_len_range/2.0])
        st_dev = np.array([1,0.1])

        mean_error = 1000
        while np.all(st_dev>0.0001):
            distribution = Normal(torch.as_tensor(mean),torch.as_tensor(st_dev))
            samples = distribution.sample((200,)).numpy()
            samples = sorted(samples, key=lambda x: self.sample_error(x,init_obj,goal_obj))
            mean_error = np.mean([self.sample_error(x,init_obj,goal_obj) for x in samples])
            mean = np.mean(samples[:20],axis=0)
            st_dev = np.std(samples[:20],axis=0)

        push_ang,push_len = distribution.mean
        init_obj_x,init_obj_y = init_obj.squeeze().detach().tolist()
        return self.sample_push(init_obj_x, init_obj_y,push_ang,push_len)

class InverseModel():
    def __init__(self):
        # load inverse model
        input_dims = (4,)
        hidden_dims = (64,64,64)
        output_dims = (4,)
        self.inverse_model = CategoricalMLP(input_dims + hidden_dims + output_dims)
        self.inverse_model.load_state_dict(torch.load("./models/inverse_model.pt"))
        self.inverse_model.eval()

    def infer(self,init_obj,goal_obj):
        init_obj = torch.FloatTensor(init_obj).unsqueeze(0)
        goal_obj = torch.FloatTensor(goal_obj).unsqueeze(0)
        obs = torch.cat((init_obj,goal_obj),1).squeeze()
        push = self.inverse_model(obs)
        return push.detach().numpy()
