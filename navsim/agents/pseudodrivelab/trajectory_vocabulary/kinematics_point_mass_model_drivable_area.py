import numpy as np
import torch 

import torch.nn.functional as F

class PointMassModel():
    def __init__ (self, dt, timestep, x_grid, y_grid):
        self.dt = dt
        self.timestep = timestep
        self.x_grid = x_grid
        self.y_grid = y_grid

    def predict(self, state, action, drivable_area):
        '''
        state: (B, 4) -> x, y, heading, v0
        action: (N, 2) -> dh, a
        drivable_area: (H, W) binary mask
        '''
        B,_ = state.shape
        Na, _ = action.shape

        ret = []
        state = state.unsqueeze(1)
        for batch in range(B):
            s = state[batch]
            for i in range(self.timestep):
                N,C = s.shape
                s = s.unsqueeze(1).repeat(1, Na, 1)
                mask = torch.ones((Na*N,), dtype=torch.int64, device=state.device)
                next_state = self.step(s, action).reshape(-1,C)

                # check bounds
                x_idx = ((next_state[:, 0] - self.x_grid[0]) / (self.x_grid[1] - self.x_grid[0])).long()
                y_idx = ((next_state[:, 1] - self.y_grid[0]) / (self.y_grid[1] - self.y_grid[0])).long()
                
                # clamp to valid range
                x_idx = torch.clamp(x_idx, 0, drivable_area.shape[1] - 1)
                y_idx = torch.clamp(y_idx, 0, drivable_area.shape[0] - 1)

                # check if in drivable area
                valid = drivable_area[y_idx, x_idx]
                mask &= valid

                # mask out invalid predictions (optional: freeze at previous position)
                next_state = next_state[mask>0]
                s = next_state
            ret.append(next_state.clone())
            print(next_state.shape)
            plt.scatter(s[:1000,0],s[:1000,1])
            plt.show()



            # ret = torch.stack(ret, dim=0)  # (T, B, T, 4)
            # ret = ret.permute(1, 2, 0, 3)  # (B, T, T, 4)
        return ret

    def step(self, state, action):
        next_state = state.clone()
        next_state[:, :, 0] += next_state[:, :, 3] * torch.cos(next_state[:, :, 2]) * self.dt
        next_state[:, :, 1] += next_state[:, :, 3] * torch.sin(next_state[:, :, 2]) * self.dt
        next_state[:, :, 2] += action[:, 0].unsqueeze(0)
        next_state[:, :, 3] += action[:, 1].unsqueeze(0)
        return next_state

# class PointMassModel():
#     def __init__ (self, dt, timestep, x_grid, y_grid):
#         self.dt = dt
#         self.timestep = timestep

#         self.x_grid = x_grid
#         self.y_grid = y_grid
        


#     def predict(self, state, action, drivable_area):
#         '''
#         state # x, y, heading, v0
#         action # dh, a
#         '''
#         state = state.unsqueeze(1).repeat(1, action.shape[0], 1)
        
#         ret = []
#         for i in range(self.timestep):
#             ret.append(self.step(state, action[:,i,:], drivable_area).clone())
#         ret = torch.stack(ret)
#         ret = ret.permute(1,2,0,3)

#         # plt.figure(1)
#         # plt.plot(ret[0,:,:,0].permute(1,0),ret[0,:,:,1].permute(1,0))
#         # # print(action[0], ret[0,0,:,0],ret[0,0,:,1])
#         # plt.show()
#         return ret
    
#     def step(self, state, action, drivable_area):
#         state[:,:,0]+=state[:,:,3]*torch.cos(state[:,:,2])*self.dt
#         state[:,:,1]+=state[:,:,3]*torch.sin(state[:,:,2])*self.dt
#         state[:,:,2]+=action[:,0]
#         state[:,:,3]+=action[:,1]
#         return state




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import itertools

    x_grid = torch.linspace(-40,40,81)
    y_grid = torch.linspace(-40,40,81)

    motion_model = PointMassModel(0.5, 8, x_grid, y_grid)
    ego_status = torch.zeros((1,4))
    # ego_status[1,2:] = 1
    ego_status = torch.tensor(ego_status)
    action1 = torch.linspace(-0.1*np.pi/2, 0.1*np.pi/2, 3)
    action2 = torch.linspace(0.2, 1, 2)
    
    # all possible single-timestep action pairs (steering, throttle)
    single_step_actions = torch.cartesian_prod(action1, action2)  # (6, 2)

    # number of timesteps
    T = 8  # you can change this

    # # create all combinations over T timesteps
    # all_combinations = list(itertools.product(range(single_step_actions.size(0)), repeat=T))  # (6^T,)

    # # build the full trajectory tensor
    # actions = torch.stack([
    #     single_step_actions[list(indices)] for indices in all_combinations
    # ])  # shape: (num_combinations, T, 2)
    
    # actions = actions[np.logical_and(torch.abs(actions.diff(1, axis=1)[:,:,1]).sum(1)<2,torch.abs(actions.diff(1, axis=1)[:,:,0]).sum(1)<0.6)]

    drivable_area = torch.zeros((81,81), dtype=torch.int64)

    drivable_area[38:47,:]=1
    drivable_area[:,38:47]=1

    trajectories = motion_model.predict(ego_status, single_step_actions, drivable_area)


    plt.figure(2)
    plt.plot(trajectories[1,:,:,0].permute(1,0),trajectories[1, :,:,1].permute(1,0))
    # print(action[0], ret[0,0,:,0],ret[0,0,:,1])
    plt.show()