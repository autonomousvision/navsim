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


        all_trajs = []

        state = state.unsqueeze(1)
        for batch in range(B):
            state_seq = state[batch].unsqueeze(1) #N, T, C
            action_seq = torch.zeros((1,1,2)) #N, T, C

            for t in range(self.timestep):
                N, T, C = state_seq.shape
                _, _, Ca = action_seq.shape
                state_seq = state_seq.unsqueeze(1).repeat(1, Na, 1, 1) #N, NA, T, C
                action_seq = action_seq.unsqueeze(1).repeat(1, Na, 1, 1) #N, NA, T, C
                
                next_state = state_seq[:,:,-1,:].clone() # N, NA, C
                next_state = self.step(next_state, action) # N, NA, C


                state_seq = state_seq.reshape(-1, t+1, C) # N*NA, T, C
                # s = s.reshape(-1, t+1, C) # N*NA, T, C
                next_state = next_state.unsqueeze(2).reshape(-1, 1, C)

                action_seq = torch.cat([action_seq, action.repeat(N,1,1).unsqueeze(2)], 2).reshape(-1,t+2,Ca)

                # check bounds
                x_idx = ((next_state[:, 0, 0] - self.x_grid[0]) / (self.x_grid[1] - self.x_grid[0])).long()
                y_idx = ((next_state[:, 0, 1] - self.y_grid[0]) / (self.y_grid[1] - self.y_grid[0])).long()

                # clamp to valid range
                x_idx = torch.clamp(x_idx, 0, drivable_area.shape[1] - 1)
                y_idx = torch.clamp(y_idx, 0, drivable_area.shape[0] - 1)

                # check if in drivable area
                valid = drivable_area[y_idx, x_idx]
                valid = torch.logical_and(valid, (torch.abs(action_seq.diff(1,axis=1)[:,:,0])>(0.2*np.pi/2)).sum(1)==0)
                valid = torch.logical_and(valid, (torch.abs(action_seq.diff(2,axis=1)[:,:,1])>(0.5*9.81)).sum(1)==0)
                stop_idx = torch.logical_or(torch.abs(action_seq[:,-1,0])<1e-8, state_seq[:,-1,3]>1e-8)
                valid = torch.logical_and(valid, stop_idx)
                valid = torch.logical_and(valid, next_state[:,-1,3]<50)
                mask = valid.bool()

                # update only valid state

                state_seq = torch.cat([state_seq[mask], next_state[mask]],1)
                action_seq = action_seq[mask]
                print(state_seq.shape)
            # concatenate all timesteps for this batch
            # trajs = torch.stack(trajs, dim=0)  # (T, valid_N, 4)
            all_trajs.append(state_seq.clone())

        # pad and stack if needed to same shape
        return all_trajs  # list of (T, N', 4), one per batch

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
    action1 = torch.linspace(-0.3*np.pi/2, 0.3*np.pi/2, 5)
    action2 = torch.linspace(0.1, 0.3, 2)*9.81   
    
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
    drivable_area[:,47:53]=1

    trajectories = motion_model.predict(ego_status, single_step_actions, drivable_area)


    plt.figure(2)
    plt.plot(trajectories[0][:,:,0].permute(1,0),trajectories[0][:,:,1].permute(1,0))
    # print(action[0], ret[0,0,:,0],ret[0,0,:,1])
    plt.show()