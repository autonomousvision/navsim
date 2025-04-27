import numpy as np
import torch 


class PointMassModel():
    def __init__ (self, dt, timestep):
        self.dt = dt
        self.timestep = timestep


    def predict(self, state, action):
        '''
        state # x, y, heading, v0
        action # dh, a
        '''
        state = state.unsqueeze(1).repeat(1, action.shape[0], 1)
        
        ret = []
        for i in range(self.timestep):
            ret.append(self.step(state, action[:,i,:]).clone())
        ret = torch.stack(ret)
        ret = ret.permute(1,2,0,3)

        plt.figure(1)
        plt.plot(ret[0,:,:,0].permute(1,0),ret[0,:,:,1].permute(1,0))
        # print(action[0], ret[0,0,:,0],ret[0,0,:,1])
        plt.show()
        return ret
    
    def step(self, state, action):
        state[:,:,0]+=state[:,:,3]*torch.cos(state[:,:,2])*self.dt
        state[:,:,1]+=state[:,:,3]*torch.sin(state[:,:,2])*self.dt
        state[:,:,2]+=action[:,0]
        state[:,:,3]+=action[:,1]
        return state




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import itertools
    motion_model = PointMassModel(0.5, 8)
    ego_status = np.zeros((2,4))
    ego_status[1,2:] = 1
    ego_status = torch.tensor(ego_status)
    action1 = torch.linspace(-0.3*np.pi/2, 0.3*np.pi/2, 3)
    action2 = torch.linspace(0.1, 0.5, 2)*9.81    
    # all possible single-timestep action pairs (steering, throttle)
    single_step_actions = torch.cartesian_prod(action1, action2)  # (6, 2)

    # number of timesteps
    T = 8  # you can change this

    # create all combinations over T timesteps
    all_combinations = list(itertools.product(range(single_step_actions.size(0)), repeat=T))  # (6^T,)

    # build the full trajectory tensor
    actions = torch.stack([
        single_step_actions[list(indices)] for indices in all_combinations
    ])  # shape: (num_combinations, T, 2)
    print(actions.shape)
    actions = actions[(torch.abs(actions.diff(1, axis=1)[:,:,0])>(0.3*np.pi/2)).sum(1)==0]
    print(actions.shape)
    actions = actions[(torch.abs(actions.diff(2, axis=1)[:,:,1])>(0.5*9.81)).sum(1)==0]
    print(actions.shape)

    trajectories = motion_model.predict(ego_status, actions)

    # from sklearn.cluster import KMeans

    # K = 8192  # 만들고 싶은 vocabulary 크기
    # kmeans = KMeans(n_clusters=K, random_state=0)
    # flat_positions = trajectories[0,:,:,[0,1]].reshape(-1,16)
    # kmeans.fit(flat_positions)

    # # cluster centers: (K, T*2)
    # centers_flat = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    # centers = centers_flat.reshape(K, -1, 2)  # shape: (K, T, 2)


    # plt.figure(2)
    # plt.plot(centers[:,:,0].permute(1,0),centers[:,:,1].permute(1,0))
    # # print(action[0], ret[0,0,:,0],ret[0,0,:,1])
    # plt.show()