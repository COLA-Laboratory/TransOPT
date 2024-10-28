import random
import sobol_seq
import numpy as np
from sklearn.cluster import KMeans




def InitData(Init_method, KB, Init, Xdim, Dty, **kwargs):

    type = Init_method.split('_')[0]
    method = Init_method.split('_')[1]
    if type=='Continuous':
        if method == 'random':
            train_x = 2 * np.random.random(size=(Init, Xdim)) - 1
        elif method == 'uniform':
            train_x = 2 * sobol_seq.i4_sobol_generate(Xdim, Init) - 1
        elif method == 'fix':
            if KB.len == 0:
                train_x = np.array([[-0.5],[-0.25],[0.5],[0.42]])
            else:
                train_x = np.array([[-0.1], [-0.8], [0.25], [0.4]])
        elif method == 'LFL':
            seed = kwargs['seed']
            quantile = kwargs['quantile']
            try:
                train_x = np.loadtxt(f'./Bench/Lifelone_env/randIni/ini_{Xdim}d_{Init}p_{seed}.txt')
                if len(train_x.shape) == 1:
                    train_x = train_x[:,np.newaxis]
            except:
                train_x = 2 * np.random.random(size=(Init, Xdim)) - 1
                np.savetxt(f'./Bench/Lifelone_env/randIni/ini_{Xdim}d_{Init}p_{seed}.txt', train_x)
            anchor_point_num = int(quantile * Init)
            temp_x = train_x[:anchor_point_num]
            random_x = 2 * np.random.random(size=(100*Xdim, Xdim)) - 1
            train_x = np.vstack((temp_x, random_x[-(Init-anchor_point_num):]))
        idxs = None
    elif type=='Tabular':
        if method == 'random':
            if 'Env' in kwargs.keys():
                data_num = kwargs['Env'].get_dataset_size()
                rand_idxs = random.sample(range(0, data_num), Init)
                train_x = kwargs['Env'].get_var(rand_idxs)
                idxs = rand_idxs
        # elif Method == 'grid':
        #     if KB.len == 0:
        #         if np.float64 == Dty:
        #             train_x = 2 * np.random.random(size=(Init, Xdim)) - 1
        #         else:
        #             print('Unsupport data type! shut down')
        #             return
        #     else:
        #         train_x = KB.local_optimal[0]
        #         for i in range(1, KB.len):
        #             train_x = np.vstack((train_x, KB.local_optimal[i]))
        #         train_x = np.unique(train_x, axis=0)
        #
        #         if len(train_x) == Init:
        #             pass
        #             # train_x = np.array(train_x, dtype=Dty)
        #         elif len(train_x) > Init:
        #             result_x = []
        #             kmn = KMeans(n_clusters=int(Init), random_state=0)
        #             kmn.fit(train_x)
        #             lables = kmn.labels_
        #             centers = kmn.cluster_centers_
        #             for c_id,center in enumerate(centers):
        #                 min_dis = 100
        #                 min_dis_x_id = 0
        #                 for x_id, x in enumerate(train_x):
        #                     if lables[x_id] == c_id:
        #                         dis = np.linalg.norm(x - center)
        #                         if dis < min_dis:
        #                             min_dis = dis
        #                             min_dis_x_id = x_id
        #                 result_x.append(train_x[min_dis_x_id])
        #
        #             train_x = np.array(result_x)
        #             # train_x = np.concatenate(
        #             #     (train_x, 2 * np.random.random(size=(Init - len(train_x), Xdim)) - 1))
        #         else:
        #             # train_x = np.array(train_x, dtype=Dty)
        #             train_x = np.concatenate(
        #                 (train_x, 2 * np.random.random(size=(Init - len(train_x), Xdim)) - 1))
    else:
        raise ValueError

    return train_x, idxs
