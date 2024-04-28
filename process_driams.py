# Description: Process and group the DRIAMS-preprocessed data to get the binned data
import os, pickle, argparse
import os.path as osp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rich.progress import track
cur_dir = osp.dirname(osp.abspath(__file__))
os.chdir(cur_dir)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default='BCD', type=str, help='data type')
    args=parser.parse_args()
    return args

def get_binned_data(file_path, bins=np.arange(2000, 20001, 3)):
    with open(file_path, 'r') as handle:
        data_lines = [line.strip().split(' ') for line in handle.readlines()][3:]
        data = np.array(data_lines, dtype=float)
        data_binned = np.histogram(data[:, 0], bins=bins, weights=data[:, 1])[0]
    return data_binned, data

if __name__ == '__main__':
    args = get_args()
    # 1. process the data
    # for s in args.data:
    #     driams_root = '../DRIAMS/DRIAMS-{}/'.format(s)
    #     driams_label_path = '../DRIAMS/DRIAMS-{}/id/2018/2018_clean.csv'.format(s)
    #     driams_preprocessed_root = '../DRIAMS/DRIAMS-{}/preprocessed/2018/'.format(s)

    #     driams_table = pd.read_csv(driams_label_path)
    
    #     data_matrix = []
    #     species_list = []

    #     for i in track(range(len(driams_table)), description='Processing DRIAMS-{}'.format(s)):
    #     # for i in track(range(10)):
    #         filename = driams_table.loc[i, 'code']
    #         file_path = osp.join(driams_preprocessed_root, filename+'.txt')
    #         if not osp.isfile(file_path):
    #             continue
    #         species = driams_table.loc[i, 'species']
    #         species_list.append(species.lower())
    #         data_binned, data_file = get_binned_data(file_path)
    #         data_matrix.append(data_binned)
    #     data_matrix = np.array(data_matrix)
    #     print(data_matrix.shape)
    #     # save the data
    #     with open('../DRIAMS/DRIAMS-{}-data.pkl'.format(s), 'wb') as handle:
    #         pickle.dump((data_matrix, species_list), handle)

    # 2. combine the data
    print('Combining the data...')
    data_matrices = []
    species_lists = []
    for s in args.data:
        with open('../DRIAMS/DRIAMS-{}-data.pkl'.format(s), 'rb') as handle:
            data_matrix, species_list = pickle.load(handle)
            data_matrices.append(data_matrix) 
            species_lists.extend(species_list)
    
    unique_species_list = sorted(set(species_lists))
    sp_idx_dict = {sp: i for i, sp in enumerate(unique_species_list)}
    sp_idx_list = np.array([sp_idx_dict[sp] for sp in species_lists])[:, np.newaxis]
    print(sp_idx_list.shape)
    data_matrices = np.concatenate(data_matrices, axis=0)
    assert data_matrices.shape[0] == sp_idx_list.shape[0]
    data_matries_w_labels = np.concatenate([data_matrices, sp_idx_list], axis=1)
    print(data_matries_w_labels.shape)

    # 3. remove singletons
    print('Removing singletons...')
    counts = np.bincount(sp_idx_list.squeeze())
    singleton_labels = np.where(counts == 1)[0]
    non_singleton_indices = [i for i in range(len(data_matries_w_labels)) if data_matries_w_labels[i, -1] not in singleton_labels]
    data_non_singleton = data_matries_w_labels[non_singleton_indices]
    print(data_non_singleton.shape)
    np.save('../DRIAMS/DRIAMS-{}-preprocessed-nonsingleton.npy'.format(args.data), data_non_singleton, allow_pickle=True)

        
