import sys
sys.path.append('../')
import pyedflib
import EvoContext.scripts.utils1 as utils1
from EvoContext.scripts.utils1 import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch
import math
import h5py
import numpy as np
import os
import pickle
import scipy
import scipy.signal
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import defaultdict
from .constants import INCLUDED_CHANNELS 
from .data_utils import comp_xcorr, keep_topk, computeFFT
import EvoContext.scripts.utils1 as utils1
repo_paths = str(Path.cwd()).split('STAGSMOTE')
repo_paths = Path(repo_paths[0]).joinpath('STAGSMOTE')
sys.path.append(repo_paths)
FREQUENCY = 256


def computeSliceMatrix(
        h5_fn,
        clip_idx,
        time_step_size=1,
        clip_len=12,
        is_fft=False):
    """
    Comvert entire EEG sequence into clips of length clip_len
    Args:
        h5_fn: file name of resampled signal h5 file (full path)
        clip_idx: index of current clip/sliding window
        time_step_size: length of each time_step_size, in seconds, int
        clip_len: sliding window size or EEG clip length, in seconds, int
        is_fft: whether to perform FFT on raw EEG data
    Returns:
        slices: list of EEG clips, each having shape (clip_len*freq, num_channels, time_step_size*freq)
        seizure_labels: list of seizure labels for each clip, 1 for seizure, 0 for no seizure
    """
    with h5py.File(h5_fn, 'r') as f:
        try:
            signal_array = f["filtered_signal"][()]
        except:
            raise ValueError(f"No clip found in {h5_fn}")

    physical_clip_len = int(FREQUENCY * clip_len)
    physical_time_step_size = int(FREQUENCY * time_step_size)

    start_window = clip_idx * physical_clip_len
    end_window = start_window + physical_clip_len
    curr_slc = signal_array[:, start_window:end_window]

    start_time_step = 0
    time_steps = []
    while start_time_step <= curr_slc.shape[1] - physical_time_step_size:
        end_time_step = start_time_step + physical_time_step_size
        # (num_channels, physical_time_step_size)
        curr_time_step = curr_slc[:, start_time_step:end_time_step]
        if is_fft:
            curr_time_step, _ = computeFFT(
                curr_time_step, n=physical_time_step_size)

        time_steps.append(curr_time_step)
        start_time_step = end_time_step
    # if len(time_steps)
    # print(f"time_steps -: {len(time_steps)}")

    eeg_clip = np.stack(time_steps, axis=0)
    return eeg_clip

def parseTxtFiles(final_label_dir, split_type, task, clip_len, cv_seed=123):
    label_file_path = os.path.join(final_label_dir, split_type + '_' + task + '_' + str(clip_len) + 's.txt')
    with open(label_file_path, 'r') as f:
        lines = f.readlines()
    
    file_tuples = []
    for line in lines:
        h5_fn, segment_idx, label = line.strip().split(',')
        file_tuples.append((h5_fn, int(segment_idx), int(label)))

    print(f"path: {label_file_path}")
    print(f"Number of clips in {split_type}: {len(file_tuples)}")
    print(f"Number of seizure clips: {len([tup for tup in file_tuples if tup[2] == 1])}")
    print(f"Number of non-seizure clips: {len([tup for tup in file_tuples if tup[2] == 0])}")
    
    file_tuples_by_label = defaultdict(list)
    for h5_fn, segment_idx, label in file_tuples:
        file_tuples_by_label[label].append((h5_fn, segment_idx, label))

    if split_type == "train": 
    
        min_count = min(len(tuples) for tuples in file_tuples_by_label.values())
        
        balanced_file_tuples = []
        for label, tuples in file_tuples_by_label.items():
            balanced_file_tuples.extend(tuples[:min_count])  
        if cv_seed is not None:
            random.seed(cv_seed)
        random.shuffle(balanced_file_tuples)


        return balanced_file_tuples

    else : 
        return file_tuples


class SeizureDataset(Dataset):
    def __init__(
            self,
            task, 
            input_dir,
            final_label_dir,
            cached_graphs_dir,
            logger,
            time_step_size=1,
            max_seq_len=12,
            standardize=True,
            scaler=None,
            split='train',
            data_augment=False,
            adj_mat_dir=None,
            graph_type=None,
            top_k=None,
            filter_type='laplacian',
            sampling_ratio=1,
            seed=123,
            use_fft=False,
            preproc_dir=None):
        """
        Args:
            input_dir: dir to resampled signals h5 files
            raw_data_dir: dir to TUSZ edf files
            time_step_size: int, in seconds
            max_seq_len: int, eeg clip length, in seconds
            standardize: if True, will z-normalize wrt train set
            scaler: scaler object for standardization
            split: train, dev or test
            data_augment: if True, perform random augmentation on EEG
            adj_mat_dir: dir to pre-computed distance graph adjacency matrix
            graph_type: 'combined' (i.e. distance graph) or 'individual' (correlation graph)
            top_k: int, top-k neighbors of each node to keep. For correlation graph only
            filter_type: 'laplacian' for distance graph, 'dual_random_walk' for correlation graph
            sampling_ratio: ratio of positive to negative examples for undersampling
            seed: random seed for undersampling
            use_fft: whether perform Fourier transform
            preproc_dir: dir to preprocessed Fourier transformed data, optional 
        """
        if standardize and (scaler is None):
            raise ValueError('To standardize, please provide scaler.')
        if (graph_type == 'individual') and (top_k is None):
            raise ValueError('Please specify top_k for individual graph.')

        self.task = task
        self.input_dir = input_dir
        self.time_step_size = time_step_size
        self.max_seq_len = max_seq_len
        self.standardize = standardize
        self.scaler = scaler
        self.split = split
        self.data_augment = data_augment
        self.adj_mat_dir = adj_mat_dir
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type
        self.use_fft = use_fft
        self.preproc_dir = preproc_dir
        self.logger = logger
        self.cached_graphs_dir = cached_graphs_dir

        self.file_tuples = parseTxtFiles(final_label_dir, split, task, max_seq_len)
        self.size = len(self.file_tuples)


        self._targets = [t[2] for t in self.file_tuples]

        # Check if the first file exists
        first_file = os.path.join(self.input_dir, self.file_tuples[0][0])
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"File not found: {first_file}")

    def __len__(self):
        return self.size

    def targets(self):
        return self._targets

    def _random_scale(self, EEG_seq):
        """
        Scale EEG signals by a random number between 0.8 and 1.2
        """
        scale_factor = np.random.uniform(0.8, 1.2)
        if self.use_fft:
            EEG_seq += np.log(scale_factor)
        else:
            EEG_seq *= scale_factor
        return EEG_seq

    def _get_indiv_graphs(self, eeg_clip, swap_nodes=None):
        """
        Compute adjacency matrix for correlation graph
        Args:
            eeg_clip: shape (seq_len, num_nodes, input_dim)
            swap_nodes: list of swapped node index
        Returns:
            adj_mat: adjacency matrix, shape (num_nodes, num_nodes)
        """
        num_sensors = 22
        adj_mat = np.eye(num_sensors, num_sensors,
                         dtype=np.float32)  

        eeg_clip = np.transpose(eeg_clip, (1, 0, 2))
        # print(f"eeg_clip shape -: {eeg_clip.shape}")
        assert eeg_clip.shape[0] == num_sensors

        eeg_clip = eeg_clip.reshape((num_sensors, -1))

        for i in range(0, num_sensors):
            for j in range(i + 1, num_sensors):
                xcorr = comp_xcorr(
                    eeg_clip[i, :], eeg_clip[j, :], mode='valid', normalize=True)
                adj_mat[i, j] = xcorr
                adj_mat[j, i] = xcorr

        adj_mat = abs(adj_mat)

        if (self.top_k is not None):
            adj_mat = keep_topk(adj_mat, top_k=self.top_k, directed=True)
        else:
            raise ValueError('Invalid top_k value!')

        return adj_mat

    def _get_combined_graph(self, swap_nodes=None):
        """
        Get adjacency matrix for pre-computed distance graph
        Returns:
            adj_mat_new: adjacency matrix, shape (num_nodes, num_nodes)
        """
        with open(self.adj_mat_dir, 'rb') as pf:
            adj_mat = pickle.load(pf)
            adj_mat = adj_mat[-1]

        adj_mat_new = adj_mat.copy()
        if swap_nodes is not None:
            for node_pair in swap_nodes:
                for i in range(adj_mat.shape[0]):
                    adj_mat_new[node_pair[0], i] = adj_mat[node_pair[1], i]
                    adj_mat_new[node_pair[1], i] = adj_mat[node_pair[0], i]
                    adj_mat_new[i, node_pair[0]] = adj_mat[i, node_pair[1]]
                    adj_mat_new[i, node_pair[1]] = adj_mat[i, node_pair[0]]
                    adj_mat_new[i, i] = 1
                adj_mat_new[node_pair[0], node_pair[1]
                            ] = adj_mat[node_pair[1], node_pair[0]]
                adj_mat_new[node_pair[1], node_pair[0]
                            ] = adj_mat[node_pair[0], node_pair[1]]

        return adj_mat_new

    def _compute_supports(self, adj_mat):
        """
        Comput supports
        """
        supports = []
        supports_mat = []
        if self.filter_type == "laplacian":  # ChebNet graph conv
            supports_mat.append(
                utils1.calculate_scaled_laplacian(adj_mat, lambda_max=None))
        elif self.filter_type == "random_walk":  # Forward random walk
            supports_mat.append(utils1.calculate_random_walk_matrix(adj_mat).T)
        elif self.filter_type == "dual_random_walk":  # Bidirectional random walk
            supports_mat.append(utils1.calculate_random_walk_matrix(adj_mat).T)
            supports_mat.append(
                utils1.calculate_random_walk_matrix(adj_mat.T).T)
        else:
            supports_mat.append(utils1.calculate_scaled_laplacian(adj_mat))
        for support in supports_mat:
            supports.append(torch.FloatTensor(support.toarray()))
        return supports

    def __getitem__(self, idx):
        """
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            a tuple of (x, y, seq_len, supports, adj_mat, writeout_fn)
        """
        h5_fn, segment_idx, seizure_label = self.file_tuples[idx]
        # print(h5_fn)

        cache_file_name = h5_fn.replace('.h5', f'_{str(self.max_seq_len)}_{str(segment_idx)}_cache.h5')
        cache_dir = os.path.join(self.cached_graphs_dir, str(self.max_seq_len), self.filter_type)
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_path = os.path.join(cache_dir, cache_file_name)

        eeg_clip = computeSliceMatrix(
                h5_fn=os.path.join(self.input_dir, h5_fn), clip_idx = segment_idx,
                time_step_size=self.time_step_size, clip_len=self.max_seq_len,
                is_fft=self.use_fft)
        if self.data_augment:
            swap_nodes = None
            curr_feature = self._random_scale(eeg_clip)
        else:
            swap_nodes = None
            curr_feature = eeg_clip.copy()

        x = torch.FloatTensor(curr_feature)
        y = torch.FloatTensor([seizure_label])
        seq_len = torch.LongTensor([self.max_seq_len])
        writeout_fn = h5_fn.split('.h5')[0]

        # Get adjacency matrix for graphs
        # if self.graph_type == 'individual':
        #     indiv_adj_mat = self._get_indiv_graphs(eeg_clip, swap_nodes)
        #     indiv_supports = self._compute_supports(indiv_adj_mat)
        #     curr_support = np.concatenate(indiv_supports, axis=0)
        #     if np.any(np.isnan(curr_support)):
        #         raise ValueError("Nan found in indiv_supports!")

        #     time_steps = eeg_clip.shape[0]  
        #     indiv_supports = torch.stack(indiv_supports)
        #     supports_seq = indiv_supports.repeat(time_steps, 1, 1, 1)
        #     adj_mat_seq = np.stack([indiv_adj_mat for _ in range(time_steps)])
            
        # elif self.graph_type == 'dynamic':
        is_cached = False
        if os.path.exists(cache_file_path):
            try:
                with h5py.File(cache_file_path, 'r') as cache_file:
                    # supports_seq = torch.from_numpy(cache_file['supports'][:])
                    adj_mat_seq = torch.from_numpy(cache_file['adj_mats'][:])
                is_cached = True
            except Exception as e:
                self.logger.info(f"Could'nt open this :- {e}, path -: {cache_file_path}")
                is_cached = False

            
        elif  not is_cached:
            adj_mats = []
            # supports = []
            for time_step in range(eeg_clip.shape[0]):
                adj_mat = self._get_indiv_graphs(eeg_clip[time_step][np.newaxis, :], swap_nodes)

                support = self._compute_supports(adj_mat)
                support = torch.stack(support)
                adj_mats.append(adj_mat)
                # supports.append(support)  

            adj_mat_seq = np.array(adj_mats)
            adj_mat_seq = torch.from_numpy(adj_mat_seq)
            # supports_seq = torch.stack(supports)
            try:
                with h5py.File(cache_file_path, 'w') as cache_file:
                    # cache_file.create_dataset('supports', data=supports_seq.numpy())
                    cache_file.create_dataset('adj_mats', data=adj_mat_seq)
            except Exception as e:
                self.logger.info(f"Could'nt save this :- {e}, path -: {cache_file_path}")


            # indiv_supports = supports_seq[-len(support):]
            # indiv_adj_mat = adj_mat_seq[-1]

        # elif self.adj_mat_dir is not None:
        #     indiv_adj_mat = self._get_combined_graph(swap_nodes)
        #     indiv_supports = self._compute_supports(indiv_adj_mat)
        # else:
        #     indiv_supports = []
        #     indiv_adj_mat = []

        if seq_len != self.max_seq_len:
            print(f"seq_len: {seq_len}")
            # print(f"supports_seq.shape: {supports_seq.shape}")
            print(f"adj_mat_seq.shape: {adj_mat_seq.shape}")
        expected_shapes = {
            "x": (12, 22, 100),
            "y": (1,),
            "seq_len": (1,),
            "supports_seq": (12, 2, 22, 22),
            "adj_mat_seq": (12, 22, 22),
        }
        # if x.shape != expected_shapes["x"]:
        #     print(f"[Mismatch] x shape at idx {idx}: {x.shape}")
        # if y.shape != expected_shapes["y"]:
        #     print(f"[Mismatch] y shape at idx {idx}: {y.shape}")
        # if seq_len.shape != expected_shapes["seq_len"]:
        #     print(f"[Mismatch] seq_len shape at idx {idx}: {seq_len.shape}")
        # if supports_seq.shape != expected_shapes["supports_seq"]:
        #     print(f"[Mismatch] supports_seq shape at idx {idx}: {supports_seq.shape}: {writeout_fn}")
        # if adj_mat_seq.shape != expected_shapes["adj_mat_seq"]:
        #     print(f"[Mismatch] adj_mat_seq shape at idx {idx}: {adj_mat_seq.shape} {writeout_fn}")

        return (x, y, seq_len, adj_mat_seq, writeout_fn)

def load_dataset_chb(
        task,
        input_dir,
        final_label_dir,
        cached_graphs_dir,
        train_batch_size,
        logger,
        test_batch_size=None,
        time_step_size=1,
        max_seq_len=12,
        standardize=True,
        num_workers=8,
        augmentation=False,
        adj_mat_dir=None,
        graph_type=None,
        top_k=None,
        filter_type='laplacian',
        use_fft=False,
        sampling_ratio=1,
        seed=123,
        preproc_dir=None):
    """
    Args:
        input_dir: dir to preprocessed h5 file
        raw_data_dir: dir to TUSZ raw edf files
        train_batch_size: int
        test_batch_size: int
        time_step_size: int, in seconds
        max_seq_len: EEG clip length, in seconds
        standardize: if True, will z-normalize wrt train set
        num_workers: int
        augmentation: if True, perform random augmentation on EEG
        adj_mat_dir: dir to pre-computed distance graph adjacency matrix
        graph_type: 'combined' (i.e. distance graph) or 'individual' (correlation graph)
        top_k: int, top-k neighbors of each node to keep. For correlation graph only
        filter_type: 'laplacian' for distance graph, 'dual_random_walk' for correlation graph
        use_fft: whether perform Fourier transform
        sampling_ratio: ratio of positive to negative examples for undersampling
        seed: random seed for undersampling
        preproc_dir: dir to preprocessed Fourier transformed data, optional
    Returns:
        dataloaders: dictionary of train/dev/test dataloaders
        datasets: dictionary of train/dev/test datasets
        scaler: standard scaler
    """
    if (graph_type is not None) and (
            graph_type not in ['individual', 'combined', 'dynamic']):
        raise NotImplementedError

    # load mean and std
    scaler = None

    dataloaders = {}
    datasets = {}
    for split in ['train', 'dev', 'test']:
        if split == 'train':
            data_augment = augmentation
        else:
            data_augment = False  

        dataset = SeizureDataset(task, 
                                 input_dir=input_dir,
                                 final_label_dir=final_label_dir,
                                 cached_graphs_dir=cached_graphs_dir,
                                 logger=logger,
                                 time_step_size=time_step_size,
                                 max_seq_len=max_seq_len,
                                 standardize=standardize,
                                 scaler=scaler,
                                 split=split,
                                 data_augment=data_augment,
                                 adj_mat_dir=adj_mat_dir,
                                 graph_type=graph_type,
                                 top_k=top_k,
                                 filter_type=filter_type,
                                 sampling_ratio=sampling_ratio,
                                 seed=seed,
                                 use_fft=use_fft,
                                 preproc_dir=preproc_dir)

        if split == 'train':
            shuffle = True
            batch_size = train_batch_size
        else:
            shuffle = False
            batch_size = test_batch_size

        loader = DataLoader(dataset=dataset,
                            shuffle=shuffle,
                            batch_size=batch_size,
                            num_workers=num_workers)
        
        dataloaders[split] = loader
        datasets[split] = dataset

    return dataloaders, datasets, scaler

# steps 

# resample all files into a specific folder. 
# now read these files using a parser that accepts (s, e, clip size, split, task)
    # reads all paricipants from s to e
    # divides each resampled file into 12s chunks, keep them strictly 12s chunks only. 
    # add the segment_idx and label in front. 
    # <edf_file_name>_idx.h5, idx, label
