
import os
import glob
import torch
import numpy as np
import datetime as dt

from neural_lam import utils, constants

class ERA5UKDataset(torch.utils.data.Dataset):
    """
    ERA5 UK dataset
    
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    N_x = 268 (width)
    N_y = 238 (height)
    N_grid = 268x238 = 63784 (total number of grid nodes)
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    def __init__(
        self,
        dataset_name,
        pattern="*.npy",
        pred_length=28, 
        split="train", 
        year="*",
        month=None,
        subsample_step=1,
        standardize=False,
        subset=False,
        control_only=False,
        args=None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)
        self.args = args
        self.split = split
        self.subsample_step = subsample_step
        # 2 init states, pred_length target states
        self.sample_length = (pred_length + 2) * self.subsample_step

        pattern = f"{year}{pattern}"
        if self.split == "train":
            sample_paths = glob.glob(os.path.join(self.sample_dir_path, pattern))
            # example name: '20200101000000.npy'
            self.sample_names = [os.path.basename(path) for path in sample_paths] 
            self.sample_names.sort()
            self.sample_times = [dt.datetime.strptime(n, '%Y%m%d%H%M%S.npy') for n in self.sample_names]

        else:
            assert month is not None, "Month must be specified for validation/test dataset"
            month_dir = os.path.join(self.sample_dir_path, month)
            sample_paths = glob.glob(os.path.join(month_dir, pattern))
            self.sample_names = [os.path.join(month, os.path.basename(path)) for path in sample_paths]
            self.sample_names.sort()
            self.sample_times = [dt.datetime.strptime(n[3:], '%Y%m%d%H%M%S.npy') for n in self.sample_names]

        if subset:
            self.sample_names = self.sample_names[:50] # Limit to 50 samples
        self.length = len(self.sample_names) - self.sample_length + 1
        
        assert (
            self.length > 0
        ), "Requesting too long time series samples"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
            )

    def __len__(self):
        return self.length

    def _get_sample(self, sample_name):
        sample_path = os.path.join(self.sample_dir_path, f"{sample_name}")
        try:
            full_sample = torch.tensor(np.load(sample_path),
                    dtype=torch.float32) # (N_lon*N_lat, N_vars*N_levels)
        except ValueError:
            print(f"Failed to load {sample_path}")
        
        return full_sample
    
    def __getitem__(self, idx):
        # === Sample ===
        prev_prev_state = self._get_sample(self.sample_names[idx])
        prev_state = self._get_sample(self.sample_names[idx+self.subsample_step])        

        # N_grid = N_x * N_y; d_features = N_vars * N_levels
        init_states = torch.stack((prev_prev_state, prev_state), dim=0) # (2, N_grid, d_features)
        
        target_states = []
        for i in range(2*self.subsample_step, self.sample_length, self.subsample_step):
            target_states.append(self._get_sample(self.sample_names[idx+i]))
        target_states = torch.stack(target_states, dim=0) # (sample_len-2, N_grid, d_features)
        
        if self.standardize:
            # Standardize sample
            init_states = (init_states - self.data_mean) / self.data_std
            target_states = (target_states - self.data_mean) / self.data_std
        
        # === Forcing features ===
        # Each step is 6 hours long
        hour_inc = torch.arange(self.sample_length // self.subsample_step) * 6 * self.subsample_step # (sample_len,)
        init_dt = self.sample_times[idx]
        
        init_hour = init_dt.hour
        hour_of_day = init_hour + hour_inc

        start_of_year = dt.datetime(init_dt.year, 1, 1)
        init_seconds_into_year = (init_dt - start_of_year).total_seconds()
        seconds_into_year = init_seconds_into_year + hour_inc * 3600

        hour_angle = (hour_of_day / 24) * 2 * torch.pi 
        year_angle = (seconds_into_year / constants.SECONDS_IN_YEAR) * 2 * torch.pi
        
        datetime_forcing = torch.stack(
            (
                torch.sin(hour_angle),
                torch.cos(hour_angle),
                torch.sin(year_angle),
                torch.cos(year_angle),
            ),
            dim=1,
        )  # (sample_len, 4)
        datetime_forcing = (datetime_forcing + 1) / 2 # Normalize to [0,1]
        datetime_forcing = datetime_forcing.unsqueeze(1).expand(
            -1, init_states.shape[1], -1
        )  # (sample_len, N_grid, 4)

        forcing = torch.cat(
            (
                datetime_forcing[:-2],
                datetime_forcing[1:-1],
                datetime_forcing[2:],
            ),
            dim=2,
        ) # (sample_len-2, N_grid, 12)
        
        if self.args and self.args.no_forcing:
            forcing = torch.zeros(target_states.shape[0], target_states.shape[1], 0) # (sample_len-2, N_grid, d_forcing)
        
        return init_states, target_states, forcing
    
def era5_dataset(
    dataset_name,
    pattern="*.npy",
    pred_length=28, 
    split="train", 
    year=[2022],
    subsample_step=1,
    standardize=False,
    subset=False,
    control_only=False,
    args=None,
):
    if split == "train":
        return ERA5UKDataset(
            dataset_name,
            pattern=pattern,
            pred_length=pred_length, 
            split=split, 
            # year="2022",
            subsample_step=subsample_step,
            standardize=standardize,
            subset=subset,
            control_only=control_only,
            args=args,
        )
    else:
        datasets = []
        for month in constants.ERA5UKConstants.VAL_MONTHS:
            datasets.append(
                ERA5UKDataset(
                    dataset_name,
                    pattern=pattern,
                    pred_length=pred_length, 
                    split=split, 
                    year="2023",
                    month=month,
                    subsample_step=subsample_step,
                    standardize=standardize,
                    subset=subset,
                    control_only=control_only,
                    args=args,
                )
            )
        return torch.utils.data.ConcatDataset(datasets)
    
class ERA5MultiResolutionDataset(torch.utils.data.Dataset):
    """
    ERA5 UK dataset
    
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    N_x = 268 (width)
    N_y = 238 (height)
    N_grid = 268x238 = 63784 (total number of grid nodes)
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    def __init__(
        self,
        dataset_names, 
        pred_length=28, 
        split="train", 
        subsample_step=1,
        standardize=False,
        subset=False,
        control_only=False,
        args=None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), "Unknown dataset split"
        
        self.datasets = []
        self.grid_sizes = []
        for dataset_name in dataset_names:
            dataset = era5_dataset(
                dataset_name, 
                pred_length=pred_length, 
                split=split, 
                subsample_step=subsample_step,
                standardize=standardize,
                subset=subset,
                control_only=control_only,
                args=args,
            )
            self.datasets.append(dataset)
            self.grid_sizes.append(dataset[0][0].shape[1])
        
    def __len__(self):
        return len(self.datasets[0])
    
    def __getitem__(self, idx):
        """
        Item consists of:
        init_states: (2, num_grid_nodes, d_features)
        target_states: (pred_steps, num_grid_nodes, d_features)
        forcing_features: (pred_steps, num_grid_nodes, d_forcing),
            where index 0 corresponds to index 1 of init_states
        """
        init_states, target_states, forcing = [], [], []
        for dataset in self.datasets:
            init_states_, target_states_, forcing_ = dataset[idx]
            init_states.append(init_states_)
            target_states.append(target_states_)
            forcing.append(forcing_)
        
        # init_states = torch.stack(init_states, dim=0)
        # target_states = torch.stack(target_states, dim=0)
        # forcing = torch.stack(forcing, dim=0)
        return init_states, target_states, forcing
    
class ERA5MultiTimeDataset(torch.utils.data.Dataset):
    """
    ERA5 UK dataset
    
    N_t' = 65
    N_t = 65//subsample_step (= 21 for 3h steps)
    N_x = 268 (width)
    N_y = 238 (height)
    N_grid = 268x238 = 63784 (total number of grid nodes)
    d_features = 17 (d_features' = 18)
    d_forcing = 5
    """
    def __init__(
        self,
        dataset_name,
        subsample_steps=[2, 1],
        pattern="*.npy",
        pred_length=28, 
        split="train", 
        year=2022,
        month=None,
        standardize=False,
        subset=False,
        control_only=False,
        args=None,
    ):
        super().__init__()
        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)
        self.args = args
        self.split = split
        
        pattern = f"{year}{pattern}"
        if self.split == "train":
            sample_paths = glob.glob(os.path.join(self.sample_dir_path, pattern))
            # example name: '20200101000000.npy'
            self.sample_names = [os.path.basename(path) for path in sample_paths] 
            self.sample_names.sort()
            self.sample_times = [dt.datetime.strptime(n, '%Y%m%d%H%M%S.npy') for n in self.sample_names]

        else:
            assert month is not None, "Month must be specified for validation/test dataset"
            month_dir = os.path.join(self.sample_dir_path, month)
            sample_paths = glob.glob(os.path.join(month_dir, pattern))
            self.sample_names = [os.path.join(month, os.path.basename(path)) for path in sample_paths]
            self.sample_names.sort()
            self.sample_times = [dt.datetime.strptime(n[3:], '%Y%m%d%H%M%S.npy') for n in self.sample_names]

        if subset:
            self.sample_names = self.sample_names[:50] # Limit to 50 samples
        
        # 2 init states, pred_length target states
        self.subsample_steps = subsample_steps
        self.pred_length = pred_length
        self.sample_length = pred_length + 2 * self.subsample_steps[0]
        self.length = len(self.sample_names) - self.sample_length + 1
        
        assert (
            self.length > 0
        ), "Requesting too long time series samples"

        # Set up for standardization
        self.standardize = standardize
        if standardize:
            ds_stats = utils.load_dataset_stats(dataset_name, "cpu")
            self.data_mean, self.data_std = (
                ds_stats["data_mean"],
                ds_stats["data_std"],
            )
        
    def __len__(self):
        return self.length

    def _get_sample(self, sample_name):
        sample_path = os.path.join(self.sample_dir_path, f"{sample_name}")
        try:
            full_sample = torch.tensor(np.load(sample_path),
                    dtype=torch.float32) # (N_lon*N_lat, N_vars*N_levels)
        except ValueError:
            print(f"Failed to load {sample_path}")
        
        return full_sample
    
    def __getitem__(self, idx):
        _init_states = []
        _target_states = []
        _forcing_features = []
        
        _start_idx = idx
        _end_idx = _start_idx + self.sample_length - 1
        for i in range(len(self.subsample_steps)):
            subsample_step = self.subsample_steps[i]
            
            if i == 0:
                start_idx = _start_idx
            else:
                start_idx = _start_idx + 2 * self.subsample_steps[0] - 2 * subsample_step
            
            # print("idx:", idx)
            # print("subsample_step:", subsample_step)
            # print("start_idx:", start_idx)
            # print()
            
            # === Sample ===
            prev_prev_state = self._get_sample(self.sample_names[start_idx])
            prev_state = self._get_sample(self.sample_names[start_idx+subsample_step])        

            # N_grid = N_x * N_y; d_features = N_vars * N_levels
            init_states = torch.stack((prev_prev_state, prev_state), dim=0) # (2, N_grid, d_features)
            
            target_states = []
            for j in range(start_idx + 2 * subsample_step, _end_idx + 1, subsample_step):
                target_states.append(self._get_sample(self.sample_names[j]))
            target_states = torch.stack(target_states, dim=0) # (sample_len-2, N_grid, d_features)
            
            if self.standardize:
                # Standardize sample
                init_states = (init_states - self.data_mean) / self.data_std
                target_states = (target_states - self.data_mean) / self.data_std
            
            # === Forcing features ===
            # Each step is 6 hours long
            hour_inc = torch.arange(len(target_states) + 2) * 6 * subsample_step # (sample_len,)
            init_dt = self.sample_times[start_idx]
            
            init_hour = init_dt.hour
            hour_of_day = init_hour + hour_inc

            start_of_year = dt.datetime(init_dt.year, 1, 1)
            init_seconds_into_year = (init_dt - start_of_year).total_seconds()
            seconds_into_year = init_seconds_into_year + hour_inc * 3600

            hour_angle = (hour_of_day / 24) * 2 * torch.pi 
            year_angle = (seconds_into_year / constants.SECONDS_IN_YEAR) * 2 * torch.pi
            
            datetime_forcing = torch.stack(
                (
                    torch.sin(hour_angle),
                    torch.cos(hour_angle),
                    torch.sin(year_angle),
                    torch.cos(year_angle),
                ),
                dim=1,
            )  # (sample_len, 4)
            datetime_forcing = (datetime_forcing + 1) / 2 # Normalize to [0,1]
            datetime_forcing = datetime_forcing.unsqueeze(1).expand(
                -1, init_states.shape[1], -1
            )  # (sample_len, N_grid, 4)

            forcing = torch.cat(
                (
                    datetime_forcing[:-2],
                    datetime_forcing[1:-1],
                    datetime_forcing[2:],
                ),
                dim=2,
            ) # (sample_len-2, N_grid, 12)
            
            if self.args and self.args.no_forcing:
                forcing = torch.zeros(target_states.shape[0], target_states.shape[1], 0) # (sample_len-2, N_grid, d_forcing)
                
            _init_states.append(init_states)
            _target_states.append(target_states)
            _forcing_features.append(forcing)
        
        return _init_states, [_target_states[-1]], _forcing_features
    
    
def era5_multi_time_dataset(
    dataset_name,
    pattern="*.npy",
    pred_length=28, 
    subsample_steps=[2, 1],
    split="train", 
    year=2022,
    standardize=False,
    subset=False,
    control_only=False,
    args=None,
):
    if split == "train":
        return ERA5MultiTimeDataset(
            dataset_name,
            subsample_steps=subsample_steps,
            pattern=pattern,
            pred_length=pred_length, 
            split=split, 
            year="2022",
            standardize=standardize,
            subset=subset,
            control_only=control_only,
            args=args,
        )
    else:
        datasets = []
        for month in constants.ERA5UKConstants.VAL_MONTHS:
            datasets.append(
                ERA5MultiTimeDataset(
                    dataset_name,
                    subsample_steps=subsample_steps,
                    pattern=pattern,
                    pred_length=pred_length, 
                    split=split, 
                    year="2023",
                    month=month,
                    standardize=standardize,
                    subset=subset,
                    control_only=control_only,
                    args=args,
                )
            )
        return torch.utils.data.ConcatDataset(datasets)