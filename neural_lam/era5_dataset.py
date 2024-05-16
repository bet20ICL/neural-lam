
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
        pred_length=28, 
        split="train", 
        subsample_step=6,
        standardize=False,
        subset=False,
        control_only=False
    ):
        super().__init__()
        assert split in ("train", "val", "test"), "Unknown dataset split"
        self.sample_dir_path = os.path.join("data", dataset_name, "samples", split)

        self.split = split
        self.sample_length = pred_length + 2 # 2 init states
        self.subsample_step = subsample_step
        
        if split == "train":
            sample_paths = glob.glob(os.path.join(self.sample_dir_path, "*.npy"))
            self.sample_names = [os.path.basename(path) for path in sample_paths]
            self.sample_names.sort()
            self.length = len(self.sample_names) - self.sample_length + 1
            # print(self.length)
            # print(len(self.sample_names))
            
        elif split == "val":
            self.val_months = ["01", "04", "07", "10"]
            self.val_samples = []
            self.month_samples = {}

            for month in self.val_months:
                month_dir = os.path.join(self.sample_dir_path, month)
                sample_paths = glob.glob(os.path.join(month_dir, "*.npy"))
                sample_names = [os.path.join(month, os.path.basename(path)) for path in sample_paths]
                sample_names.sort()
                self.month_samples[month] = sample_names
                n_samples = len(sample_names) - self.sample_length + 1
                for i in range(n_samples):
                    self.val_samples.append((month, i))
                self.length = len(self.val_samples)

        if subset:
            self.sample_names = self.sample_names[:50] # Limit to 50 samples

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
        # validation dataset has different structure
        if self.split == "val":
            month, idx = self.val_samples[idx]
            self.sample_names = self.month_samples[month]
        
        # === Sample ===
        prev_prev_state = self._get_sample(self.sample_names[idx])
        prev_state = self._get_sample(self.sample_names[idx+1])        

        # N_grid = N_x * N_y; d_features = N_vars * N_levels
        init_states = torch.stack((prev_prev_state, prev_state), dim=0) # (2, N_grid, d_features)
        target_states = []
        for i in range(2, self.sample_length):
            target_states.append(self._get_sample(self.sample_names[idx+i]))
        target_states = torch.stack(target_states, dim=0) # (sample_len-2, N_grid, d_features)
        
        # TODO: add static and forcing features
        forcing = torch.zeros(target_states.shape[0], target_states.shape[1], 0) # (sample_len-2, N_grid, d_forcing)
        
        if self.standardize:
            # Standardize sample
            init_states = (init_states - self.data_mean) / self.data_std
            target_states = (target_states - self.data_mean) / self.data_std
        
        return init_states, target_states, forcing

        # # Time of day and year
        # sample_datetime = self.sample_names[idx].split('.')[0] # '2020010100'
        # dt_obj = dt.datetime.strptime(sample_datetime, "%Y%m%d%H")
        # dt_obj = dt_obj + dt.timedelta(
        #     hours=2 + subsample_index
        # )  # Offset for first index
        # # Extract for initial step
        # init_hour_in_day = dt_obj.hour
        # start_of_year = dt.datetime(dt_obj.year, 1, 1)
        # init_seconds_into_year = (dt_obj - start_of_year).total_seconds()

        # # Add increments for all steps
        # hour_inc = (
        #     torch.arange(self.sample_length) * self.subsample_step
        # )  # (sample_len,)
        # hour_of_day = (
        #     init_hour_in_day + hour_inc
        # )  # (sample_len,), Can be > 24 but ok
        # second_into_year = (
        #     init_seconds_into_year + hour_inc * 3600
        # )  # (sample_len,)
        # # can roll over to next year, ok because periodicity

        # # Encode as sin/cos
        # hour_angle = (hour_of_day / 12) * torch.pi  # (sample_len,)
        # year_angle = (
        #     (second_into_year / constants.SECONDS_IN_YEAR) * 2 * torch.pi
        # )  # (sample_len,)
        # datetime_forcing = torch.stack(
        #     (
        #         torch.sin(hour_angle),
        #         torch.cos(hour_angle),
        #         torch.sin(year_angle),
        #         torch.cos(year_angle),
        #     ),
        #     dim=1,
        # )  # (N_t, 4)
        # datetime_forcing = (datetime_forcing + 1) / 2  # Rescale to [0,1]
        # datetime_forcing = datetime_forcing.unsqueeze(1).expand(
        #     -1, target_states.shape[1], -1
        # )  # (sample_len, N_grid, 4)

        # # Put forcing features together
        # # forcing_features = torch.cat(
        # #     (flux, datetime_forcing), dim=-1
        # # )  # (sample_len, N_grid, d_forcing)

        # forcing_features = datetime_forcing

        # # Combine forcing over each window of 3 time steps
        # forcing_windowed = torch.cat(
        #     (
        #         forcing_features[:-2],
        #         forcing_features[1:-1],
        #         forcing_features[2:],
        #     ),
        #     dim=2,
        # )  # (sample_len-2, N_grid, 3*d_forcing)
        # # Now index 0 of ^ corresponds to forcing at index 0-2 of sample

        # static_features = torch.zeros(target_states.shape[1], 0) # (N_grid, 0)
        # forcing_windowed = torch.zeros(target_states.shape[0], target_states.shape[1], 0) # (sample_len-2, N_grid, df)
        # return init_states, target_states, static_features, forcing_windowed