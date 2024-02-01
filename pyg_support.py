
class WeatherDataset(torch.utils.data.Dataset):
    ...
    def __getitem__(self, idx):
        # === Sample ===
        sample_path = os.path.join(self.sample_dir_path, f"nwp_{idx}.npy")
        ...
        # for example let init_states.shape = (x, y, z)
        return init_states, target_states, static_features, forcing_windowed

args.batch_size = 2
train_loader = pyg.loader.DataLoader(
            WeatherDataset(
                args.dataset, ...),
            args.batch_size, ...)

for batch in train_loader:
    batch[0].shape 
    # Expected: [2*x, y, z]
    # Actual: [2, x, y, z]