# Reduced UK ERA5 dataset:

## Needed Files:

TODO: convert to Lambert Projection or use icosohedral mesh
- nwp_xy.npy (xy coordinates of the grid nodes)
- 

## Missing Features:

Add dummy files for these for now, since the code is expecting them. They won't be used by the dataloader however.

Atmospheric
- flux.pt (solar flux)

Static
- wtr_*.pt (water coverage)
- surface_geopotential.npy
    - https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form

Forcing
- nwp_toa_downwelling_shortwave_flux_* (shortwave flux?)


## ASK VICTOR

1) Neural LAM uses dataset in Lambert Projection coordinates, but ERA5 dataset is in lat/lon coordinates

Use Lambert Projection in order to use rectangular mesh graph. 

I'm still thinking whether to try to make ERA5 into Lambert Projection and use rectangular graph, OR
Keep data as is and use icosohedral graph

2) In the repo, dataset is split over several files. Aman put them all into the same file, claiming it would train faster. Which approach is correct?


## 

Experiments

Models:
Local GCN                 x
Local GraphCast           x
Local Hierchal GraphCast  x

Update Aaron ASAP
