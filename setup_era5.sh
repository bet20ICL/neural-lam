# Very Long!!
# python era5_download_data.py

# Long!
# python era5_data_proc.py

# python create_mesh.py --dataset era5_uk --graph uk_graphcast

# python create_grid_features.py --dataset era5_uk

# python create_parameter_weights.py --dataset era5_uk

# ======= UK big =======
python era5_data_proc.py

python create_mesh.py --dataset era5_uk_big --graph uk_big_graphcast

python create_grid_features.py --dataset era5_uk_big

python create_parameter_weights.py --dataset era5_uk_big