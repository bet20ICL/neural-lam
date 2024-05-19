# Third-party
import cartopy
import numpy as np

class ERA5UKConstants:
    GRID_SHAPE = (65, 57)  # (y, x)
    GRID_FORCING_DIM = 12 # 4 variables * 3 time steps
    GRID_STATE_DIM = 8 * 6 # 8 levels * 6 variables
    HEAT_MAP_VARS = [4, 22, 13, 30, 38, 46]
    VAL_STEP_LOG_ERRORS = np.array([1, 2, 4, 8, 16]) 
    METRICS_WATCH = [
        "val_rmse",
    ]
    VAR_LEADS_METRICS_WATCH = {
        4: [1, 2, 4, 8, 16],
        22: [1, 2, 4, 8, 16],
        13: [1, 2, 4, 8, 16],
        30: [1, 2, 4, 8, 16],
        38: [1, 2, 4, 8, 16],
        46: [1, 2, 4, 8, 16],
    }
    LEVELS = ['50', '150', '250', '400', '500', '600', '850', '1000']
    PARAM_NAMES = ['z50', 'z150', 'z250', 'z400', 'z500', 'z700', 'z850', 'z1000', 'q50', 'q150', 'q250', 'q400', 'q500', 'q700', 'q850', 'q1000', 't50', 't150', 't250', 't400', 't500', 't700', 't850', 't1000', 'u50', 'u150', 'u250', 'u400', 'u500', 'u700', 'u850', 'u1000', 'v50', 'v150', 'v250', 'v400', 'v500', 'v700', 'v850', 'v1000', 'w50', 'w150', 'w250', 'w400', 'w500', 'w700', 'w850', 'w1000']
    PARAM_NAMES_SHORT = PARAM_NAMES
    PARAM_UNITS = ['m**2 s**-2', 'm**2 s**-2', 'm**2 s**-2', 'm**2 s**-2', 'm**2 s**-2', 'm**2 s**-2', 'm**2 s**-2', 'm**2 s**-2', 'kg kg**-1', 'kg kg**-1', 'kg kg**-1', 'kg kg**-1', 'kg kg**-1', 'kg kg**-1', 'kg kg**-1', 'kg kg**-1', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'K', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'm s**-1', 'Pa s**-1', 'Pa s**-1', 'Pa s**-1', 'Pa s**-1', 'Pa s**-1', 'Pa s**-1', 'Pa s**-1', 'Pa s**-1']
    VAL_MONTHS = ["01", "04", "07", "10"]
        
class MEPSConstants:
    GRID_SHAPE = (268, 238)  # (y, x)
    GRID_FORCING_DIM = 5 * 3 + 1  # 5 feat. for 3 time-step window + 1 batch-static
    GRID_STATE_DIM = 17
    HEAT_MAP_VARS = list(range(0, 48))
    VAL_STEP_LOG_ERRORS = np.array([1, 2, 3, 5, 10, 15, 19])
    METRICS_WATCH = []
    VAR_LEADS_METRICS_WATCH = {
        6: [2, 19],  # t_2
        14: [2, 19],  # wvint_0
        15: [2, 19],  # z_1000
    }
    # Variable names
    PARAM_NAMES = [
        "pres_heightAboveGround_0_instant",
        "pres_heightAboveSea_0_instant",
        "nlwrs_heightAboveGround_0_accum",
        "nswrs_heightAboveGround_0_accum",
        "r_heightAboveGround_2_instant",
        "r_hybrid_65_instant",
        "t_heightAboveGround_2_instant",
        "t_hybrid_65_instant",
        "t_isobaricInhPa_500_instant",
        "t_isobaricInhPa_850_instant",
        "u_hybrid_65_instant",
        "u_isobaricInhPa_850_instant",
        "v_hybrid_65_instant",
        "v_isobaricInhPa_850_instant",
        "wvint_entireAtmosphere_0_instant",
        "z_isobaricInhPa_1000_instant",
        "z_isobaricInhPa_500_instant",
    ]

    PARAM_NAMES_SHORT = [
        "pres_0g",
        "pres_0s",
        "nlwrs_0",
        "nswrs_0",
        "r_2",
        "r_65",
        "t_2",
        "t_65",
        "t_500",
        "t_850",
        "u_65",
        "u_850",
        "v_65",
        "v_850",
        "wvint_0",
        "z_1000",
        "z_500",
    ]
    PARAM_UNITS = [
        "Pa",
        "Pa",
        "W/m\\textsuperscript{2}",
        "W/m\\textsuperscript{2}",
        "-",  # unitless
        "-",
        "K",
        "K",
        "K",
        "K",
        "m/s",
        "m/s",
        "m/s",
        "m/s",
        "kg/m\\textsuperscript{2}",
        "m\\textsuperscript{2}/s\\textsuperscript{2}",
        "m\\textsuperscript{2}/s\\textsuperscript{2}",
    ]

WANDB_PROJECT = "neural-lam-bet20"

SECONDS_IN_YEAR = (
    365 * 24 * 60 * 60
)  # Assuming no leap years in dataset (2024 is next)

# Log prediction error for these lead times
VAL_STEP_LOG_ERRORS = np.array([1, 2, 3, 5, 10, 15, 19])

# Log these metrics to wandb as scalar values for
# specific variables and lead times
# List of metrics to watch, including any prefix (e.g. val_rmse)
METRICS_WATCH = []
# Dict with variables and lead times to log watched metrics for
# Format is a dictionary that maps from a variable index to
# a list of lead time steps
VAR_LEADS_METRICS_WATCH = {
    6: [2, 19],  # t_2
    14: [2, 19],  # wvint_0
    15: [2, 19],  # z_1000
}

# Variable names
PARAM_NAMES = [
    "pres_heightAboveGround_0_instant",
    "pres_heightAboveSea_0_instant",
    "nlwrs_heightAboveGround_0_accum",
    "nswrs_heightAboveGround_0_accum",
    "r_heightAboveGround_2_instant",
    "r_hybrid_65_instant",
    "t_heightAboveGround_2_instant",
    "t_hybrid_65_instant",
    "t_isobaricInhPa_500_instant",
    "t_isobaricInhPa_850_instant",
    "u_hybrid_65_instant",
    "u_isobaricInhPa_850_instant",
    "v_hybrid_65_instant",
    "v_isobaricInhPa_850_instant",
    "wvint_entireAtmosphere_0_instant",
    "z_isobaricInhPa_1000_instant",
    "z_isobaricInhPa_500_instant",
]

PARAM_NAMES_SHORT = [
    "pres_0g",
    "pres_0s",
    "nlwrs_0",
    "nswrs_0",
    "r_2",
    "r_65",
    "t_2",
    "t_65",
    "t_500",
    "t_850",
    "u_65",
    "u_850",
    "v_65",
    "v_850",
    "wvint_0",
    "z_1000",
    "z_500",
]
PARAM_UNITS = [
    "Pa",
    "Pa",
    "W/m\\textsuperscript{2}",
    "W/m\\textsuperscript{2}",
    "-",  # unitless
    "-",
    "K",
    "K",
    "K",
    "K",
    "m/s",
    "m/s",
    "m/s",
    "m/s",
    "kg/m\\textsuperscript{2}",
    "m\\textsuperscript{2}/s\\textsuperscript{2}",
    "m\\textsuperscript{2}/s\\textsuperscript{2}",
]

# Projection and grid
# Hard coded for now, but should eventually be part of dataset desc. files
GRID_SHAPE = (268, 238)  # (y, x)

LAMBERT_PROJ_PARAMS = {
    "a": 6367470,
    "b": 6367470,
    "lat_0": 63.3,
    "lat_1": 63.3,
    "lat_2": 63.3,
    "lon_0": 15.0,
    "proj": "lcc",
}

GRID_LIMITS = [  # In projection
    -1059506.5523409774,  # min x
    1310493.4476590226,  # max x
    -1331732.4471934352,  # min y
    1338267.5528065648,  # max y
]

# # Create projection
# LAMBERT_PROJ = cartopy.crs.LambertConformal(
#     central_longitude=LAMBERT_PROJ_PARAMS["lon_0"],
#     central_latitude=LAMBERT_PROJ_PARAMS["lat_0"],
#     standard_parallels=(
#         LAMBERT_PROJ_PARAMS["lat_1"],
#         LAMBERT_PROJ_PARAMS["lat_2"],
#     ),
# )
LAMBERT_PROJ = None

# Data dimensions
GRID_FORCING_DIM = 5 * 3 + 1  # 5 feat. for 3 time-step window + 1 batch-static
GRID_STATE_DIM = 17
