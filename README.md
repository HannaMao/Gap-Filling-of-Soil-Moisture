# Gap Filling of High-Resolution Soil Moisture for SMAP/Sentinel-1: A Two-Layer Machine Learning-Based Framework
Code for paper [Gap Filling of High-Resolution Soil Moisture for SMAP/Sentinel-1: A Two-Layer Machine Learning-Based Framework
](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019WR024902). An open access version of the paper can be found [here](https://eartharxiv.org/ce865/).

## Data Preprocessing
Code for data preprocessing is contained in folder **data_preprocessing**. Functions are then called by generate_experiment_data.py to generate experiment data. 

Data from various sources are first converted to a unified format [netCDF4](https://unidata.github.io/netcdf4-python/netCDF4/index.html) with their original resolutions being kept. They are then rescaled to have the same resolution as the SMAP/Sentinel-1 3 km soil moisture product. More details can be found in the paper.  


## Machine Learning Modeling
1. Features for brightness temperature downnscaling and soil moisture prediction are defined in _queries_single_day/queries_tb_v_disaggregated.txt_ and _queries_single_day/queries_soil_moisture.txt_ separately. You can define different feature sets at the same time by giving each set a unique number. Predictions will be output to a subfolder named by the given number.
2. Experiments including regional learning ones (spatial, temporal, and spatiotemporal), temporal limitation exploration, real gap filling are called from **regional_learning_experiments.py**.

   Experiments for the spatial limitation exploration are called from single_day_experiments.py.
3. Code for machine learning models are contained in **soil_moisture_downscaling/machine_learning**. New machine learning models can be added here.

## Cite this work
Mao, H., Kathuria, D., Duffield, N., & Mohanty, B. P. ( 2019). Gap filling of high‐resolution soil moisture for SMAP/Sentinel‐1: A two‐layer machine learning‐based framework. Water Resources Research, vol. 55, no. 8, pp. 6986–7009, 2019. https://doi.org/10.1029/2019WR024902

Biblatex entry:

    @article{map2019gap,
        author = {Mao, Hanzi and Kathuria, Dhruva and Duffield, Nick and Mohanty, Binayak P.},
        title = {Gap Filling of High-Resolution Soil Moisture for SMAP/Sentinel-1: A Two-Layer Machine Learning-Based Framework},
        journal = {Water Resources Research},
        volume = {55},
        number = {8},
        pages = {6986--7009},
        keywords = {soil moisture, machine learning, multiresolution gap filling, SMAP satellite, SENTINEL-1 satellite, spatial/temporal machine learning},
        doi = {10.1029/2019WR024902},
        url = {https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2019WR024902},
        eprint = {https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2019WR024902},
        year = {2019}
    }