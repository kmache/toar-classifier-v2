## TOAR-classifier v2: A data-driven classification tool for global air quality stations
This notebook implements various machine learning approaches to obtain an objective station classification for global air quality monitoring stations as described in [1]. It has been developed in support of the international Tropospheric Ozone Assessment Report initiative, phase 2 (TOAR-II) [2]. TOAR has implemented a terabyte-scale database for global air quality data [3] with multiannual time series from over 23,000 stations. The objective of the station classification performed in this notebook is to create objective labels for the measurement sites as "urban", "suburban", or "rural" based on various features that provide hints of the characteristics of a station location. To this end, the machine learning models implemented here make use of the extensive metadata in the TOAR database, in particular the "global metadata" that is derived from various Earth Observation satellite data products (for details, see [1])
<img src="./figures/toar_classifier_v2.png" alt="My image" with="200">

### Files
- `data/`:  Contains all datasets used in this work, including machine learning model predictions for station categories
- `figures/`: contains all the figures.
- `TOAR-classifier_v2.ipynb`: Jupyter notebook containing the implementation.
- `requirements.txt`: Lists all required Python packages.

### Running the Code

**Tested on:** Ubuntu 24.04 with Python 3.12

### 🛠 Prerequisites
Ensure you have Python 3.12 installed along with either Jupyter Notebook, JupyterLab, or VS Code

1. clone the repository:
   - `git clone https://gitlab.jsc.fz-juelich.de/esde/toar-public/ml_toar_station_classification.git`
2. Change directory to ml_toar_station_classification:
   - `cd ml_toar_station_classification`
3. Creat virtual environment: 
   - `python -m venv TOAR-classifier_v2` # feel free to change the virtual environment as convenient 
4. Activate the virtual environment and register it with Jupyter: 
   - `python -m ipykernel install --user --name=TOAR-classifier_v2 --display-name "Python (TOAR-classifier_v2)"`
5. Open the notebook and select kernel:
   - open jupyter notebook, `jupyter-notebook` and select kernel `TOAR-classifier_v2` (or the name you assigned to your virtual environment).
6. Install required packages
   - Uncomment and run the first cell of the notebook to install all required packages.
7. Run the notebook cell by cell.


[1] Our paper https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1399/

[2] a reference to https://igacproject.org/activities/TOAR/TOAR-II

[3] a reference to https://toar-data.fz-juelich.de


### Citation

If you use this please cite

@article{Mache2025TOARClassifier,
  author = {Ramiyou Karim Mache and Sabine Schröder and Michael Langguth and Ankit Patnala and Martin G. Schultz},
  title = {TOAR-classifier v2: A data-driven classification tool for global air quality stations},
  year = {2025},
  note = {Correspondence: Ramiyou Karim Mache (k.mache@fz-juelich.de)},
  url = {https://egusphere.copernicus.org/preprints/2025/egusphere-2025-1399/}
}

