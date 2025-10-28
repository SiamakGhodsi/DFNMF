# A Deep Latent Factor Graph Clustering <br />with Fairness-Utility Trade-off Perspective

Repository to demonstrate codes, instructions, dependencies (i.e., package requirements), and data, of the paper titled "A Deep Latent Factor Model for Fairness-Utility Trade-off in Graph Clustering" accepted in IEEE BigData 2025 Macau.

### Here is the code structure 
<pre>
📦 project-root/
├── 📂 data/                   # Data loaders & datasets
│   ├── __init__.py
│   ├── data_loader.py
│   ├── sbm_generator.py
│   ├── 📁 DrugNet/
│   ├── 📁 LastFM/
│   ├── 📁 NBA/
│   ├── 📁 Pokec/
│   └── 📁 School/
│
├── 📂 models/                 # Model implementations
│   ├── __init__.py
│   ├── competitors.py
│   ├── dfnmf.py
│   ├── dmon.py
│   ├── nmf_helpers.py
│   └── sc_helpers.py
│
├── 📂 utils/                  # Utilities & metric functions
│   ├── __init__.py
│   ├── evaluations.py
│   ├── evaluations_optmzd.py
│   └── utils.py
│
├── 📂 experiments/            # Main experiments and results
│   ├── spm.py
│   ├── experiments.py
│   ├── 📁 cd/
│   ├── 📁 drug/
│   ├── 📁 lfm/
│   ├── 📁 fb/
│   ├── 📁 fr/
│   ├── 📁 nba/
│   ├── 📁 Pokec/
│   ├── 📁 SBM/
│   └── 📁 visualizations/
│       └── visualizations.ipynb
</pre>


## Instructions 
Our code also provides comparisons to the spectral-based clustering models-- including vanilla SC, FairSC and scalable FairSC-models--, as well as GNN-based and NMF-based baselines implemented in Python and utilizes the respective baseline implementations automatically.
Start by installing the requirements.
<pre>
pip install -r requirements.txt
</pre>


### Run SBM Experiments
In order to run the SBM experiments, redirect to the experiments directory and run the "sbm.py" script using python3. It itertaes over n=2000, n=5000, 
and, n=10000 number of nodes for k=5 clusters and h=2 groups as in our experiments and compared our model **_DFNMF_** with the baselines.

<pre>
python sbm.py
</pre>

For different parameter setups, please change these values directly in the code snippet "sbm.py" and run. 

### Run Real-Data Experiments
We have provided the cleaned, pre-processed, ready-to-use real datasets used in the DFNMF\data directory. For each dataset, the data_loader function loads the corresponding dataset based on the input argument in the main function.
In order to run the experiment for either of the real datasets, redirect to the experiments directory and run the "experiments.py" script. 
As also commented in the code, you need to select the dataset to execute analysis: 
<pre>
Real Dataset Names
  <i> 
  1) Diaries 
  2) Facebook 
  3) Friendship 
  4) Drugnet 
  5) NBA 
  6) LastFM, 
  7) Pokec 
  </i>
</pre>
  
Please input the dataset ID as an ordinal number from $1-7$ in the configuration data structure in the main function to indicate the name of the dataset in the above order. 

<pre>
python experiments.py
</pre>

The code will automatically load the data using the proper data_loader and run a grid-search for DFNMF with a range of $\lambda \in [0.001,\cdots,1000]$, 
compared to the baselines for $k \in [2,10]$ clusters.

<!-- ### Visualizations
The notebook that visualizes our obtained results can be found under the "experiments\visualizations" directory. -->

<pre>
Don't forget to cite this work if you use our material: such as the paper, idea, or code material, including datasets, baselines and etc. 
**Citation**
Ghodsi, S., Seyedi, A., Le Quy, T., Karimi, F., Ntoutsi, E. (2025).
*A Deep Latent Factor Graph Clustering with a Fairness–Utility Trade-off Perspective.*
IEEE Big Data 2025. Preprint: https://arxiv.org/abs/2510.23507

@inproceedings{Ghodsi2025DLFGC,
  author    = {Siamak Ghodsi and Amjad Seyedi and Tai Le Quy and Fariba Karimi and Eirini Ntoutsi},
  title     = {A Deep Latent Factor Graph Clustering with a Fairness--Utility Trade-off Perspective},
  booktitle = {IEEE International Conference on Big Data (BigData)},
  year      = {2025},
  note      = {Preprint: \url{https://arxiv.org/abs/2510.23507}}
}
</pre>
