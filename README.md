# cancer_miRNA_classification_with_spiking_neural_network

| Authors |
|---------|
| Atanasio Giuseppe |
| Carachino Alessio |
| Di Gangi Francesco |
| Sorrentino Francesco |

## Requirements Instructions
Create a `conda` environment with the following command:

```
conda create -n "<env_name>" python=3.8.10
```

After that, perform the following commands:
```
conda activate <env_name>
pip install -r requirements.txt
```

## Repository Structure

```
.
├── README.md
├── cnn_search_space.json
├── data
│   ├── MLinApp_course_data
│   │   ├── tcga_mir_label.csv
│   │   └── tcga_mir_rpm.csv
│   ├── features
│   │   ├── filter_features.csv
│   │   ├── reduced_data_metalabel0_train.csv
│   │   ├── reduced_data_metalabel0_val.csv
│   │   ├── reduced_data_metalabel1_train.csv
│   │   ├── reduced_data_metalabel1_val.csv
│   │   ├── reduced_data_metalabel2_train.csv
│   │   ├── reduced_data_metalabel2_val.csv
│   │   ├── reduced_data_metalabel3_train.csv
│   │   ├── reduced_data_metalabel3_val.csv
│   │   ├── reduced_data_metalabel4_train.csv
│   │   └── reduced_data_metalabel4_val.csv
│   ├── metadata
│   │   ├── metadata_1_train.pkl
│   │   ├── metalabel_1_train.pkl
│   │   └── superclass_1_trainset.pkl
│   ├── models
│   │   └── cnn
│   ├── params
│   │   ├── cnn
│   │   │   ├── cnn_class0.json
│   │   │   ├── cnn_class1.json
│   │   │   ├── cnn_class2.json
│   │   │   ├── cnn_class3.json
│   │   │   └── cnn_class4.json
│   │   └── scnn
│   │       ├── scnn_class0.json
│   │       ├── scnn_class1.json
│   │       ├── scnn_class2.json
│   │       ├── scnn_class3.json
│   │       └── scnn_class4.json
│   └── results
│       └── scnn_population_encoding.csv
├── environment.yml
├── images
│   ├── image-1.png
│   └── image.png
├── models
│   ├── cnn
│   │   ├── cnn_class0.pth
│   │   ├── cnn_class1.pth
│   │   ├── cnn_class2.pth
│   │   ├── cnn_class3.pth
│   │   └── cnn_class4.pth
│   ├── instructions.md
│   └── scnn
│       ├── scnn_class0.pth
│       ├── scnn_class0_neurons100.pth
│       ├── scnn_class0_neurons25.pth
│       ├── scnn_class0_neurons50.pth
│       ├── scnn_class1.pth
│       ├── scnn_class1_neurons100.pth
│       ├── scnn_class1_neurons25.pth
│       ├── scnn_class1_neurons50.pth
│       ├── scnn_class2.pth
│       ├── scnn_class2_neurons100.pth
│       ├── scnn_class2_neurons25.pth
│       ├── scnn_class2_neurons50.pth
│       ├── scnn_class3.pth
│       ├── scnn_class3_neurons100.pth
│       ├── scnn_class3_neurons25.pth
│       ├── scnn_class3_neurons50.pth
│       ├── scnn_class4.pth
│       ├── scnn_class4_neurons100.pth
│       ├── scnn_class4_neurons25.pth
│       └── scnn_class4_neurons50.pth
├── nni_cnn_config.yml
├── nni_cnn_optimizer.py
├── nni_experiment_handler.py
├── nni_experiment_handler_scnn.py
├── nni_scnn_config.yml
├── nni_scnn_optimizer.py
├── notebooks
│   └── lab.ipynb
├── representation.ipynb
├── requirements.txt
├── scnn_search_space.json
├── src
│   ├── models
│   │   ├── CNN.py
│   │   ├── SCNN.py
│   │   ├── __pycache__
│   │   │   ├── CNN.cpython-310.pyc
│   │   │   └── SCNN.cpython-310.pyc
│   │   ├── instructions.md
│   │   └── metadata
│   │       └── rf_trained_1.pkl
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       │   ├── __init__.cpython-310.pyc
│       │   ├── data_loading_functions.cpython-310.pyc
│       │   ├── dataloader.cpython-310.pyc
│       │   ├── feature_selection.cpython-310.pyc
│       │   ├── metadata_functions.cpython-310.pyc
│       │   ├── statistics.cpython-310.pyc
│       │   ├── superclasses_functions.cpython-310.pyc
│       │   └── utils.cpython-310.pyc
│       ├── data_loading_functions.py
│       ├── dataloader.py
│       ├── feature_selection.py
│       ├── metadata_functions.py
│       ├── nni_cnn_optimization.py
│       ├── statistics.py
│       ├── superclasses_functions.py
│       └── utils.py
├── test.ipynb
└── train_population_enc.py
```

## Run Test

To run tests, please consider `test.ipynb`. The results are already printed out in the notebook.

## Parameters Optimization. 

There are two different ways to run [NNI](https://nni.readthedocs.io/en/stable/#):
- By using `python nni_experiment_handler.py` and `python nni_experiment_handler_scnn.py`;
- By using `nnictl` (check the usage [here](https://nni.readthedocs.io/en/stable/experiment/experiment_management.html)) with `nni_cnn_config.yml` and `nni_scnn_config.yml`.

## Representation

For super-class classification training, please check `representation.ipynb`.

## Feature Selection

This is a class to perform Feature Selection, and you can find it in `src/utils/feature_selection.py`.
