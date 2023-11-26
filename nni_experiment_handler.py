from pathlib import Path
import os
from nni.experiment import Experiment

search_space = {
    "nf1": {"_type": "randint", "_value": [2, 5]},
    "nf2": {"_type": "randint", "_value": [2, 80]},
    "nf3": {"_type": "randint", "_value": [2, 50]},
    "nf4": {"_type": "randint", "_value": [5, 150]},
    "cw1": {"_type": "randint", "_value": [5, 90]},
    "cw2": {"_type": "randint", "_value": [2, 250]},
    "cw3": {"_type": "randint", "_value": [5, 50]},
    "pw1": {"_type": "randint", "_value": [5, 60]},
    "pw2": {"_type": "randint", "_value": [10, 60]},
    "pw3": {"_type": "randint", "_value": [2, 40]},
    "dropout_0": {"_type": "uniform", "_value": [0.2, 0.5]},
    "dropout_1": {"_type": "uniform", "_value": [0.2, 0.5]},

    "batch_size": {"_type": "choice", "_value": [32, 64, 128]},
    "lr": {"_type": "uniform", "_value": [0.00001, 0.0025]}
}

experiment = Experiment('local')
experiment.config.experiment_name = 'cancer mirna case'
experiment.config.trial_concurrency = 1
experiment.config.max_trial_number = 1000
experiment.config.max_experiment_duration = '2h'
experiment.config.search_space = search_space
experiment.config.experiment_working_directory = os.path.join('results')
experiment.config.trial_command = 'python nni_cnn_optimizer.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'Anneal'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.trial_gpu_number = 0
experiment.config.training_service.use_active_gpu = False
# experiment.config.training_service.use_active_gpu = True

experiment.run(8080)
experiment.export_data()
#print(experiment.)
experiment.stop()
