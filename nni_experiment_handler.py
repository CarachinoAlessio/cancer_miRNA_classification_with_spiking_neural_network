from pathlib import Path

from nni.experiment import Experiment

search_space = {
    "nf1": {"_type": "randint", "_value": [2, 256]},
    "nf2": {"_type": "randint", "_value": [2, 256]},
    "nf3": {"_type": "randint", "_value": [2, 256]},
    "nf4": {"_type": "randint", "_value": [2, 256]},
    "cw1": {"_type": "randint", "_value": [2, 7]},
    "cw2": {"_type": "randint", "_value": [2, 7]},
    "cw3": {"_type": "randint", "_value": [2, 7]},
    "pw1": {"_type": "randint", "_value": [2, 7]},
    "pw2": {"_type": "randint", "_value": [2, 7]},
    "pw3": {"_type": "randint", "_value": [2, 7]},

    "batch_size": {"_type": "choice", "_value": [32, 64, 128]},
    "lr": {"_type": "uniform", "_value": [0.0001, 0.1]}
}

experiment = Experiment('local')
experiment.config.experiment_name = 'cancer mirna case'
experiment.config.trial_concurrency = 2
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_command = 'python nni_cnn_optimizer.py'
experiment.config.trial_code_directory = Path(__file__).parent
experiment.config.tuner.name = 'Anneal'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.trial_gpu_number = 1
experiment.config.training_service.use_active_gpu = True
# experiment.config.training_service.use_active_gpu = True

experiment.run(8080)
#print(experiment.)
experiment.stop()
