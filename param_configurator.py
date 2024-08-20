import configparser
import torch
import os
import uuid


class ParamConfigurator:
    """Parameter configurator class for deep learning pipeline."""

    def __init__(self):
        """# TODO: Docstring"""
        config = configparser.ConfigParser()
        config.read('/home/johannes/Code/ResponsePrediction/config.ini')

        # Global
        self.seed = config['global'].getint('seed')
        self.device = config['global']['device']

        if self.device == 'cuda':
            torch.cuda.empty_cache()

        # Data
        self.dataset = config['data']['dataset']
        assert self.dataset in ["sarcoma", "glioblastoma"]

        self.sequence = config['data']['sequence']
        assert self.sequence in ["T1", "T2", "T1T2"], "Please choose 'T1', 'T2' or 'T1T2'!"

        self.examination = config['data']['examination']
        assert self.examination in ['pre', 'post', 'prepost'], "Please choose examination out of 'pre', 'post' or 'prepost'!"

        self.channels = None

        self.artifact_dir = config['data']['artifact_directory']
        if not os.path.exists(self.artifact_dir):
            os.mkdir(self.artifact_dir)
        
        self.run_uuid = uuid.uuid4().hex
        self.run_dir = os.path.join(os.path.abspath(self.artifact_dir), self.run_uuid)        

        if not os.path.exists(self.run_dir):
            os.mkdir(self.run_dir)
        else:
            raise ValueError("Run dir does already exist!")

        # Architecture
        self.model_name = config['architecture']['model_name']
        self.model_depth = config['architecture'].getint('model_depth')
        assert self.model_depth in [10, 18, 34, 50, 101, 152, 200], "Please choose model depth out ouf '10', '18', '34', '50', '101', '152', '200'"

        # Training
        self.task = config['training']['task']
        assert self.task in ["classification", "regression"], "Please choose 'classification' or 'regression'!"
        self.batch_size = config['training'].getint('batch_size')
        self.accumulation_steps = config['training'].getint('accumulation_steps')
        self.epochs = config['training'].getint('epochs')
        self.num_workers = config['training'].getint('num_workers')       
        self.imbalance = config['training']['imbalance']
        assert self.imbalance in ['weight', 'oversample'] 
        self.imbalance_loss = config['training']['imbalance_loss']
        assert self.imbalance_loss in ['MCC', 'F1']
        self.augmentation = config['training'].getboolean('augmentation')
        self.pretrained = config['training'].getboolean('pretrained')

        # Optimizer
        self.learning_rate = config['optimizer'].getfloat('learning_rate')        
        self.optimizer = config['optimizer']['optimizer']
        assert self.optimizer in ["SGD", "Adam", "Novograd"]
        self.nesterov = config['optimizer'].getboolean('nesterov')
        self.momentum = config['optimizer'].getfloat('momentum')
        self.weight_decay = config['optimizer'].getfloat('weight_decay')
        self.scheduler_gamma = config['optimizer'].getfloat('scheduler_gamma')
        self.scheduler_step = config['optimizer'].getint('scheduler_step')