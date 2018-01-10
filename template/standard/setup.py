# Utils
import json
import os
import random
import sys
import time

# Torch related stuff
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

# DeepDIVA
import datasets
import models
from init.initializer import *


def set_up_model(num_classes, model_name, pretrained, optimizer_name, lr, no_cuda, resume, start_epoch, **kwargs):
    """
    Instantiate model, optimizer, criterion. Init or load a pretrained model or resume from a checkpoint.
    :param num_classes: int
        Number of classes for the model
    :param model: string
        Name of the model
    :param pretrained: bool
        Specify whether to load a pretrained model or not
    :param optimizer_name: string
        Name of the optimizer
    :param lr: float
        Value for learning rate
    :param no_cuda: bool
        Specify whether to use the GPU or not
    :param resume: string
        Path to a saved checkpoint
    :param kwargs: dict
        Any additional arguments.
    :return: model, criterion, optimizer, best_value, start_epoch
    """
    # Initialize the model
    logging.info('Setting up model {}'.format(model_name))
    model = models.__dict__[model_name](num_classes=num_classes, pretrained=pretrained)
    optimizer = torch.optim.__dict__[optimizer_name](model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    # Init the model
    # if args.init:
    #    init_model(model=model, data_loader=train_loader, num_points=50000)

    # Transfer model to GPU (if desired)
    if not no_cuda:
        logging.info('Transfer model to GPU')
        model = torch.nn.DataParallel(model).cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    # Resume from checkpoint
    if resume:
        if os.path.isfile(resume):
            logging.info("Loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume)
            start_epoch = checkpoint['epoch']
            best_value = checkpoint['best_value']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            val_losses = [checkpoint['val_loss']]
            logging.info("Loaded checkpoint '{}' (epoch {})".format(resume, checkpoint['epoch']))
        else:
            logging.error("No checkpoint found at '{}'".format(resume))
            sys.exit(-1)
    else:
        best_value = 0.0
    return model, criterion, optimizer, best_value, start_epoch


def set_up_dataloaders(model_expected_input_size, dataset, dataset_folder, batch_size, workers, **kwargs):
    """
    Set up the dataloaders for the specified datasets.
    :param model_expected_input_size: tuple
        Specify the height and width that the model expects.
    :param dataset_name: string
        Name of the datasets
    :param batch_size: int
        Number of datapoints to process at once
    :param workers: int
        Number of workers to use for the dataloaders
    :param kwargs: dict
        Any additional arguments.
    :return: dataloader, dataloader, dataloader, int
        Three dataloaders for train, val and test. Number of classes for the model.
    """

    logging.info('Loading datasets')
    # If the dataset selected is a class in dataset, use it.
    if (dataset is not None) and (dataset in datasets.__dict__):

        logging.debug('Using an user defined class to load: ' + dataset)
        train_ds = datasets.__dict__[dataset](root='.data/',
                                                  train=True,
                                                  download=True)

        val_ds = datasets.__dict__[dataset](root='.data/',
                                                train=False,
                                                val=True,
                                                download=True)

        test_ds = datasets.__dict__[dataset](root='.data/',
                                                 train=False,
                                                 download=True)
    # Else, assume it is an image folder whose path is passed as 'args.dataset_folder'
    else:
        logging.debug('Using the image folder routine to load from: ' + dataset_folder)
        """
        Structure of the dataset expected
        
        Split folders 
        -------------
        'args.dataset_folder' has to point to the three folder train/val/test. 
        Example:  
        
        ~/../../data/svhn
        
        where the dataset_folder contains the splits sub-folders as follow:
        
        args.dataset_folder/train
        args.dataset_folder/val
        args.dataset_folder/test
        
        Classes folders
        ---------------
        In each of the three splits (train,val,test) should have different classes in a separate folder with the class 
        name. The file name can be arbitrary (e.g does not have to be 0-* for classes 0 of MNIST).
        Example:
        
        train/dog/whatever.png
        train/dog/you.png
        train/dog/like.png
        
        train/cat/123.png
        train/cat/nsdf3.png
        train/cat/asd932_.png
        """

        # Get the splits folders
        traindir = os.path.join(dataset_folder, 'train')
        valdir = os.path.join(dataset_folder, 'test')  # TODO change as soon as svhn has a val set :)
        testdir = os.path.join(dataset_folder, 'test')

        # Sanity check on the splits folders
        if not os.path.isdir(traindir):
            logging.error("Train folder not found in the args.dataset_folder=" + dataset_folder)
            sys.exit(-1)
        if not os.path.isdir(valdir):
            logging.error("Val folder not found in the args.dataset_folder=" + dataset_folder)
            sys.exit(-1)
        if not os.path.isdir(testdir):
            logging.error("Test folder not found in the args.dataset_folder=" + dataset_folder)
            sys.exit(-1)

        # Init the dataset splits
        train_ds = torchvision.datasets.ImageFolder(traindir)
        val_ds = torchvision.datasets.ImageFolder(valdir)
        test_ds = torchvision.datasets.ImageFolder(testdir)

    # TODO what about the normalization?
    # train_ds.__getitem__(0) to get an image but its weird?
    # Set up dataset transforms
    logging.debug('Setting up dataset transforms')
    train_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    val_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    test_ds.transform = transforms.Compose([
        transforms.Scale(model_expected_input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=train_ds.mean, std=train_ds.std)
    ])

    # Setup dataloaders
    logging.debug('Setting up dataloaders')
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               shuffle=True,
                                               batch_size=batch_size,
                                               num_workers=workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=batch_size,
                                             num_workers=workers,
                                             pin_memory=True)

    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=batch_size,
                                              num_workers=workers,
                                              pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.num_classes


#######################################################################################################################


def set_up_logging(experiment_name, log_dir, log_folder, dataset, model_name, optimizer_name, lr, quiet, args_dict, **kwargs):
    """
    Set up a logger for the experiment
    :param experiment_name: string
        Name of the experiment. If not specify, accepted from command line.
    :param log_dir: string
        Path to where all experiment logs are stored.
    :param log_folder: string
        Used to override default log_folder generated using log_dir, experiment_name and other params. Useful on resume.
    :param dataset: string
        Name of the datasets.
    :param model_name: string
        Name of the model
    :param optimizer_name: string
        Name of the optimizer
    :param lr: float
        Value for learning rate used
    :param quiet: bool
        Specify whether to print log to console or only to text file
    :param args_dict: dict
        Contains the entire argument dictionary specified via command line.
    :return: log_folder
        Path to where logs for the experiment are stored
    """
    # Experiment name override
    if experiment_name is None:
        experiment_name = input("Experiment name:")

    # Setup Logging
    if dataset is None:
        dataset = os.path.dirname(kwargs['dataset_folder'])

    basename = log_dir
    experiment_name = experiment_name
    if log_folder is None:
        log_folder = os.path.join(basename,
                                  experiment_name,
                                  dataset,
                                  model_name,
                                  optimizer_name,
                                  str(lr),
                                  '{}'.format(time.strftime('%y-%m-%d-%Hh-%Mm-%Ss')))
    else:
        log_folder = log_folder
    logfile = 'logs.txt'
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    logging.basicConfig(
        format='%(asctime)s - %(filename)s:%(funcName)s %(levelname)s: %(message)s',
        filename=os.path.join(log_folder, logfile),
        level=logging.INFO)
    # Set up logging to console
    if not quiet:
        fmtr = logging.Formatter(fmt='%(funcName)s %(levelname)s: %(message)s')
        stderr_handler = logging.StreamHandler()
        stderr_handler.formatter = fmtr
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger().addHandler(stderr_handler)
        logging.info('Printing activity to the console')

    logging.info(
        'Set up logging. Log file: {}'.format(os.path.join(log_folder, logfile)))

    # Save args to logs_folder
    logging.info(
        'Arguments saved to: {}'.format(os.path.join(log_folder, 'args.txt')))
    with open(os.path.join(log_folder, 'args.txt'), 'w') as f:
        f.write(json.dumps(args_dict))



    return log_folder


def set_up_env(gpu_id, seed, multi_run, workers, no_cuda, **kwargs):
    """
    Set up the execution environment.
    Parameters
    ----------
    :param gpu_id: string
        Specify the GPUs to be used
    :param seed:    int
        Seed all possible seeds for deterministic run
    :param multi_run: int
        Number of runs over the same code to produce mean-variance graph.
    :param workers: int
        Number of workers to use for the dataloaders
    :param no_cuda: bool
        Specify whether to use the GPU or not
    :return: None
    """
    # Set visible GPUs
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # Seed the random

    if seed is not None:
        try:
            assert multi_run == None
        except:
            logging.warning('Arguments for seed AND multi-run should not be active at the same time!')
            raise SystemExit
        if workers > 1:
            logging.warning('Setting seed when workers > 1 may lead to non-deterministic outcomes!')
        # Python
        random.seed(seed)

        # Numpy random
        np.random.seed(seed)

        # Torch random
        torch.manual_seed(seed)
        if not no_cuda:
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.enabled = False
    return


