"""
This file is the template for the boilerplate of train/test of a DNN for dropout testing

There are a lot of parameter which can be specified to modify the behaviour and they should be used 
instead of hard-coding stuff.
"""

import logging
import sys

# Utils
import numpy as np

# DeepDIVA
import models
# Delegated
from template.runner.dropout_testing import evaluate, train
from template.setup import set_up_model, set_up_dataloaders
from util.misc import checkpoint, adjust_learning_rate


class DropoutTesting:
    @staticmethod
    def single_run(writer, current_log_folder, model_name, epochs, lr, decay_lr, validation_interval,
                   **kwargs):
        """
        This is the main routine where train(), validate() and test() are called.

        Parameters
        ----------
        writer : Tensorboard.SummaryWriter
            Responsible for writing logs in Tensorboard compatible format.
        current_log_folder : string
            Path to where logs/checkpoints are saved
        model_name : string
            Name of the model
        epochs : int
            Number of epochs to train
        lr : float
            Value for learning rate
        kwargs : dict
            Any additional arguments.
        decay_lr : boolean
            Decay the lr flag
        validation_interval: int
            Run evaluation on validation set every N epochs

        Returns
        -------
        train_value : ndarray[floats] of size (1, `epochs`)
            Accuracy values for train split
        val_value : ndarray[floats] of size (1, `epochs`+1)
            Accuracy values for validation split
        test_value : float
            Accuracy value for test split
        """
        # Get the selected model input size
        model_expected_input_size = models.__dict__[model_name]().expected_input_size
        DropoutTesting._validate_model_input_size(model_expected_input_size, model_name)
        logging.info('Model {} expects input size of {}'.format(model_name, model_expected_input_size))

        # Setting up the dataloaders
        train_loader, val_loader, test_loader, num_classes = set_up_dataloaders(model_expected_input_size, **kwargs)

        # Setting up model, optimizer, criterion
        model, criterion, optimizer, best_value, start_epoch = set_up_model(num_classes=num_classes,
                                                                            model_name=model_name,
                                                                            lr=lr,
                                                                            train_loader=train_loader,
                                                                            **kwargs)

        # Core routine
        logging.info('Begin training')
        val_value = np.zeros((epochs + 1 - start_epoch))
        train_value = np.zeros((epochs - start_epoch))

        val_value[-1], val_classes = DropoutTesting._validate(val_loader, model, criterion, writer, -1, 'start_val', **kwargs)

        for epoch in range(start_epoch, epochs):
            # Train
            train_value[epoch] = DropoutTesting._train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

            # Validate
            if epoch % validation_interval == 0 and epoch == epochs-1:
                val_value[epoch], val_mean, val_std = DropoutTesting._validate(val_loader, model, criterion, writer, epoch, 'last_val', val_classes, **kwargs)
            elif epoch % validation_interval == 0:
                val_value[epoch] = DropoutTesting._validate(val_loader, model, criterion, writer, epoch, 'val', None, **kwargs)
            if decay_lr is not None:
                adjust_learning_rate(lr=lr, optimizer=optimizer, epoch=epoch, decay_lr_epochs=decay_lr)
            best_value = checkpoint(epoch, val_value[epoch], best_value, model, optimizer, current_log_folder)

        # Test
        test_value = DropoutTesting._test(test_loader, model, criterion, writer, epochs - 1, val_mean=val_mean, val_std=val_std, val_classes=val_classes, **kwargs)
        logging.info('Training completed')

        return train_value, val_value, test_value

    ####################################################################################################################
    @staticmethod
    def _validate_model_input_size(model_expected_input_size, model_name):
        """
        This method verifies that the model expected input size is a tuple of 2 elements.
        This is necessary to avoid confusion with models which run on other types of data.

        Parameters
        ----------
        model_expected_input_size
            The item retrieved from the model which corresponds to the expected input size
        model_name : String
            Name of the model (logging purpose only)

        Returns
        -------
            None
        """
        if type(model_expected_input_size) is not tuple or len(model_expected_input_size) != 2:
            logging.error('Model {model_name} expected input size is not a tuple. '
                          'Received: {model_expected_input_size}'
                          .format(model_name=model_name,
                                  model_expected_input_size=model_expected_input_size))
            sys.exit(-1)

    ####################################################################################################################
    """
    These methods delegate their function to other classes in this package. 
    It is useful because sub-classes can selectively change the logic of certain parts only.
    """

    @classmethod
    def _train(cls, train_loader, model, criterion, optimizer, writer, epoch, **kwargs):
        return train.train(train_loader, model, criterion, optimizer, writer, epoch, **kwargs)

    @classmethod
    def _validate(cls, val_loader, model, criterion, writer, epoch, logging_label, val_classes=None, **kwargs):
        return evaluate.validate(val_loader, model, criterion, writer, epoch, logging_label, val_classes, **kwargs)

    @classmethod
    def _test(cls, test_loader, model, criterion, writer, epoch, val_mean, val_std, val_classes=None, **kwargs):
        return evaluate.test(test_loader, model, criterion, writer, epoch, val_mean, val_std, val_classes, **kwargs)
