# Utils
import logging

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
# Torch related stuff
from tqdm import tqdm
import torch

from util.evaluation.metrics import accuracy
from util.misc import save_image_and_log_to_tensorboard
# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label
from util.visualization.confusion_matrix_heatmap import make_heatmap


def feature_extract(data_loader, model, writer, epoch, no_cuda, log_interval, classify, dropout_samples=0, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set

    model : torch.nn.module
        The network model being used

    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    epoch : int
        Number of the epoch (for logging purposes)

    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.

    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.

    classify : boolean
        Specifies whether to generate a classification report for the data or not.

    Returns
    -------
        None
    """
    logging_label = 'apply'

    # Switch to evaluate mode (turn off dropout & such )
    model.eval()

    labels, features, preds, filenames = [], [], [], []
    #top1 = AverageMeter()

    multi_crop = False
    # Iterate over whole evaluation set
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=200)
    for batch_idx, (data, label, filename) in pbar:
        if len(data.size()) == 5:
            multi_crop = True
            bs, ncrops, c, h, w = data.size()
            data = data.view(-1, c, h, w)
        if not no_cuda:
            data = data.cuda()

        with torch.nograd():
            data_a = torch.autograd.Variable(data)

        # Compute forward passes with dropout turned on during the testing phase
        if dropout_samples > 1 and classify:
            model.train()

            # dropout_samples is the number of samples of forward passes for each batch

            # Compute a matrix that contains the outputs of the different forward passes with dropout active
            # Matrix of size n, batch_size, nb_classes
            output_configs = np.zeros((dropout_samples, input.size(0), len(data_loader.dataset.classes)),
                                      dtype=np.float32)

            for i in range(dropout_samples):
                with torch.no_grad():
                    output_configs[i] = model(data_a)

            # Array of shape (batch_size, nb_classes) containing the standard deviation of outputs the subnetworks
            # which shows the reliability of each subnetwork
            output_std = output_configs.std(axis=0)

            # Array of shape (batch_size, nb_classes) containing the mean of the outputs of the subnetwork
            output_mean = output_configs.mean(axis=0)

            # Array of shape (nb_classes,) containing the mean of the standard variation for every class in order
            # to not change the values drastically
            output_std_mean = output_configs.std(axis=0).mean(axis=0)

            out = torch.from_numpy(output_mean * output_std_mean / output_std)

            if not no_cuda:
                out = out.cuda()

        elif dropout_samples == 1 and classify:
            model.train()
            out = model(data_a)

        # Standard output computation
        else:
            with torch.no_grad():
                out = model(data_a)

        if multi_crop:
            out = out.view(bs, ncrops, -1).mean(1)

        preds.append([np.argmax(item.data.cpu().numpy()) for item in out])
        features.append(out.data.cpu().numpy())
        labels.append(label)
        filenames.append(filename)

        # Compute and record the accuracy
        #acc1 = accuracy(out.data, label, topk=(1,))[0]
        #top1.update(acc1[0], data.size(0))

        # Log progress to console
        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label + ' Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data_a), len(data_loader.dataset),
                       100. * batch_idx / len(data_loader)))

    # Measure accuracy (FPR95)
    num_tests = len(data_loader.dataset)
    labels = np.concatenate(labels, 0).reshape(num_tests)
    features = np.concatenate(features, 0)
    preds = np.concatenate(preds, 0)
    filenames = np.concatenate(filenames, 0)

    #logging.info(_prettyprint_logging_label(logging_label) +
    #            ' epoch[{}]: '
    #             'Acc@1={top1.avg:.3f}\t'
    #             .format(epoch, top1=top1))

    if classify:
        # Make a confusion matrix
        try:
            cm = confusion_matrix(y_true=labels, y_pred=preds)
            confusion_matrix_heatmap = make_heatmap(cm, data_loader.dataset.classes)
            save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                              image_tensor=confusion_matrix_heatmap, global_step=epoch)
        except ValueError:
            logging.warning('Confusion matrix received weird values')

        # Generate a classification report for each epoch
        logging.info('Classification Report for epoch {}\n'.format(epoch))
        logging.info('\n' + classification_report(y_true=labels,
                                                  y_pred=preds,
                                                  digits=5,
                                                  target_names=[str(item) for item in data_loader.dataset.classes]))
    else:
        preds = None


    return features, preds, labels, filenames
