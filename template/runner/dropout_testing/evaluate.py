# Utils
import logging
import time
import warnings

import numpy as np
# Torch related stuff
import torch
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from util.evaluation.metrics import accuracy
# DeepDIVA
from util.misc import AverageMeter, _prettyprint_logging_label, save_image_and_log_to_tensorboard
from util.visualization.confusion_matrix_heatmap import make_heatmap


def validate(val_loader, model, criterion, writer, epoch, logging_label, val_classes, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to validate the model."""
    return _evaluate(val_loader, model, criterion, writer, epoch, logging_label, None, None, val_classes, no_cuda, log_interval, **kwargs)


def test(test_loader, model, criterion, writer, epoch, val_mean, val_std, val_classes, no_cuda=False, log_interval=20, **kwargs):
    """Wrapper for _evaluate() with the intent to test the model"""
    return _evaluate(test_loader, model, criterion, writer, epoch, 'test', val_mean, val_std, None, no_cuda, log_interval, **kwargs)


def _evaluate(data_loader, model, criterion, writer, epoch, logging_label, val_mean=None, val_std=None, val_classes=None, no_cuda=False, log_interval=10, dropout_samples=0, **kwargs):
    """
    The evaluation routine

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    model : torch.nn.module
        The network model being used
    criterion: torch.nn.loss
        The loss function used to compute the loss of the model
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.
    epoch : int
        Number of the epoch (for logging purposes)
    logging_label : string
        Label for logging purposes. Typically 'test' or 'valid'. Its prepended to the logging output path and messages.
    no_cuda : boolean
        Specifies whether the GPU should be used or not. A value of 'True' means the CPU will be used.
    log_interval : int
        Interval limiting the logging of mini-batches. Default value of 10.
    dropout_samples : int
        Number of forward passes used for dropout adjustments at test time

    Returns
    -------
    top1.avg : float
        Accuracy of the model of the evaluated split
    """
    multi_run = kwargs['run'] if 'run' in kwargs else None

    # Instantiate the counters
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    data_time = AverageMeter()

    # Switch to training mode (turn on dropout & such )
    model.eval()

    # Iterate over whole evaluation set
    end = time.time()

    # Empty lists to store the predictions and target values
    preds = []
    targets = []

    if dropout_samples > 0 and logging_label == 'test':
        right_dropout = 0
        right_mean = 0
        right_median = 0

    if dropout_samples > 1 and logging_label == 'test':
        conflicting_right = np.array([], dtype=np.int32)
        conflicting_wrong = np.array([], dtype=np.int32)
        nb_conflicting = 0
        nb_conflicting_25 = 0
        nb_conflicting_50 = 0
        nb_conflicting_100 = 0
        nb_conflicting_200 = 0
        right_predictions = 0
        wrong_predictions = 0
        wrong_predictions2 = 0
        wrong_predictions3 = 0
        """
        changed_wrong = 0
        changed_right = 0
        iterations = 0
        """
        diff_predictions = 0
        dropout_right = 0
        at_least_one_right = 0
        both_wrong = 0
        """
        changed_to_dropout = 0
        changed_to_median = 0

        conf_median_right = np.array([], dtype=np.int32)
        conf_median_wrong = np.array([], dtype=np.int32)
        right_pred_median = 0
        wrong_pred_median = 0
        
        #count when the mean of samples, the validation mean and the validation std get preds right
        count_RRR = 0
        #mean and val-mean right but std wrong
        count_RRW = 0
        # mean wrong, val-mean and std right
        count_WRR = 0
        #etc.
        count_RWW, count_WRW, count_WWW, count_RWR, count_WWR = 0,0,0,0,0
        """

    if logging_label == 'start_val':
        val_classes = np.zeros(len(data_loader.dataset.classes), dtype=np.int32)

    if logging_label == 'last_val':
        out_count = np.zeros(len(data_loader.dataset.classes), np.int32)
        outputs = []
        for i in range(len(data_loader.dataset.classes)):
            outputs.append(np.empty((val_classes[i], len(data_loader.dataset.classes)), dtype=np.float32))

        outputs = np.array(outputs)


    pbar = tqdm(enumerate(data_loader), total=len(data_loader), unit='batch', ncols=150, leave=False)
    for batch_idx, (input, target) in pbar:

        # Measure data loading time
        data_time.update(time.time() - end)

        # Moving data to GPU
        if not no_cuda:
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        # Convert the input and its labels to Torch Variables
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        model.eval()

        if logging_label == 'test':
            with torch.no_grad():
                dropout_output = model(input_var)
                if no_cuda:
                    dropout_output_np = dropout_output.numpy()
                else:
                    dropout_output_np = dropout_output.cpu().numpy()
            model.train()

        if dropout_samples == 0 and logging_label == 'test':
            output = dropout_output

        # Compute forward passes with dropout turned on during the testing phase
        elif dropout_samples > 1 and logging_label =='test':
            # Compute a matrix that contains the outputs of the different forward passes with dropout active
            # Matrix of size n, batch_size, nb_classes
            output_configs = np.zeros((dropout_samples, input.size(0), len(data_loader.dataset.classes)), dtype=np.float32)

            for i in range(dropout_samples):
                with torch.no_grad():
                    output_configs[i] = model(input_var)

            # Array of shape (batch_size, nb_classes) containing the standard deviation of outputs the subnetworks
            # which shows the reliability of each subnetwork
            # output_std = output_configs.std(axis=0)

            # Array of shape (batch_size, nb_classes) containing the mean of the outputs of the subnetworks
            output_mean = np.mean(output_configs, axis=0)
            out_median = np.median(output_configs, axis=0)

            for i in range(input.size(0)):
                if dropout_output_np[i].argmax() == target_var[i]:
                    right_dropout += 1
                if output_mean[i].argmax() == target_var[i]:
                    right_mean += 1
                if out_median[i].argmax() == target_var[i]:
                    right_median += 1

            # Array of shape (batch_size,) containing the mean of the standard variation of values of the mini-batch
            #  in order to not change the values drastically
            # output_std_mean = output_std.mean(axis=1)

            target_var = target_var.cpu()

            """
            Small iteration to see if the median is more often right than standard dropout by looking at the cases where
            dropout and the median of subnetworks have different predictions.
            """
            for i in range(input.size(0)):
                pred_mean = output_mean[i].argmax()
                pred_dropout = dropout_output_np[i].argmax()
                if pred_mean != pred_dropout:
                    diff_predictions += 1
                    if pred_dropout == target_var[i]:
                        at_least_one_right += 1
                        dropout_right += 1
                    elif pred_mean == target_var[i]:
                        at_least_one_right += 1
                    else:
                        both_wrong += 1

            """
            Here we store the number of dropout samples that don't predict the same thing as the mean and
            separate them between those where the mean predicts the right value (in conflicting_right) and
            those where the mean predicts the wrong value (in conflicting_wrong)
            
            The out_mean array will be changed by the function get_conflicting_wrong when it is called with
            the argument replacemax=True
            """
            out_mean = np.copy(output_mean)

            right_pred_mean_mb, wrong_pred_mean_mb, conflicting_right, conflicting_wrong, conflicting_configs\
                = get_conflicting_configs(input.size(0), out_mean, target_var, dropout_samples, output_configs, conflicting_right, conflicting_wrong, True)

            for i in range(input.size(0)):
                temp = conflicting_configs[i]
                if temp >= 200:
                    nb_conflicting += 1
                    nb_conflicting_25 += 1
                    nb_conflicting_50 += 1
                    nb_conflicting_100 += 1
                    nb_conflicting_200 += 1
                elif temp >= 100:
                    nb_conflicting += 1
                    nb_conflicting_25 += 1
                    nb_conflicting_50 += 1
                    nb_conflicting_100 += 1
                elif temp >= 50:
                    nb_conflicting += 1
                    nb_conflicting_25 += 1
                    nb_conflicting_50 += 1
                elif temp >= 25:
                    nb_conflicting += 1
                    nb_conflicting_25 += 1
                elif temp > 0:
                    nb_conflicting += 1

            right_predictions += right_pred_mean_mb
            wrong_predictions += wrong_pred_mean_mb

            """
            right_pred_median_mb, wrong_pred_median_mb, conf_median_right, conf_median_wrong\
                = get_conflicting_configs(input.size(0), out_median, target_var, dropout_samples, output_configs, conf_median_right, conf_median_wrong)

            right_pred_median += right_pred_median_mb
            wrong_pred_median += wrong_pred_median_mb
            """

            wrong_predictions2 += get_wrong_predictions(input.size(0), out_mean, target, True)

            wrong_predictions3 += get_wrong_predictions(input.size(0), out_mean, target, False)

            """
            From the 3 predictions dropout, mean and median, pick the one which has the smallest top value.
            
            
            for i in range(input.size(0)):
                pred_drop = dropout_output_np[i].argmax()
                pred_mean = output_mean[i].argmax()
                pred_median = out_median[i].argmax()
                
                if dropout_output_np[i][pred_drop] < output_mean[i][pred_mean]:
                    output_mean[i] = dropout_output_np[i]
                    pred_mean = pred_drop
                    changed_to_dropout += 1
                
                if output_mean[i][pred_mean] > out_median[i][pred_median]:
                    output_mean[i] = out_median[i]
                    changed_to_median += 1
            """

            """
            From the 3 predictions dropout, mean and median, pick the one which has the biggest difference between the
            top 2 values.
            

            for i in range(input.size(0)):
                top_drop = dropout_output_np[i].argsort()[-len(data_loader.dataset.classes):][::-1]
                top_mean = output_mean[i].argsort()[-len(data_loader.dataset.classes):][::-1]
                top_median = out_median[i].argsort()[-len(data_loader.dataset.classes):][::-1]
                
                diff_drop = dropout_output_np[i][top_drop[0]] - dropout_output_np[i][top_drop[1]]
                diff_mean = output_mean[i][top_mean[0]] - output_mean[i][top_mean[1]]
                diff_median = out_median[i][top_median[0]] - out_median[i][top_median[1]]
                
                if diff_drop < diff_mean:
                    output_mean[i] = dropout_output_np[i]
                    diff_mean = diff_drop
                
                if diff_median < diff_mean:
                    output_mean[i] = out_median[i]
            """

            """
            Multiply the top two values by the standard deviation of the other top value. This is in order to give more
            importance to the value that doesn't fluctuate. We do this only when there is at least one of the dropout
            configurations which have a different prediction value than the mean of all configurations.
            for i in range(input.size(0)):
                if conflicting_configs[i] > 0:
                    iterations += 1
                    top = output_mean[i].argsort()[-len(data_loader.dataset.classes):][::-1]
                    #print("\nOutput mean: " + list_to_string(output_mean[i]))
                    #print("\nTop values: " + list_to_string(top))
                    #print("\nTarget value: " + str(target_var[i].numpy()))
                    #print("\nOutput std: " + list_to_string(output_std[i]))
                    output_mean[i][top[0]] = output_mean[i][top[0]] * (output_std[i][top[1]]+1)
                    output_mean[i][top[1]] = output_mean[i][top[1]] * (output_std[i][top[0]]+1)
                    if output_mean[i][top[1]] > output_mean[i][top[0]]:
                        print("\nNew outputs: %0.2f, %0.2f\n" % (output_mean[i][top[0]], output_mean[i][top[1]]))
                        print("Changed from first prediction to second\n")
                        changed += 1
            """

            """
            Get the distance between the output and the training arithmetic mean using the squared euclidian distance to
            see if it looks like the mean of the outputs we had during validation or not, but only if the Dropout configs
            are giving conflicting predictions.
            Couldn't find a direct link with the accuracy of predictions


            dist_mean = np.empty((input.size(0), len(data_loader.dataset.classes)), dtype=np.float32)
            dist_std = np.empty((input.size(0), len(data_loader.dataset.classes)), dtype=np.float32)
            
            for i in range(input.size(0)):
                for j in range(len(data_loader.dataset.classes)):
                    dist_mean[i][j] = -np.square(output_mean[i] - val_mean[j]).sum()
                    dist_std[i][j] = -np.square(val_std[j] - output_std[i]).sum()
                        
            preds_val_mean = dist_mean.argmax(axis=1)
            preds_val_std = dist_std.argmax(axis=1)
            
            for i in range(input.size(0)):
                if conflicting_configs[i] > 0:
                    if output_mean[i].argmax() == target_var[i]:
                        if preds_val_mean[i] == target_var[i]:
                            if preds_val_std[i] == target_var[i]:
                                count_RRR += 1
                            else:
                                count_RRW += 1
                        else:
                            if preds_val_std[i] == target_var[i]:
                                count_RWR += 1
                            else:
                                count_RWW += 1
                    else:
                        if preds_val_mean[i] == target_var[i]:
                            if preds_val_std[i] == target_var[i]:
                                count_WRR += 1
                            else:
                                count_WRW += 1
                        else:
                            if preds_val_std[i] == target_var[i]:
                                count_WWR += 1
                            else:
                                count_WWW += 1
            """

            # wrong_count = 0.
            # wrong_index = []
            # wrong_pred = False

            """
            for i in range(len(input_var)):
                if y_pred[i] != target_var[i]:
                    wrong_index.append(i)
                    wrong_count += 1
            """

            # wrong_outputs = np.empty((wrong_count, len(output[0])), dtype=np.float32)

            """
            for i in range(wrong_count):
                print("\noutput")
                print(output[wrong_index[i]])
                print("\nstd")
                print(output_std[wrong_index[i]])
            """

            """
            for i in range(input.size(0)):
                if conflicting_configs[i] > 100:
                    pred = output_mean[i].argmax()
                    iterations += 1

                    output_mean[i] = output_mean[i] / val_std[pred] * output_std[i]

                    pred2 = output_mean[i].argmax()

                    if pred != pred2:
                        if pred == target_var[i]:
                            changed_wrong += 1
                        elif pred2 == target_var[i]:
                            changed_right += 1

            """

            """
            mean_val_std = np.mean(val_std, axis=0)

            for i in range(input.size(0)):
                if conflicting_configs[i] > 10:
                    iterations += 1

                    pred = output_mean[i].argmax()

                    for j in range(len(data_loader.dataset.classes)):
                        output_mean[i] = output_mean[i] / mean_val_std * output_std[i]

                    pred2 = output_mean[i].argmax()

                    if pred != pred2:
                        if pred == target_var[i]:
                            changed_wrong += 1
                        elif pred2 == target_var[i]:
                            changed_right += 1
            """

            """
            Take the config with the most difference between the top 2 values (hoping to get the config that looks less
            like other options)
            => Drop of ~5% accuracy on CIFAR-10
            for i in range(input.size(0)):
                max_diff = 0
                for j in range(dropout_samples):
                    idx = (-output_configs[j][i]).argsort()
                    for id in range(len(idx)):
                        if idx[id] == 0:
                            top_val = output_configs[j][i][id]
                        elif idx[id] == 1:
                            top2_val = output_configs[j][i][id]
                    diff = top_val - top2_val
                    if max_diff < diff:
                        max_diff = diff
                        output_mean[i] = output_configs[j][i]
            """

            """
            Use the config that has the highest top value (hoping to get the config that is most sure of itself
            => Drop of ~5% accuracy on CIFAR-10
            for i in range(input.size(0)):
                top_val = 0
                for j in range(dropout_samples):
                    idx = (-output_configs[j][i]).argsort()
                    for id in range(len(idx)):
                        if idx[id] == 0:
                            top_val_config = output_configs[j][i][id]
                    if top_val_config > top_val:
                        top_val = top_val_config
                        output_mean[i] = output_configs[j][i]
            """

            # if wrong_count == 0:
            #    wrong_outputs = []
            output = torch.from_numpy(output_mean)

            if not no_cuda:
                target_var = target_var.cuda()
                output = output.cuda()

        elif dropout_samples == 1 and logging_label == 'test':
            with torch.no_grad():
                output = model(input_var)

            if not no_cuda:
                output_np = output.cpu().numpy()
            else:
                output_np = output_np.numpy()

            for i in range(input.size(0)):
                if dropout_output_np[i].argmax() == target_var[i]:
                    right_dropout += 1
                if output_np[i].argmax() == target_var[i]:
                    right_mean += 1
                    right_median += 1

        # Standard output computation
        else:
            with torch.no_grad():
                output = model(input_var)

        if logging_label == 'start_val':
            for i in range(input.size(0)):
                val_classes[target_var[i]] += 1

        elif logging_label == 'last_val':
            model.train()
            for i in range(input.size(0)):
                outputs[target_var[i]][out_count[target_var[i]]] = output[i].cpu().numpy()
                out_count[target_var[i]] += 1
            model.eval()

        # Compute and record the loss
        with torch.no_grad():
            loss = criterion(output, target_var)
        losses.update(loss.data[0], input.size(0))

        # Compute and record the accuracy
        acc1 = accuracy(output.data, target, topk=(1,))[0]
        top1.update(acc1[0], input.size(0))

        # Get the predictions
        _ = [preds.append(item) for item in [np.argmax(item) for item in output.data.cpu().numpy()]]
        _ = [targets.append(item) for item in target.cpu().numpy()]

        # Add loss and accuracy to Tensorboard
        if multi_run is None:
            writer.add_scalar(logging_label + '/mb_loss', loss.data[0], epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_accuracy', acc1.cpu().numpy(), epoch * len(data_loader) + batch_idx)
        else:
            writer.add_scalar(logging_label + '/mb_loss_{}'.format(multi_run), loss.data[0],
                              epoch * len(data_loader) + batch_idx)
            writer.add_scalar(logging_label + '/mb_accuracy_{}'.format(multi_run), acc1.cpu().numpy(),
                              epoch * len(data_loader) + batch_idx)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx % log_interval == 0:
            pbar.set_description(logging_label +
                                 ' epoch [{0}][{1}/{2}]\t'.format(epoch, batch_idx, len(data_loader)))

            pbar.set_postfix(Time='{batch_time.avg:.3f}\t'.format(batch_time=batch_time),
                             Loss='{loss.avg:.4f}\t'.format(loss=losses),
                             Acc1='{top1.avg:.3f}\t'.format(top1=top1),
                             Data='{data_time.avg:.3f}\t'.format(data_time=data_time))

    if logging_label == 'test' and dropout_samples > 0:
        f = open("log/predictions_info.txt", 'a')

        f.write("\n\n---------------RUN START---------------\n\n")

        f.write("\nRight predictions from standard dropout\n")
        f.write(str(right_dropout))
        f.write("\nRight predictions from the arithmetic mean\n")
        f.write(str(right_mean))
        f.write("\nRight predictions from the median\n")
        f.write(str(right_median) + "\n")

        if dropout_samples > 1:
            f.write("\nNumber of predictions with at least 1 conflicting configs\n")
            f.write(str(nb_conflicting)+"\n")
            f.write("Number of predictions with at least 25 conflicting configs\n")
            f.write(str(nb_conflicting_25) + "\n")
            f.write("Number of predictions with at least 50 conflicting configs\n")
            f.write(str(nb_conflicting_50) + "\n")
            f.write("Number of predictions with at least 100 conflicting configs\n")
            f.write(str(nb_conflicting_100) + "\n")
            f.write("Number of predictions with at least 200 conflicting configs\n")
            f.write(str(nb_conflicting_200) + "\n")

            f.write("\nConflicting right mean\n")
            f.write("%0.2f\n" % conflicting_right.mean())
            f.write("Conflicting right std\n")
            f.write("%0.2f\n" % conflicting_right.std())
            f.write("Conflicting right max\n")
            f.write(str(conflicting_right.max()) + "\n")
            f.write("Conflicting right min\n")
            f.write(str(conflicting_right.min()) + "\n")

            f.write("\nConflicting wrong mean\n")
            f.write("%0.2f\n" % conflicting_wrong.mean())
            f.write("Conflicting wrong std\n")
            f.write("%0.2f\n" % conflicting_wrong.std())
            f.write("Conflicting wrong max\n")
            f.write(str(conflicting_wrong.max()) + "\n")
            f.write("Conflicting wrong min\n")
            f.write(str(conflicting_wrong.min()) + "\n")

            f.write("\nNumber of right predictions of the average of configs\n")
            f.write(str(right_predictions) + "\n")
            f.write("Number of wrong predictions of the average of configs\n")
            f.write(str(wrong_predictions) + "\n")
            f.write("Number of wrong predictions if we take the 2nd class when we're wrong\n")
            f.write(str(wrong_predictions2) + "\n")
            f.write("Number of wrong predictions if we take the 3rd class when we're wrong\n")
            f.write(str(wrong_predictions3) + "\n")

            """
            f.write("\nNumber of cases where we multiplied by the output std\n")
            f.write(str(iterations))
            f.write("\nNumber of cases where this changed the prediction to the right value\n")
            f.write(str(changed_right))
            f.write("\nNumber of cases where this added an error\n")
            f.write(str(changed_wrong) + "\n")
            """


            """
            f.write("\nConflicting right median mean\n")
            f.write("%0.2f\n" % conf_median_right.mean())
            f.write("Conflicting right std\n")
            f.write("%0.2f\n" % conf_median_right.std())
            f.write("Conflicting right max\n")
            f.write("%0.2f\n" % conf_median_right.max())
            f.write("Conflicting right min\n")
            f.write("%0.2f\n" % conf_median_right.min())
        
            f.write("\nConflicting wrong median mean\n")
            f.write("%0.2f\n" % conf_median_wrong.mean())
            f.write("Conflicting wrong std\n")
            f.write("%0.2f\n" % conf_median_wrong.std())
            f.write("Conflicting wrong max\n")
            f.write("%0.2f\n" % conf_median_wrong.max())
            f.write("Conflicting wrong min\n")
            f.write("%0.2f\n" % conf_median_wrong.min())
        
            f.write("\nNumber of right predictions by using the median\n")
            f.write("%0.2f\n" % right_pred_median)
            f.write("Number of wrong predictions\n")
            f.write("%0.2f\n" % wrong_pred_median)
            """

            f.write("\nNumber of predictions where avg/dropout are conflicting using " + str(dropout_samples) + " samples\n")
            f.write(str(diff_predictions) + "\n")
            f.write("When only dropout gets it right\n")
            f.write(str(dropout_right) + "\n")
            f.write("When only the average gets it right\n")
            f.write(str(diff_predictions-dropout_right-both_wrong) + "\n")
            f.write("When both methods give wrong predictions\n")
            f.write(str(both_wrong) + "\n\n")

            """
            f.write("\ncount_RRR, RRW, RWR, RWW, WRR, WRW, WWR, WWW\n")
            f.write(str(count_RRR) + '\n')
            f.write(str(count_RRW) + '\n')
            f.write(str(count_RWR) + '\n')
            f.write(str(count_RWW) + '\n')
            f.write(str(count_WRR) + '\n')
            f.write(str(count_WRW) + '\n')
            f.write(str(count_WWR) + '\n')
            f.write(str(count_WWW) + "\n\n")
            """

        f.close()

        """
        f = open("log/predictions_info.txt", 'ab')

        np.savetxt(f, val_mean, delimiter=', ', header="Mean of outputs for every class during validation", fmt='%0.2f')

        np.savetxt(f, val_std, delimiter=', ', header="Standard deviation of outputs for every class during validation", fmt='%0.2f')

        f.close()
        """

    # Make a confusion matrix
    try:
        cm = confusion_matrix(y_true=targets, y_pred=preds)
        confusion_matrix_heatmap = make_heatmap(cm, data_loader.dataset.classes)
    except ValueError:
        logging.warning('Confusion Matrix did not work as expected')

        confusion_matrix_heatmap = np.zeros((10, 10, 3))

    # Logging the epoch-wise accuracy and confusion matrix
    if multi_run is None:
        writer.add_scalar(logging_label + '/accuracy', top1.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix',
                                          image_tensor=confusion_matrix_heatmap, global_step=epoch)
    else:
        writer.add_scalar(logging_label + '/accuracy_{}'.format(multi_run), top1.avg, epoch)
        save_image_and_log_to_tensorboard(writer, tag=logging_label + '/confusion_matrix_{}'.format(multi_run),
                                          image_tensor=confusion_matrix_heatmap, global_step=epoch)

    logging.info(_prettyprint_logging_label(logging_label) +
                 ' epoch[{}]: '
                 'Acc@1={top1.avg:.3f}\t'
                 'Loss={loss.avg:.4f}\t'
                 'Batch time={batch_time.avg:.3f} ({data_time.avg:.3f} to load data)'
                 .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

    # Generate a classification report for each epoch
    _log_classification_report(data_loader, epoch, preds, targets, writer)

    if logging_label == 'start_val':
        return top1.avg, val_classes

    if logging_label == 'last_val':
        val_mean = np.empty((len(data_loader.dataset.classes), len(data_loader.dataset.classes)), dtype=np.float32)
        val_std = np.empty((len(data_loader.dataset.classes), len(data_loader.dataset.classes)), dtype=np.float32)

        for i in range(len(data_loader.dataset.classes)):
            val_mean[i] = outputs[i].mean(axis=0)
            val_std[i] = outputs[i].std(axis=0)

        return top1.avg, val_mean, val_std
    else:
        return top1.avg


def _log_classification_report(data_loader, epoch, preds, targets, writer):
    """
    This routine computes and prints on Tensorboard TEXT a classification
    report with F1 score, Precision, Recall and similar metrics computed
    per-class.

    Parameters
    ----------
    data_loader : torch.utils.data.DataLoader
        The dataloader of the evaluation set
    epoch : int
        Number of the epoch (for logging purposes)
    preds : list
        List of all predictions of the model for this epoch
    targets : list
        List of all correct labels for this epoch
    writer : tensorboardX.writer.SummaryWriter
        The tensorboard writer object. Used to log values on file for the tensorboard visualization.

    Returns
    -------
        None
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        classification_report_string = str(classification_report(y_true=targets,
                                                                 y_pred=preds,
                                                                 target_names=[str(item) for item in
                                                                               data_loader.dataset.classes]))
    # Fix for TB writer. Its an ugly workaround to have it printed nicely in the TEXT section of TB.
    classification_report_string = classification_report_string.replace('\n ', '\n\n       ')
    classification_report_string = classification_report_string.replace('precision', '      precision', 1)
    classification_report_string = classification_report_string.replace('avg', '      avg', 1)

    writer.add_text('Classification Report for epoch {}\n'.format(epoch), '\n' + classification_report_string, epoch)


def get_wrong_predictions(input_size, output, target, replacemax=False):
    predictions = output.argmax(axis=1)
    wrong_pred_mb = 0

    for i in range(input_size):
        if predictions[i] != target[i]:
            wrong_pred_mb += 1
            if replacemax:
                output[i][predictions[i]] = output[i].min()

    return wrong_pred_mb


def get_conflicting_configs(input_size, output, target, samples, output_configs, conf_right, conf_wrong, replacemax=False):
    conflicting_configs = np.zeros(input_size, dtype=np.int32)
    # get the predictions as an array of shape (batch_size) that contains the indexes of the top value
    predictions = output.argmax(axis=1)
    right_pred_mb = 0
    wrong_pred_mb = 0

    for i in range(input_size):
        for j in range(samples):
            if predictions[i] != output_configs[j].argmax(axis=1)[i]:
                conflicting_configs[i] += 1

    for i in range(input_size):
        if predictions[i] == target[i]:
            right_pred_mb += 1
            conf_right = np.append(conf_right, conflicting_configs[i])
        else:
            wrong_pred_mb += 1
            conf_wrong = np.append(conf_wrong, conflicting_configs[i])
            if replacemax:
                output[i][predictions[i]] = output[i].min()

    return right_pred_mb, wrong_pred_mb, conf_right, conf_wrong, conflicting_configs


def list_to_string(list):
    str = ""

    for i in range(len(list)):
        if i > 0:
            str = str + "  "
        if isinstance(list[i], int):
            str = str + "%d" % list[i]
        else:
            str = str + "%.2f" % list[i]

    return str


def softmax(x):
    exp_x = np.exp(x)
    exp_sum = exp_x.sum()
    return exp_x / exp_sum






