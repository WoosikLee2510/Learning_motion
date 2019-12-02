import torch
import torch.nn as nn
import glob
import os
import funcs as f
import sys
# Ignore warnings
import warnings
import numpy as np

warnings.filterwarnings("ignore")

# Create a PyTorch tensor with a fixed seed
torch.manual_seed(0)

kitti_data_dir = "/media/woosik/RPNG FLASH 2/kitti"
save_dir = "save"

if __name__ == '__main__':
    # load the dataset
    t_train, data_lengths_train, p_gt_train, v_gt_train, ang_gt_train, gyro_bis_train, acc_bis_train, t_test, data_lengths_test, p_gt_test, v_gt_test, ang_gt_test, gyro_bis_test, acc_bis_test = f.read_KITTI_data(
        kitti_data_dir)

    batch_size = 20
    input_dim = 6
    hidden_dim = 250
    layer_dim = 2
    output_dim = 1
    learning_rate = 0.001

    model = f.LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    iter = 0
    for repeat in range(10):
        for i in range(data_lengths_train.shape[0]):
            # compute total iteration
            total_iter_train = int(data_lengths_train[i] / batch_size)
            for j in range(total_iter_train):

                inputs = f.get_inputs(i, j, gyro_bis_train, acc_bis_train, batch_size)
                labels = f.get_labels(i, j, p_gt_train, v_gt_train, ang_gt_train, gyro_bis_train, acc_bis_train, batch_size)

                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Getting gradients w.r.t. parameters
                loss.backward()

                # Updating parameters
                optimizer.step()

                iter += 1
                if iter % 50 == 0:
                    # Calculate Accuracy
                    correct = 0
                    total = 0
                    # Iterate through test dataset
                    # compute total iteration
                    total_iter_test = int(data_lengths_test / batch_size)
                    for tj in range(total_iter_test):
                        inputs_test = f.get_inputs(0, tj, gyro_bis_test, acc_bis_test, batch_size)
                        labels_test = f.get_labels(0, tj, p_gt_test, v_gt_test, ang_gt_test, gyro_bis_test, acc_bis_test, batch_size)
                        # Forward pass only to get logits/output
                        outputs = model(inputs_test)

                        # Total number of labels
                        total += labels.size(0)
                        # Total correct predictions
                        correct += np.linalg.norm(
                            1 - (outputs.data.numpy() - labels.data.numpy().reshape((batch_size, 1))))

                    accuracy = 100 * correct / total
                    # Print Loss
                    print('Global Iteration: {}. Local Iteration: {}. Loss: {}. Accuracy: {}'.format(repeat, total_iter_train, loss.item(), accuracy))
