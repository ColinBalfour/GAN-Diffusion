# Lets train a CIFAR10 image classifier
import importlib
import torch
import numpy as np
import networks as net
import os

importlib.reload(net)

pipeline = net.Pipeline()
model = net.CustomCNN().to(pipeline.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

home_path = os.path.expanduser("~")
JOB_FOLDER = os.path.join(home_path, "outputs/")
TRAINED_MDL_PATH = os.path.join(JOB_FOLDER, "cifar/cnn_custom_layer/")

os.makedirs(JOB_FOLDER, exist_ok=True)
os.makedirs(TRAINED_MDL_PATH, exist_ok=True)

epochs = 40
trainLossList = []
valAccList = []
for eIndex in range(epochs):
    # print("Epoch count: ", eIndex)

    train_epochloss = pipeline.train_step(model, optimizer)
    print("train complete")
    val_acc = pipeline.val_step(model)

    print(eIndex, train_epochloss, val_acc)

    valAccList.append(val_acc)
    trainLossList.append(train_epochloss)

    trainedMdlPath = TRAINED_MDL_PATH + f"{eIndex}.pth"
    torch.save(model.state_dict(), trainedMdlPath)

trainLosses = np.array(trainLossList)
testAccuracies = np.array(valAccList)

np.savetxt("train.log", trainLosses)
np.savetxt("test.log", testAccuracies)
