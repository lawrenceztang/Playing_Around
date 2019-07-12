import torchvision.models as models
from torchvision import transforms
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
from skimage import io
from torch import optim
import time
import os
import copy
from PIL import Image
import torch.nn.functional as functional
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.markers as markers
from sklearn.cluster import KMeans

def test(model, device, test_loader):
    model.eval()
    outputs = []
    targets = []
    with torch.no_grad():
        i = 0
        for data, target in test_loader:
            if i == 1000:
                break
            data, target = data.to(device), target.to(device)
            output = model(data)
            targets.append(target)
            outputs.append(output)
            i+=1

    return outputs, targets

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

num_classes = 10
model = models.resnet18(pretrained=True)
print(model)
input_size = 224
model.fc = nn.Linear(512, num_classes)
model.fc = Identity()
model.avgpool = Identity()
model.to(torch.device("cuda:0"))

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('data', transform=transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]), download=True),
    batch_size=1, shuffle=True)


outputs, targets = test(model, torch.device("cuda:0"), test_loader)
targets = [x.reshape(-1).cpu().numpy() for x in targets]
outputs = [x.reshape(-1).cpu().numpy() for x in outputs]


kmeans = KMeans(n_clusters=num_classes).fit(outputs)
labels = kmeans.labels_
classesForEachCluster = [[0] * num_classes for i in range(num_classes)]

for i in range(len(outputs)):
    classesForEachCluster[kmeans.predict([outputs[i]])[0]][targets[i][0]] += 1

winningClassForEachCluster = [0] * num_classes
for i in range(len(classesForEachCluster)):
    max = -1
    classIndex = -1
    for j in range(len(classesForEachCluster)):
        if classesForEachCluster[i][j] > max:
            max = classesForEachCluster[i][j]
            classIndex = j
    winningClassForEachCluster[i] = classIndex

for i in range(len(labels)):
    labels[i] = winningClassForEachCluster[labels[i]]

right = 0
for i in range(len(labels)):
    if labels[i] == targets[i]:
        right += 1
print("Accuracy with Features: " + str(right / len(labels)))



outputs = []
i = 0
for data, target in test_loader:
    if i == 1000:
        break
    outputs.append(data.reshape(-1).numpy())
    i += 1

kmeans = KMeans(n_clusters=num_classes).fit(outputs)
labels = kmeans.labels_
classesForEachCluster = [[0] * num_classes for i in range(num_classes)]

for i in range(len(outputs)):
    classesForEachCluster[kmeans.predict([outputs[i]])[0]][targets[i][0]] += 1

winningClassForEachCluster = [0] * num_classes
for i in range(len(classesForEachCluster)):
    max = -1
    classIndex = -1
    for j in range(len(classesForEachCluster)):
        if classesForEachCluster[i][j] > max:
            max = classesForEachCluster[i][j]
            classIndex = j
    winningClassForEachCluster[i] = classIndex

for i in range(len(labels)):
    labels[i] = winningClassForEachCluster[labels[i]]

right = 0
for i in range(len(labels)):
    if labels[i] == targets[i]:
        right += 1
print("Accuracy without Features: " + str(right / len(labels)))



pca = PCA(n_components=2)
principalComponents = pca.fit_transform(outputs)
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.scatter(principalComponents[:, 0], principalComponents[:, 1], c=targets, cmap = plt.cm.nipy_spectral, marker = markers.MarkerStyle("."))
plt.show()
