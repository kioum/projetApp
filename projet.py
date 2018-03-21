cifar10_dir= 'cifar-10-batches-py'

#Label de ce qu'on peut trouver dans le data
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# Setting up the environment
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# Function to load a batch into memory
def load_batch(data_dir, batch_id):
    with open(os.path.join(cifar10_dir, 'data_batch_%i' % batch_id), mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    feats = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    lbls = batch['labels']
    return feats, lbls
    
# and load the first batch
feats, labels = load_batch(cifar10_dir, 1)

#test
img_id = 17
img = feats[img_id]
lbl = labels[img_id]
print('Label Id: {} - Class: {}'.format(lbl, label_names[lbl]))
plt.imshow(img)
# some stats
print([img.min(), img.max(), img.shape])