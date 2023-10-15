import torch
import sys
sys.path.append("/home/huan1932/CertNID/bars/")
from certify.smoothing import Smooth2, Noise, Smooth
import os 
from dataset.datasets import get_dataset

from torch.utils.data import DataLoader, TensorDataset     
from dataset.datasets import *
from certify.distribution_transformer import distribution_transformers, loss_functions
from captum.attr import IntegratedGradients,Saliency
from certify.cert import *
from art.utils import load_mnist

# Load dataset
data = np.load('/home/huan1932/CertNID/CIC-IDS2018/IDS2018CADE/data/IDS_new_Infilteration.npz')
for i in data.keys():
    print(i)
# dataset = "ids18"
# data,y = get_dataset(dataset, "train")[0].to(device),get_dataset(dataset, "train")[1].to(device)
# encoder_list = torch.load(os.path.join(indir, "acidcheckpoint-encoder_list"))
# encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
# kernel_net_list = torch.load(os.path.join(indir, "acidcheckpoint-kernel_net_list"))
# kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list])
# classifier = AdaptiveClusteringClassifier(encoder_list, kernel_net_list, device)
# classifier.eval()
# unique_classes = torch.unique(y)

# # Create a dictionary to store TensorDatasets for each group
# grouped_datasets = {}

# # Loop through each unique class label and create a subset
# for class_label in unique_classes:
#     # Create a mask to filter samples of a specific class
#     mask = (y == class_label)
    
#     # Use the mask to select samples of the current class
#     x_group = data[mask]
#     y_group = y[mask]
    
#     # Create a TensorDataset for the current group
#     dataset = TensorDataset(x_group, y_group)
    
#     # Store the dataset in the dictionary with the class label as the key
#     grouped_datasets[class_label.item()] = dataset

# # Now, grouped_datasets is a dictionary where each key is a class label,
# # and the corresponding value is a TensorDataset containing samples of that class.
# grads = {}  
# for key,value in grouped_datasets.items():
#     summ = np.zeros((1,data.size(1)))
#     dataloader = DataLoader(value,batch_size=512, shuffle=False)
#     for (X,y) in dataloader:
#         saliency = Saliency(classifier)
#         # integrated_gradients = IntegratedGradients(classifier)
#         # attributions_ig = integrated_gradients.attribute(X, target=y, n_steps=200)
#         attribution = saliency.attribute(X, target=y)
#         summ += np.sum(attribution.cpu().numpy(),axis = 0)
#     summ /= len(dataloader.dataset)
#     grads[key] = summ
    
# for key,value in grads.items():
#     x_axis = np.arange(83)
#     plt.figure(figsize=(8, 6))  #
#     plt.plot(x_axis,value.squeeze())
#     plt.xlabel('features')
#     plt.ylabel('Saliency Value')
#     plt.title('Custom Saliency Map Visualization')

    
#     plt.savefig(os.path.join("resultsCertNID",'custom_saliency_map{}.png'.format(key)))
# noise_generator = create_noise_generator("gaussian",83)
# noise_generator.distribution_transformer = torch.load(os.path.join("resultsCertNID", "checkpoint-distribution-transformer")).to(device)
# plt.figure(figsize=(8, 6))  #
# x_axis = np.arange(83)
# plt.plot(x_axis,noise_generator.get_weight()[1:].cpu().detach().numpy())
# plt.xlabel('features')
# plt.ylabel('Saliency Value')
# plt.title('Custom Saliency Map Visualization')


# plt.savefig(os.path.join("resultsCertNID",'custom_saliency_map{}.png'.format('bars')))