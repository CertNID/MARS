import sys
from captum.attr import IntegratedGradients,Saliency
sys.path.append('/home/huan1932/CertNID/bars/')
from certify.smoothing import Smooth2, Noise, Smooth1, Sigma
from certify.distribution_transformer import distribution_transformers, loss_functions
from certify.optimizer import *
from certify.optimizing_noise import optimizing_noise, optimizing_sigma
from time import time
import torch
import os
from dataset.datasets import *
import torch.nn as nn
from classifier.detect import calc_centroid_dis_median_mad, Detector

from classifier.classifier import AdaptiveClusteringClassifier
import os, random
import numpy as np
from tqdm import tqdm
import pickle
import sys
sys.path.append('../bars/')

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import seaborn as sns
sns.set_style("darkgrid")

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from train_and_evaluate.train_utils import *





indir = "/home/huan1932/CertNID/resultsCertNID/old-trained-model"
save_dir_certify = 'CERTs/'

n_gpu = torch.cuda.device_count()







if not os.path.exists(save_dir_certify):
    os.makedirs(save_dir_certify)
num_samples_certify = 10000 # Number of certified samples sampled from dataset
# device = "cuda:1" if torch.cuda.is_available() else "cpu"

def create_noise_generator(feature_noise_distribution,d,device):
    dist_trans = distribution_transformers[feature_noise_distribution](d).to(device)
    return Noise(dist_trans, d, device)
def create_sigma_generator(d,device):
    return Sigma(d, device)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
 
def cadecertify(dataset,feature_noise_distribution,save_dir_certify,certify_class,mad_threshold,norm,certmethod,seed,device,purpose):
    learning_rate_shape = 2e-4 # Learning rate for optimzing noise shape
    indir = "/home/huan1932/CertNID/resultsCertNID/trained-model"
    nt_shape = 1000 # Number of noised samples for optimzing noise shape
    lambda_shape = 1e-1 # Regularizer weight
    num_epochs_shape = 5 # Number of epochs for optimzing noise shape
    x_train, _, _, _ = get_dataset(dataset, "train")
    d = x_train.size(1) # Number of feature dimensions
    num_classes_certify = 2 # Number of certified classes. i.e., Known class(0), Drift class(1)
    n0 = 100 # Number of noised samples for identify cA
    n = 10000 # Number of noised samples for estimate pA
    alpha = 1e-3 # Failure probability
    init_step_size_scale = 5e-2 # Initial update step size of t for optimzing noise scale
    init_ptb_t_scale = 1e-2 # Initial perturbation of t for optimzing noise scale
    decay_factor_scale = 0.5 # Decay factor for optimzing noise scale
    max_decay_scale = 6 # Maximum decay times for optimzing noise scale
    max_iter_scale = 100 # Maximum iteration times for optimzing noise scale
    batch_size_iteration_certify = 128 # Batch size of certified samples for robustness certification
    batch_size_memory_certify = 1000000 # Batch size of noised samples for robustness certification
    print_step_certify = 20 # Step size for showing certification progress
    if feature_noise_distribution == "gaussian" or certmethod != "certnid":
        feature_noise_distribution = "norm"
    set_seed(seed)
    print("\n***** Optimize noise *****")
    indir = os.path.join(indir,"cade-{}".format(dataset))
    param = torch.load(os.path.join(indir, f"cadecheckpoint-param-{dataset}-42"))
    encoder = torch.load(os.path.join(indir, f"cadecheckpoint-encoder-{dataset}-42"))
    detector = Detector(encoder.to(device), param["centroid"].to(device), param["dis_median"].to(device), param["mad"].to(device), mad_threshold, device)

    noise_generator = create_noise_generator("gaussian",d,device)

    criterion_shape = loss_functions["cade"](lambda_shape, mad_threshold)

    x_train, _, num_classes_train, class_map = get_dataset(dataset, "train")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    dataset1 = TensorDataset(x_train)
    data_loader = DataLoader(dataset1, batch_size=batch_size_iteration_certify, shuffle=False)
    x_train_c = torch.tensor([], dtype=torch.float32)
    noiseop_dir = "/home/huan1932/CertNID/resultsCertNID/noiseoptimizer"
    imagedir = "/home/huan1932/CertNID/resultsCertNID/plt"
    print()
    for (X,) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)
        closest_class = detector.closest_class(X)
        x_train_c = torch.cat([x_train_c, X[(closest_class == certify_class), :].detach().cpu()], 0)
    if (certmethod == "certnid" or certmethod == "certnid_gaussian_estimate" or certmethod == "bars") and (os.path.exists(os.path.join(noiseop_dir, "checkpoint-distribution-transformercade{}{}".format(feature_noise_distribution,0))) or os.path.exists(os.path.join(noiseop_dir, "tcade{}{}".format(feature_noise_distribution,0)))):
        optimizing_noise(
            x_train_c,
            0, # Certifying known class
            detector,
            noise_generator,
            criterion_shape,
            learning_rate_shape,
            nt_shape,
            num_epochs_shape,
            d,
            num_classes_certify,
            n0,
            n,
            alpha,
            init_step_size_scale,
            init_ptb_t_scale,
            decay_factor_scale,
            max_decay_scale,
            max_iter_scale,
            batch_size_iteration_certify,
            batch_size_memory_certify,
            print_step_certify,
            noiseop_dir,
            "cade",
            feature_noise_distribution)
        
    if (certmethod == "certnid" or certmethod == "certnid_gaussian_estimate" or certmethod == "bars"):
        noise_generator.distribution_transformer = torch.load(os.path.join(noiseop_dir, "checkpoint-distribution-transformercade{}{}".format(feature_noise_distribution,0))).to(device)
        # with open('data.pkl', 'rb') as file:
        #     a = pickle.load(file)
        r = open(os.path.join(noiseop_dir, "tcade{}{}".format(feature_noise_distribution,0)), "r")
        t = float(r.readline())
        r.close()
    print("\n***** Certify robustness *****")

    x_test, _, num_classes_train, class_map, y_test_drift =get_dataset(dataset, "test")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    dataset2 = TensorDataset(x_test, y_test_drift)
    data_loader = DataLoader(dataset2, batch_size=batch_size_iteration_certify, shuffle=False)
    x_test_c = torch.tensor([], dtype=torch.float32)
    y_test_drift_c = torch.tensor([], dtype=torch.long)
    for (X, y) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)
        y = y.to(device)
        closest_class = detector.closest_class(X)

        x_test_c = torch.cat([x_test_c, X[(closest_class == certify_class), :].detach().cpu()], 0)
        y_test_drift_c = torch.cat([y_test_drift_c, y[(closest_class == certify_class)].detach().cpu()], 0)

    idx = torch.arange(x_test_c.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_certify, idx.shape[0])])[0]
    x_test_c, y_test_drift_c = x_test_c[idx, :], y_test_drift_c[idx]

    dataset3 = TensorDataset(x_test_c, y_test_drift_c)
    data_loader = DataLoader(dataset3, batch_size=batch_size_iteration_certify, shuffle=False)

    detector.eval()
    
    cA_record = np.array([], dtype=np.int64)
   
    
    robust_radius_record = np.array([], dtype=np.float32)
    
    label_record = np.array([], dtype=np.int64)
    if certmethod == "certnid":
        smoothed_classifier = Smooth2(detector, d, num_classes_certify,noise_generator, noise_generator, device,feature_noise_distribution)
        if norm == "l2":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t, norm, alpha, batch_size_memory_certify)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            # print("XXX method-certify XX-L2-Mean Robustness Radius with first order: %.6e" % (mean_robust_radius))
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")

            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}{}class{}".format(dataset,certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        elif norm == "l1":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t, norm, alpha, batch_size_memory_certify)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)

            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        elif norm == "linf":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t, norm, alpha, batch_size_memory_certify)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
    elif certmethod == "certnid_gaussian_estimate":
        smoothed_classifier = Smooth2(detector, d, num_classes_certify,noise_generator, noise_generator, device,"gaussian")
        if norm == "l2":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t, norm, alpha, batch_size_memory_certify,True)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            # print("Mean Robustness L2Radius with first order: %.6e" % (mean_robust_radius))
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}{}class{}".format(dataset,certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        elif norm == "l1":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t, norm, alpha, batch_size_memory_certify,True)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        elif norm == "linf":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t, norm, alpha, batch_size_memory_certify,True)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
    elif certmethod == "vanilla_random":
        smoothed_classifier = Smooth1(detector, d, device=device)
        torch.set_grad_enabled(False)
        for X, y in tqdm(data_loader, desc="Certify"):
            X = X.to(device)

            
            cA, robust_radius = smoothed_classifier.certify(X, n0, n,alpha, batch_size_memory_certify)
        
        
            cA_record =np.concatenate([cA_record, cA], 0)
            
            
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

            label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

        torch.set_grad_enabled(True)
        torch.cuda.empty_cache()

        mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
    
        # print("Mean Robustness L2Radius: %.6e" % (mean_robust_radius))
        print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
        
        save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
        if not os.path.exists(save_dir_certify):                
            os.makedirs(save_dir_certify)

        w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}{}class{}".format(dataset,certmethod,seed,certify_class)), "w")
        w.write(str(mean_robust_radius))
        
        w.close()

        max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
        robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
        certified_accuracy_plot = np.array([], dtype=np.float32)
        for r in robust_radius_plot:
            certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
        
        plt.figure(1)
        plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
        plt.legend()
        plt.ylim((0, 1))
        plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
        plt.tick_params(labelsize=14)
        plt.xlabel("Robustness Radius", fontsize=16)
        plt.ylabel("Certified Accuracy", fontsize=16)
        class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
        plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
        plt.close()
        
    elif certmethod == "random_first_order":
        smoothed_classifier = Smooth1(detector, d, device=device)
        if norm == "l2":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certifywithfirst(X, n0, n, alpha, batch_size_memory_certify,norm)
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            # print("Mean Robustness L2Radius with first order: %.6e" % (mean_robust_radius))
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)

            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}{}class{}".format(dataset,certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        elif norm == "l1":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certifywithfirst(X, n0, n, alpha, batch_size_memory_certify,norm)
            
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)

            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()

            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        elif norm == "linf":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

                
                cA, robust_radius = smoothed_classifier.certifywithfirst(X, n0, n, alpha, batch_size_memory_certify,norm)
            
            
            
                cA_record =np.concatenate([cA_record, cA], 0)
                
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/cade/{}/{}/{}".format(purpose,dataset,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
            plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
            plt.close()
        
    elif certmethod == "bars":
        smoothed_classifier = Smooth2(detector, d, num_classes_certify,noise_generator, noise_generator, device,feature_noise_distribution)
        torch.set_grad_enabled(False)
        for X, y in tqdm(data_loader, desc="Certify"):
            X = X.to(device)

            
            cA, _,robust_radius = smoothed_classifier.bars_certify(X, n0, n,t, alpha, batch_size_memory_certify)
        
        
            cA_record =np.concatenate([cA_record, cA], 0)
            
            
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)

            label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

        torch.set_grad_enabled(True)
        torch.cuda.empty_cache()

        mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
    
        # print("Mean Robustness L2Radius with first order: %.6e" % (mean_robust_radius))
        print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
        

        w = open(os.path.join(save_dir_certify, "certification_resultscade{}{}{}class{}".format(dataset,certmethod,seed,certify_class)), "w")
        w.write(str(mean_robust_radius))
        
        w.close()

        max_robust_radius = {"newinfiltration-0": 5, "newinfiltration-1": 4, "newinfiltration-2": 4, "newhulk-2": 4,"newhulk-0": 4, "newhulk-1": 4}
        robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
        certified_accuracy_plot = np.array([], dtype=np.float32)
        for r in robust_radius_plot:
            certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
        
        plt.figure(1)
        plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
        plt.legend()
        plt.ylim((0, 1))
        plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
        plt.tick_params(labelsize=14)
        plt.xlabel("Robustness Radius", fontsize=16)
        plt.ylabel("Certified Accuracy", fontsize=16)
        class_name = {"newinfiltration-0": "benign", "newinfiltration-1": "ssh-bruteforce", "newinfiltration-2": "dos-hulk", "newhulk-2": "infiltration","newhulk-0":"benign","newhulk-1":"ssh-bruteforce"}
        plt.title("CADE %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir_certify, "certified_accuracy_robustness_radius_curve{}-{}.png".format(dataset,certify_class)))
        plt.close()
        

def featureimportance(dataset,device,batch_size=512):
    if dataset == 'ImageNet':
        data = get_dataset(dataset, "train")
    elif dataset == "ids18":
        indir = os.path.join(indir,"acid")
        data,y = get_dataset(dataset, "train")[0].to(device),get_dataset(dataset, "train")[1].to(device)
        encoder_list = torch.load(os.path.join(indir, "acidunnormcheckpoint-encoder_listnew"))
        encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
        kernel_net_list = torch.load(os.path.join(indir, "acidunnormcheckpoint-kernel_net_listnew"))
        kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list])
        classifier = AdaptiveClusteringClassifier(encoder_list, kernel_net_list, device)
        classifier.eval()
        unique_classes = torch.unique(y)

        # Create a dictionary to store TensorDatasets for each group
        grouped_datasets = {}

        # Loop through each unique class label and create a subset
        for class_label in unique_classes:
            # Create a mask to filter samples of a specific class
            mask = (y == class_label)
            
            # Use the mask to select samples of the current class
            x_group = data[mask]
            y_group = y[mask]
            
            # Create a TensorDataset for the current group
            dataset = TensorDataset(x_group, y_group)
            
            # Store the dataset in the dictionary with the class label as the key
            grouped_datasets[class_label.item()] = dataset

        # Now, grouped_datasets is a dictionary where each key is a class label,
        # and the corresponding value is a TensorDataset containing samples of that class.
        grads = {}  
        for key,value in grouped_datasets.items():
            summ = np.zeros((1,data.size(1)))
            dataloader = DataLoader(value,batch_size=batch_size, shuffle=False)
            for (X,y) in dataloader:
                    
                integrated_gradients = Saliency(classifier)
                attributions_ig = integrated_gradients.attribute(X, target=y)
                summ += np.sum(np.abs(attributions_ig.cpu().numpy()),axis=0)
            summ /= len(dataloader.dataset)
            output = summ
            grads[key] = output
        return grads
    elif dataset == "newhulk" or "newinfiltration":
        data,y,_,_= get_dataset(dataset, "train")
        indir = os.path.join(indir,"cade-{}".format(dataset))
        param = torch.load(os.path.join(indir, "cadecheckpoint-param"+dataset))
        encoder = torch.load(os.path.join(indir, "cadecheckpoint-encoder"+dataset))
        detector = Detector(encoder.to(device), param["centroid"].to(device), param["dis_median"].to(device), param["mad"].to(device), 3.5, device)
        detector.eval()
        unique_classes = torch.unique(y)

        # Create a dictionary to store TensorDatasets for each group
        grouped_datasets = {}

        # Loop through each unique class label and create a subset
        for class_label in unique_classes:
            # Create a mask to filter samples of a specific class
            mask = (y == class_label)
            
            # Use the mask to select samples of the current class
            x_group = data[mask]
            y_group = y[mask]
            
            # Create a TensorDataset for the current group
            dataset = TensorDataset(x_group, y_group)
            
            # Store the dataset in the dictionary with the class label as the key
            grouped_datasets[class_label.item()] = dataset

        # Now, grouped_datasets is a dictionary where each key is a class label,
        # and the corresponding value is a TensorDataset containing samples of that class.
        grads = {}  
        for key,value in grouped_datasets.items():
            summ = np.zeros((1,data.size(1)))
            dataloader = DataLoader(value,batch_size=batch_size, shuffle=False)
            for (X,y) in dataloader:
                    
                integrated_gradients = Saliency(detector.closest_class())
                attributions_ig = integrated_gradients.attribute(X, target=y)
                summ += np.sum(np.abs(attributions_ig.cpu().numpy()),axis=0)
            summ /= len(dataloader.dataset)
            output = summ
            grads[key] = output
        return grads
def acidcertify(dataset,certify_class,feature_noise_distribution,save_dir_certify,norm,certmethod,seed,device,purpose):
    # The certified class. e.g., Benign(0), FTP-Bruteforce(1), DDoS-HOIC(2), Botnet-Zeus&Ares(3)
    indir = "/home/huan1932/CertNID/resultsCertNID/trained-model"
    learning_rate_shape = 1e-2 # Learning rate for optimzing noise shape
    nt_shape = 1000 # Number of noised samples for optimzing noise shape
    lambda_shape = 1e-2 # Regularizer weight
    num_epochs_shape = 10 # Number of epochs for optimzing noise shape
    x = get_dataset(dataset,'train')[0]
    d = x.size(1) # Number of feature dimensions
    num_classes_certify = get_num_classes(dataset) # Number of certified classes
    n0 = 100 # Number of noised samples for identify cA
    n = 10000 # Number of noised samples for estimate pA
    alpha = 1e-3 # Failure probability
    init_step_size_scale = 5e-2 # Initial update step size of t for optimzing noise scale
    init_ptb_t_scale = 1e-2 # Initial perturbation of t for optimzing noise scale
    decay_factor_scale = 0.5 # Decay factor for optimzing noise scale
    max_decay_scale = 6 # Maximum decay times for optimzing noise scale
    max_iter_scale = 100 # Maximum iteration times for optimzing noise scale
    batch_size_iteration_certify = 128 # Batch size of certified samples for robustness certification
    batch_size_memory_certify = 1000000 # Batch size of noised samples for robustness certification
    print_step_certify = 20 # Step size for showing certification progress
    
    if feature_noise_distribution == "gaussian" or certmethod != "certnid":
        feature_noise_distribution = "norm"
    set_seed(seed)
    print("\n***** Optimize noise *****")
    indir = os.path.join(indir,"acid")
    encoder_list = torch.load(os.path.join(indir, f"acidunnormcheckpoint-encoder_listnew-{dataset}-42"))
    encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
    kernel_net_list = torch.load(os.path.join(indir, f"acidunnormcheckpoint-kernel_net_listnew-{dataset}-42"))
    kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list])
    classifier = AdaptiveClusteringClassifier(encoder_list, kernel_net_list, device)

    noise_generator = create_noise_generator("gaussian",d,device)
    criterion_shape = loss_functions["acid"](lambda_shape, certify_class)

    x_train = get_dataset(dataset, "train")[0]
    
    dataset1 = TensorDataset(x_train)
    data_loader = DataLoader(dataset1, batch_size=batch_size_iteration_certify, shuffle=True)
    x_train_c = torch.tensor([], dtype=torch.float32)
    for (X,) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)

        pred = classifier(X)

        x_train_c = torch.cat([x_train_c, X[(pred == certify_class), :].detach().cpu()], 0)
    noiseop_dir = "/home/huan1932/CertNID/resultsCertNID/noiseoptimizer"
    imagedir = "/home/huan1932/CertNID/resultsCertNID/plt"
    if (certmethod == "certnid" or certmethod == "certnid_gaussian_estimate" or certmethod == "bars") and ( os.path.exists(os.path.join(noiseop_dir, "checkpoint-distribution-transformeracid{}{}".format(feature_noise_distribution,certify_class))) or os.path.exists(os.path.join(noiseop_dir, "tacid{}{}".format(feature_noise_distribution,certify_class)))):
        optimizing_noise(
            x_train_c,
            certify_class,
            classifier,
            noise_generator,
            criterion_shape,
            learning_rate_shape,
            nt_shape,
            num_epochs_shape,
            d,
            num_classes_certify,
            n0,
            n,
            alpha,
            init_step_size_scale,
            init_ptb_t_scale,
            decay_factor_scale,
            max_decay_scale,
            max_iter_scale,
            batch_size_iteration_certify,
            batch_size_memory_certify,
            print_step_certify,
            noiseop_dir,
            "acid",
            feature_noise_distribution)
    if (certmethod == "certnid" or certmethod == "certnid_gaussian_estimate" or certmethod == "bars"):  
        noise_generator.distribution_transformer = torch.load(os.path.join(noiseop_dir, "checkpoint-distribution-transformeracid{}{}".format(feature_noise_distribution,certify_class))).to(device)
        # with open('data.pkl', 'rb') as file:
        #     a = pickle.load(file)
        r = open(os.path.join(noiseop_dir, "tacid{}{}".format(feature_noise_distribution,certify_class)), "r")
        t = float(r.readline())
        r.close()
    
    print("\n***** Certify robustness *****")

    x_test, y_test = get_dataset(dataset, "test")[0],get_dataset(dataset, "test")[1]

    dataset2 = TensorDataset(x_test, y_test)
    data_loader = DataLoader(dataset2, batch_size=batch_size_iteration_certify, shuffle=False)
    x_test_c = torch.tensor([], dtype=torch.float32)
    y_test_c = torch.tensor([], dtype=torch.long)
    for (X, y) in tqdm(data_loader, desc="Select class %d" % (certify_class)):
        X = X.to(device)
        y = y.to(device)
        pred = classifier(X)

        x_test_c = torch.cat([x_test_c, X[(pred == certify_class), :].detach().cpu()], 0)
        y_test_c = torch.cat([y_test_c, y[(pred == certify_class)].detach().cpu()], 0)

    idx = torch.arange(x_test_c.shape[0])
    idx = idx[torch.randperm(idx.size(0))]
    idx = torch.sort(idx[:min(num_samples_certify, idx.shape[0])])[0]
    x_test_c, y_test_c = x_test_c[idx, :], y_test_c[idx]

    dataset3 = TensorDataset(x_test_c, y_test_c)
    data_loader = DataLoader(dataset3, batch_size=batch_size_iteration_certify, shuffle=False)

    classifier.eval()
    # a = featureimportance(dataset)
    # with open('data.pkl', 'wb') as file:
    #     pickle.dump(a, file)
    
    
    cA_record = np.array([], dtype=np.int64)
    
    
    robust_radius_record = np.array([], dtype=np.float32)
    
    label_record = np.array([], dtype=np.int64)

    if certmethod == "certnid":
        smoothed_classifier = Smooth2(classifier, d, num_classes_certify,noise_generator, noise_generator, device,feature_noise_distribution)
        if norm == "l2":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t,norm, alpha, batch_size_memory_certify)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
        elif norm == "l1":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t,norm, alpha, batch_size_memory_certify)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
        elif norm == "linf":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t,norm, alpha, batch_size_memory_certify)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
    elif certmethod == "certnid_gaussian_estimate":
        smoothed_classifier = Smooth2(classifier, d, num_classes_certify,noise_generator, noise_generator, device,"norm")
        if norm == "l2":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t,norm, alpha, batch_size_memory_certify)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
        elif norm == "l1":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t,norm, alpha, batch_size_memory_certify)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
        elif norm == "linf":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certify(X, n0, n,t,norm, alpha, batch_size_memory_certify)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
    elif certmethod == "vanilla_random":
        smoothed_classifier = Smooth1(classifier, num_classes_certify,device)
    
        torch.set_grad_enabled(False)
        for X, y in tqdm(data_loader, desc="Certify"):
            X = X.to(device)


            cA, robust_radius = smoothed_classifier.certify(X, n0, n,alpha, batch_size_memory_certify)
            
            cA_record = np.concatenate([cA_record, cA], 0)
            
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
            
            label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

        torch.set_grad_enabled(True)
        torch.cuda.empty_cache()

        mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
    
        print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
        save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
        if not os.path.exists(save_dir_certify):                
            os.makedirs(save_dir_certify)
        # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
        w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
        w.write(str(mean_robust_radius))
    
        w.close()
        max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
        robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
        certified_accuracy_plot = np.array([], dtype=np.float32)
        for r in robust_radius_plot:
            certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
        
        plt.figure(1)
        plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
        plt.legend()
        plt.ylim((0, 1))
        plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
        plt.tick_params(labelsize=14)
        plt.xlabel("Robustness Radius", fontsize=16)
        plt.ylabel("Certified Accuracy", fontsize=16)
        class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
        plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
        plt.close()
    
    elif certmethod == "random_first_order":
        smoothed_classifier = Smooth1(classifier, num_classes_certify,device)
    
        if norm == "l2":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certifywithfirst(X, n0, n,alpha, batch_size_memory_certify,norm)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
        elif norm == "l1":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certifywithfirst(X, n0, n,alpha, batch_size_memory_certify,norm)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
        elif norm == "linf":
            torch.set_grad_enabled(False)
            for X, y in tqdm(data_loader, desc="Certify"):
                X = X.to(device)

    
                cA, robust_radius = smoothed_classifier.certifywithfirst(X, n0, n,alpha, batch_size_memory_certify,norm)
                
                cA_record = np.concatenate([cA_record, cA], 0)
                
                robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
                
                label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

            torch.set_grad_enabled(True)
            torch.cuda.empty_cache()

            mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
        
            print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
            save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
            if not os.path.exists(save_dir_certify):                
                os.makedirs(save_dir_certify)
            # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
            w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)+norm), "w")
            w.write(str(mean_robust_radius))
        
            w.close()
            max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
            robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
            certified_accuracy_plot = np.array([], dtype=np.float32)
            for r in robust_radius_plot:
                certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
            
            plt.figure(1)
            plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
            plt.legend()
            plt.ylim((0, 1))
            plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
            plt.tick_params(labelsize=14)
            plt.xlabel("Robustness Radius", fontsize=16)
            plt.ylabel("Certified Accuracy", fontsize=16)
            class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
            plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
            plt.close()
    elif certmethod == "bars":
        smoothed_classifier = Smooth2(classifier,d, num_classes_certify, noise_generator,noise_generator,device,feature_noise_distribution)
    
        torch.set_grad_enabled(False)
        for X, y in tqdm(data_loader, desc="Certify"):
            X = X.to(device)


            cA, _,robust_radius  = smoothed_classifier.bars_certify(X, n0, n,t,alpha, batch_size_memory_certify,feature_noise_distribution)
            
            cA_record = np.concatenate([cA_record, cA], 0)
            
            robust_radius_record = np.concatenate([robust_radius_record, robust_radius], 0)
            
            label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

        torch.set_grad_enabled(True)
        torch.cuda.empty_cache()

        mean_robust_radius = calc_metrics_certify(label_record, cA_record, robust_radius_record)
    
        print(f"{certmethod}-certifyclass-{certify_class}-{norm}-{seed}-Mean Robustness Radius: {mean_robust_radius:.6e}")
        save_dir_certify = os.path.join(save_dir_certify,"table/{}/acid/{}/{}".format(purpose,certmethod,certify_class))
        if not os.path.exists(save_dir_certify):                
            os.makedirs(save_dir_certify)
        # print("Mean Robustness Radius with first order linf: %.6e" % (mean_robust_radius3))
        w = open(os.path.join(save_dir_certify, "certification_resultsacid{}{}class{}".format(certmethod,seed,certify_class)), "w")
        w.write(str(mean_robust_radius))
    
        w.close()
        max_robust_radius = {"ids18-0": 3.5, "ids18-1": 3, "ids18-2": 3.5, "ids18-3": 5}
        robust_radius_plot = np.arange(0, max_robust_radius["%s-%d" % (dataset, certify_class)], max_robust_radius["%s-%d" % (dataset, certify_class)] * 1e-3)
        certified_accuracy_plot = np.array([], dtype=np.float32)
        for r in robust_radius_plot:
            certified_accuracy_plot = np.append(certified_accuracy_plot, np.mean((label_record == cA_record) & (robust_radius_record > r)))
        
        plt.figure(1)
        plt.plot(robust_radius_plot, certified_accuracy_plot,color = "red",label = "L2")
        plt.legend()
        plt.ylim((0, 1))
        plt.xlim((0, max_robust_radius["%s-%d" % (dataset, certify_class)]))
        plt.tick_params(labelsize=14)
        plt.xlabel("Robustness Radius", fontsize=16)
        plt.ylabel("Certified Accuracy", fontsize=16)
        class_name = {"ids18-0": "benign", "ids18-1": "ftp-bruteforce", "ids18-2": "ddos-hoic", "ids18-3": "botnet-zeus&ares"}
        plt.title("ACID %s" % (class_name["%s-%d" % (dataset, certify_class)]), fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(imagedir, "certified_accuracy_robustness_radius_curve{}lp.png".format(certify_class)))
        plt.close()
        
        
        