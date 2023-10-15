import numpy as np
import torch
from tqdm import tqdm

import torch.nn as nn

  
import os                  
import torch                  
from torch.utils.data import DataLoader, TensorDataset                    
from dataset.datasets import *                                    
import torch.optim as optim   
from classifier.autoencoder import Autoeconder, Loss_Contrastive
from classifier.detect import calc_centroid_dis_median_mad, Detector
from train_and_evaluate.train_utils import *
import random
import sys
sys.path.append("/home/huan1932/CertNID/classifier/")
from classifier.subnet import AdaptiveClustering, Loss_AdaptiveClustering
from classifier.classifier import AdaptiveClusteringClassifier
sys.path.append('/home/huan1932/CertNID/bars/')
from certify.cert import *

ARCHITECTURES = ['acid',"cade"]

def get_architecture(arc:str, dataset: str,device) -> torch.nn.Module:
    
    if (dataset == "ids18" or dataset == "ids18_unnormalized")and arc == "acid":
        return AdaptiveClustering(get_dataset(dataset, "train")[0].shape[1], get_num_classes(dataset), device)
    elif (dataset == 'newhulk' or dataset == 'newinfiltration') and arc == 'cade':
        train, _, num_classes_train, _ = get_dataset(dataset, "train")
        return Autoeconder(train.size(1), num_classes_train, device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

def create_classifier(encoder_list: nn.ModuleList, kernel_net_list: nn.ModuleList,device):
    return AdaptiveClusteringClassifier(encoder_list, kernel_net_list, device)

def acidtrain(args,dataset,model,num_classes,learing_rate,batch,epoches,print_freq,outdir,device):
    
            
    print("\n***** Run training *****")
    print("Number of classes for classifier:", num_classes)

    ac = get_architecture(model, dataset,device)
    ac.to(device)
    x_train, y_train = get_dataset(dataset, "train")[:2]
    d = x_train.size(1)
    data_dataset = TensorDataset(x_train, y_train)
    data_loader = DataLoader(data_dataset, batch_size=batch, shuffle=True)

    opt = optim.Adam(ac.parameters(), lr=learing_rate)

    criterion = Loss_AdaptiveClustering(num_classes)

    loss_record = Tracker()
    loss_classification_record = Tracker()
    loss_clustering_close_record = Tracker()
    loss_clustering_dist_record = Tracker()
    ac.to(ac.device)
    ac.train()
    
    for epoch in range(epoches + 1):
        for i, (X, y) in zip(range(1, len(data_loader) + 1), data_loader):
            X, y = X.to(ac.device), y.to(ac.device)
            z, kernel_weight, o = ac(X)
            loss, loss_classification, loss_clustering_close, loss_clustering_dist = criterion(z, kernel_weight, o, y)
            
            if epoch > 0:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_record.update(loss.item())
            loss_classification_record.update(loss_classification.item())
            loss_clustering_close_record.update(loss_clustering_close.item())
            loss_clustering_dist_record.update(loss_clustering_dist.item())
            if i % print_freq == 0:
                print(("Batch: [%d/%d][%d/%d] | Loss: %.6f | " + \
                    "Loss classification: %.6f | " + \
                    "Loss clustering_close: %.6f | " + \
                    "Loss clustering_dist: %.6f") % ( \
                    epoch, epoches, i, len(data_loader), loss_record.val, \
                    loss_classification_record.val, \
                    loss_clustering_close_record.val, \
                    loss_clustering_dist_record.val))

        print(('Epoch: [%d/%d] | Loss (Avg): %.6f | ' + \
            'Loss classification (Avg): %.6f | ' + \
            'Loss clustering_close (Avg): %.6f | ' + \
            'Loss clustering_dist (Avg): %.6f') % ( \
            epoch, epoches, loss_record.avg, \
            loss_classification_record.avg, \
            loss_clustering_close_record.avg, \
            loss_clustering_dist_record.avg))

        loss_record.reset()
        loss_classification_record.reset()
        loss_clustering_close_record.reset()
        loss_clustering_dist_record.reset()
    outdirformodel = os.path.join(outdir,"trained-model/acid")
    print("outdirformodel:",outdirformodel)
    
    if not os.path.exists(outdirformodel):
        # os.makedirs(outformodel)
        os.mkdir(outdirformodel)
            
    torch.save(ac.encoder_list, os.path.join(outdirformodel, f"acidunnormcheckpoint-encoder_listnew-{dataset}-{args.seed}"    ))
    torch.save(ac.kernel_weight, os.path.join(outdirformodel, f"acidunnormcheckpoint-kernel_weightnew-{dataset}-{args.seed}"    ))
    torch.save(ac.kernel_net_list, os.path.join(outdirformodel, f"acidunnormcheckpoint-kernel_net_listnew-{dataset}-{args.seed}"    ))

def acidevaluate(args,dataset,num_classes,batch,outdir,device):
    print("\n***** Run evaluating *****")
    print("Number of classes for classifier:", num_classes)
    outdirformodel = os.path.join(outdir,"trained-model/acid")
    print("outformodel:",outdirformodel)
    
    if not os.path.exists(outdirformodel):
        # os.makedirs(outformodel)
        os.mkdir(outdirformodel)
        
            
    encoder_list = torch.load(os.path.join(outdirformodel, f"acidunnormcheckpoint-encoder_listnew-{dataset}-{args.seed}"))
    encoder_list = nn.ModuleList([e.to(device) for e in encoder_list])
    kernel_net_list = torch.load(os.path.join(outdirformodel, f"acidunnormcheckpoint-kernel_net_listnew-{dataset}-{args.seed}"))
    kernel_net_list = nn.ModuleList([n.to(device) for n in kernel_net_list])
    classifier = create_classifier(encoder_list, kernel_net_list,device)

    x_test, y_test = get_dataset(dataset, "test")[0],get_dataset(dataset, "test")[1]

    data_dataset = TensorDataset(x_test, y_test)
    data_loader = DataLoader(data_dataset, batch_size=batch, shuffle=False)
    label_record = np.array([], dtype=np.int64)
    pred_record = np.array([], dtype=np.int64)    
    classifier.eval()
    torch.set_grad_enabled(False)
    for X, y in tqdm(data_loader, desc="Evaluate"):
        X, y = X.to(classifier.device), y.to(classifier.device)
        pred = classifier(X)

        pred_record = np.concatenate([pred_record, pred.detach().cpu().numpy()], 0)
        label_record = np.concatenate([label_record, y.detach().cpu().numpy()], 0)

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    acc, m = calc_metrics_classifieracid(label_record, pred_record)
    print(('Accuracy: %.4f') % (acc))
    print("Confusion matrix (Row: ground truth class, Col: prediction class):")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if j < m.shape[1] - 1:
                print("%d " % (m[i, j]), end="")
            else:
                print("%d" % (m[i, j]))
                

def create_detector(encoder: nn.Module, centroid: torch.tensor, dis_median: torch.tensor, mad: torch.tensor,device,mad_threshold = 3.5):
    return Detector(encoder, centroid, dis_median, mad, mad_threshold, device)

def cadetrain(args, dataset,model,learing_rate,epoches,batch,print_freq,outdir,device,margin = 10):
    print("\n***** Run training *****")
    
    ae = get_architecture(model,dataset,device)
    ae.to(device)
    x_train, y_train,_,_ = get_dataset(dataset, "train")
    d = x_train.size(1)
    data_dataset = TensorDataset(x_train, y_train)
    data_loader = DataLoader(data_dataset, batch_size=batch, shuffle=True)
    opt = optim.Adam(ae.parameters(), lr=learing_rate)

    criterion = Loss_Contrastive(margin, 1e-1)

    _,_, num_classes_train, class_map = get_dataset(dataset, "train")
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))


    loss_record = Tracker()
    loss_reconstructed_record = Tracker()
    loss_contrastive_record = Tracker()
    z_record = torch.tensor([], dtype=torch.float32)
    label_class_record = torch.tensor([], dtype=torch.long)
    ae.train()
    for epoch in range(epoches + 1):
        for i, (X, y_class) in zip(range(1, len(data_loader) + 1), data_loader):
            X, y_class = X.to(device), y_class.to(device)

            z, x_reconstructed = ae(X)
            loss, loss_reconstructed, loss_contrastive = criterion(X, z, x_reconstructed, y_class)

            if epoch > 0:
                opt.zero_grad()
                loss.backward()
                opt.step()

            loss_record.update(loss.item())
            loss_reconstructed_record.update(loss_reconstructed.item())
            loss_contrastive_record.update(loss_contrastive.item())
            if i % print_freq == 0:
                print(("Batch: [%d/%d][%d/%d] | Loss: %.6f | " + \
                    "Loss reconstructed: %.6f | " + \
                    "Loss contrastive: %.6f") % ( \
                    epoch, epoches, i, len(data_loader), loss_record.val, \
                    loss_reconstructed_record.val, \
                    loss_contrastive_record.val))

            if epoch == print_freq:
                z, x_reconstructed = ae(X)

                z_record = torch.cat([z_record, z.detach().cpu()], 0)
                label_class_record = torch.cat([label_class_record, y_class.detach().cpu()], 0)

        print(("Epoch: [%d/%d] | Loss (Avg): %.6f | " + \
            "Loss reconstructed (Avg): %.6f | " + \
            "Loss contrastive (Avg): %.6f") % ( \
            epoch, print_freq, loss_record.avg, \
            loss_reconstructed_record.avg, \
            loss_contrastive_record.avg))

        loss_record.reset()
        loss_reconstructed_record.reset()
        loss_contrastive_record.reset()
        
    
    # outformodel = os.path.join(outdir,"trained-model/cade-{}".format(dataset))
    outformodel = os.path.join(outdir,f"trained-model/{model}-{dataset}")

    print("outformodel:",outformodel)
    
    if not os.path.exists(outformodel):
        # os.makedirs(outformodel)
        os.mkdir(outformodel)

    # torch.save(ae.encoder, os.path.join(outformodel, "cadecheckpoint-encoder"+dataset+str(np.random.seed)))
    # torch.save(ae.decoder, os.path.join(outformodel, "cadecheckpoint-decoder"+dataset+str(np.random.seed)))
    torch.save(ae.encoder, os.path.join(outformodel, f"cadecheckpoint-encoder-{dataset}-{args.seed}"   ))
    torch.save(ae.decoder, os.path.join(outformodel, f"cadecheckpoint-decoder-{dataset}-{args.seed}"   ))

    # raise Exception("maggie stop")
    centroid, dis_median, mad = calc_centroid_dis_median_mad(z_record, label_class_record)
    torch.save({
        "centroid": centroid,
        "dis_median": dis_median,
        "mad": mad
    }, os.path.join(outformodel, f"cadecheckpoint-param-{dataset}-{args.seed}"    ))
    
def evaluate(args,dataset,batch,outdir,device,mad):
    print("\n***** Run evaluating *****")
    outformodel = os.path.join(outdir,f"trained-model/{args.model}-{dataset}")

    print("outformodel:",outformodel)
    
    if not os.path.exists(outformodel):
        # os.makedirs(outformodel)
        os.mkdir(outformodel)
    param = torch.load(os.path.join(outformodel, f"cadecheckpoint-param-{dataset}-{args.seed}"))
    encoder = torch.load(os.path.join(outformodel, f"cadecheckpoint-encoder-{dataset}-{args.seed}"))
    detector = create_detector(encoder.to(device), param["centroid"].to(device), param["dis_median"].to(device), \
        param["mad"].to(device),device,mad_threshold=mad)

    x_test, y_test_class, num_classes_train, class_map, y_test_drift = get_dataset(dataset, "test")
    print(f'{dataset}-dataset:')
    print("Number of classes for training:", num_classes_train)
    print("Class Map from original class to new class (First %d classes are for training): %s" % (num_classes_train, str(class_map)))

    dataset = TensorDataset(x_test, y_test_class, y_test_drift)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False)

    closest_class_record = np.array([], dtype=np.int64)
    drift_record = np.array([], dtype=np.int64)
    label_class_record = np.array([], dtype=np.int64)
    label_drift_record = np.array([], dtype=np.int64)    
    detector.eval()
    torch.set_grad_enabled(False)
    for X, y_class, y_drift in tqdm(data_loader, desc="Evaluate"):
        X, y_class, y_drift = X.to(detector.device), y_class.to(detector.device), y_drift.to(detector.device)
        
        closest_class = detector.closest_class(X)
        drift = detector(X)

        closest_class_record = np.concatenate([closest_class_record, closest_class.detach().cpu().numpy()], 0)
        drift_record = np.concatenate([drift_record, drift.detach().cpu().numpy()], 0)
        label_class_record = np.concatenate([label_class_record, y_class.detach().cpu().numpy()], 0)
        label_drift_record = np.concatenate([label_drift_record, y_drift.detach().cpu().numpy()], 0)

    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()

    _, m_class = calc_metrics_classifier_class(label_class_record, closest_class_record)
    save_idx = np.array([], dtype=np.int64)

    for c in range(num_classes_train):
        save_idx = np.concatenate([save_idx, np.where(label_class_record == c)[0]], 0)
    acc_class, _ = calc_metrics_classifier_class(closest_class_record[save_idx], label_class_record[save_idx])

    print(('Accuracy (Class for training): %.4f') % (acc_class))
    print("Confusion matrix (Row: ground truth class, Col: prediction class):")
    for i in range(m_class.shape[0]):
        for j in range(m_class.shape[1]):
            if j < m_class.shape[1] - 1:
                print("%d " % (m_class[i, j]), end="")
            else:
                print("%d" % (m_class[i, j]))

    acc, p, r, f1 = calc_metrics_classifiercade(label_drift_record, drift_record)
    _, m_drift = calc_metrics_classifier_class(label_class_record, drift_record)
    m_drift = m_drift[:, :2]

    # print("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f" % (acc, p, r, f1))
    print("Drift Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f" % (acc, p, r, f1))

    print("Confusion matrix (Row: ground truth class, Col: drift result): ")
    for i in range(m_drift.shape[0]):
        for j in range(m_drift.shape[1]):
            if j < m_drift.shape[1] - 1:
                print("%d " % (m_drift[i, j]), end="")
            else:
                print("%d" % (m_drift[i, j]))