import torch

def ComputeClassWeight(trainset_label_counts_list):

    # [(0, 49166), (1, 33760), (2, 8780), (3, 2488), (4, 75)]
    
    weight = []
    # for i in range(len(classlist))
    #     cla_weg[i] = classlist[1]

    # weight.append(c1_weg)
    # weight = torch.tensor([5, 5, 1, 1, 1]).float()

    return weight
