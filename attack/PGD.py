# from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import ProjectedGradientDescent
import numpy as np
from art.estimators.classification import PyTorchClassifier


def generate_atk(cleanx,model,criterion,optimizer,min:int,max:int,shape,number,eps=0.2,eps_step=0.1,max_iter= 20):

    classifier = PyTorchClassifier(
        model=model,
        clip_values=(min, max),
        loss=criterion,
        optimizer=optimizer,
        input_shape= shape,
        nb_classes=number,
    )
    
    attack = ProjectedGradientDescent(estimator=classifier, eps=eps,eps_step=eps_step,targeted=True,max_iter= max_iter)
    x_test_adv = attack.generate(x=cleanx,y=np.zeros((cleanx.shape[0],)))
    return x_test_adv
