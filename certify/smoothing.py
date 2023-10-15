from scipy.stats import norm, binom_test
import numpy as np
from math import ceil, floor
from statsmodels.stats.proportion import proportion_confint
from typing import *
import numpy as np
from scipy.stats import norm, laplace,uniform
import time
from scipy import integrate
import torch
from certify.numberical import numerical_radius, calculate_radius
from certify.optimizer import Sigma_Adaptation


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        nA = counts_estimation[np.arange(cAHat.shape[0]), cAHat]
        pABar = self._lower_confidence_bound(nA, n, alpha/2)
        radius_norm = norm.ppf(pABar)
        idx_abstain = np.where(pABar < 0.5)[0]
        cAHat[idx_abstain] = self.ABSTAIN
        radius_norm[idx_abstain] = 0
        
        return cAHat, radius_norm

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]
    
    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [feature_size]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                dim = len(x.shape)
                dim = (1,) * dim
                batch = x.repeat((this_batch_size, *dim))
                
                noise = torch.randn_like(batch, device=batch.device) * self.sigma
                predictions = 0
                if len(x) == 3:
                    predictions = self.base_classifier((batch + noise))
                elif len(x) ==1 :
                    predictions = self.base_classifier((batch + noise).unsqueeze(1).unsqueeze(3))
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint( NA, N, alpha=2 * alpha, method="beta")[0] 

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]

# Certifying multiple samples in parallel
class Smooth2(Smooth):
    
    def __init__(self, base_classifier: torch.nn.Module, d: int, num_classes: int,sigma:object, noise_generator: object,device: object,distribution = "norm"):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes: the number of classes, Kitsune: 2, CADE: 2, ACID: 4
        :param noise_generator: optimized noise generator
        :param d: the number of feature vector dimensions
        :param device: cpu / cuda
        """
        self.base_classifier = base_classifier
        self.d = d
        self.num_classes = num_classes
        self.noise_generator = noise_generator
        self.sigma = sigma
        self.distribution = distribution
        self.device = device

        assert noise_generator.d == self.d
        assert noise_generator.device == self.device

        self.eps = 1e-16
    

    def _certify2(self, x: torch.tensor, n0: int, n: int, t: float, alpha: float, batch_size_memory: int) -> (np.ndarray, np.ndarray):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [batch_size_iteration x feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size_memory: maximum batch size in memory for parallel smoothing
        :return: (predicted class: np.ndarray, certified normalized robustness radius: np.ndarray)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise2(x, n0, t, batch_size_memory)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise2(x, n, t, batch_size_memory)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[np.arange(cAHat.shape[0]), cAHat]
        pABar = self._lower_confidence_bound2(nA, n, alpha/2)
        # use pA lower bound to calculate normalized robustness radius
        radius_norm = norm.ppf(pABar)
        radius_l1 = laplace.ppf(pABar,scale = 1)
        radius_li = uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).ppf(pABar)
        # when pABar < 0.5, abstain from making robustness certification
        idx_abstain = np.where(pABar < 0.5)[0]
        cAHat[idx_abstain] = self.ABSTAIN
        radius_norm[idx_abstain] = 0
        radius_l1[idx_abstain] = 0
        if self.distribution == "norm":
            return cAHat, radius_norm,radius_norm
        elif self.distribution == "uni":
            return cAHat, radius_li,radius_li
        elif self.distribution == "lap":
            return cAHat, radius_l1,radius_l1

    def certify(self, x: torch.tensor, n0: int, n: int, t,norm,alpha: float, batch_size: int,estimate = False) -> (int, float):
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        
        counts_selection = self._sample_noise3(x, n0, t,batch_size)
        # use these samples to take a guess at the top class
        
        cAHat = counts_selection.argmax(1)
        
        # draw more samples of f(x + epsilon)
        counts, maxC, meanC, meanl2, n1, n2 = self._sample_noise1storder(
            x, n, t, cAHat, batch_size)
        
        # use these samples to estimate a lower bound on pA
        nA = counts[np.arange(cAHat.shape[0]), cAHat]
        
        pABar = self._lower_confidence_bound2(nA, n, alpha/2)
        
        idx_abstain = np.where(pABar < 0.5)[0]
        
        cAHat[idx_abstain] = self.ABSTAIN
        if estimate == False:
            if norm =="l2":
            
                radiusL2list = []
                
                # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
                for i in range(x.size(0)):
                    maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                        maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                    
                    radiusL2 = calculate_radius(pABar[i], meanBarL2, distribution=self.distribution)
                    radiusL2list.append(radiusL2)
                    
    
                radiusL2list = np.array(radiusL2list)
                
                radiusL2list[idx_abstain] = 0
                
                
                return cAHat, radiusL2list
            elif norm =="l1":
            
                radiusL1list = []
                
                # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
                for i in range(x.size(0)):
                    maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                        maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                    
                    radius1 = calculate_radius(pABar[i], meanDir,distribution=self.distribution)
                    radiusL1list.append(radius1)
                    
                    
                radiusL1list = np.array(radiusL1list)
                
                radiusL1list[idx_abstain] = 0
                
                
                return cAHat, radiusL1list
            elif norm =="linf":
            
                radiusLInflist = []
                
                # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
                for i in range(x.size(0)):
                    maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                        maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                    
                    
                    radiusi = calculate_radius(pABar[i], maxDir,distribution=self.distribution)
                    radiusi /= np.sqrt(np.prod(x[i].shape))
                    
                    radiusLInflist.append(radiusi)
                radiusLInflist = np.array(radiusLInflist)
                
                radiusLInflist[idx_abstain] = 0
                return cAHat, radiusLInflist
        else:
            if norm =="l2":
            
                radiusL2list = []
                
                # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
                for i in range(x.size(0)):
                    maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                        maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                    
                    radiusL2 = calculate_radius(pABar[i], meanBarL2)
                    radiusL2list.append(radiusL2)
                    
    
                radiusL2list = np.array(radiusL2list)
                
                radiusL2list[idx_abstain] = 0
                
                
                return cAHat, radiusL2list
            elif norm =="l1":
            
                radiusL1list = []
                
                # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
                for i in range(x.size(0)):
                    maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                        maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                    
                    radius1 = calculate_radius(pABar[i], meanDir)
                    radiusL1list.append(radius1)
                    
                    
                radiusL1list = np.array(radiusL1list)
                
                radiusL1list[idx_abstain] = 0
                
                
                return cAHat, radiusL1list
            elif norm =="linf":
            
                radiusLInflist = []
                
                # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
                for i in range(x.size(0)):
                    maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                        maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                    
                    
                    radiusi = calculate_radius(pABar[i], maxDir)
                    radiusi /= np.sqrt(np.prod(x[i].shape))
                    
                    radiusLInflist.append(radiusi)
                radiusLInflist = np.array(radiusLInflist)
                
                radiusLInflist[idx_abstain] = 0
                return cAHat, radiusLInflist
                    
                    
            
                    
            
            
    
    def _count_arr2(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros((length, self.num_classes), dtype=int)
        arr = arr.reshape(-1, length).T
        for c in range(self.num_classes):
            counts[:, c] += np.array(arr == c, dtype=int).sum(1)
        return counts

    def _lower_confidence_bound2(self, NA: np.ndarray, N: int, alpha: float) -> np.ndarray:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes" for each example
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: an ndarray of lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return np.array([proportion_confint(NA_, N, alpha=2 * alpha, method="beta")[0] for NA_ in NA])
    def _sample_noise2(self, x: torch.tensor, num: int, t: float, batch_size_memory: int) -> np.ndarray:
            """ Sample the base classifier's prediction under noisy corruptions of the input x.

            :param x: the input [batch_size_iteration x feature_size]
            :param num: number of samples to collect
            :param t: scale factor
            :param batch_size_memory: maximum batch size in memory for parallel smoothing
            :return: an ndarray[int] of length num_classes containing the per-class counts
            """
            # batch size for iteration should be less than or equal to maximum batch size in memory
            assert x.size(0) <= batch_size_memory
            with torch.no_grad():
                counts = np.zeros((x.size(0), self.num_classes), dtype=int)
                while num > 0:
                    batch_size_per_example = min(floor(batch_size_memory / x.size(0)), num)
                    num -= batch_size_per_example

                    batch = x.repeat((batch_size_per_example, 1))
                    noise = self.noise_generator.sample_feat(x.size(0) * batch_size_per_example,self.distribution) * t
                    predictions = self.base_classifier(batch, noise)
                    counts += self._count_arr2(predictions.cpu().numpy(), x.size(0))

                return counts
    def _sample_noise3(self, x: torch.tensor, num: int,t, batch_size_memory: int) -> np.ndarray:
            """ Sample the base classifier's prediction under noisy corruptions of the input x.

            :param x: the input [batch_size_iteration x feature_size]
            :param num: number of samples to collect
            :param t: scale factor
            :param batch_size_memory: maximum batch size in memory for parallel smoothing
            :return: an ndarray[int] of length num_classes containing the per-class counts
            """
            # batch size for iteration should be less than or equal to maximum batch size in memory
            assert x.size(0) <= batch_size_memory
            with torch.no_grad():
                counts = np.zeros((x.size(0), self.num_classes), dtype=int)
                while num > 0:
                    batch_size_per_example = min(floor(batch_size_memory / x.size(0)), num)
                    num -= batch_size_per_example

                    batch = x.repeat((batch_size_per_example, 1))
                    noise = self.noise_generator.sample_feat(x.size(0) * batch_size_per_example,self.distribution) * t
                    predictions = self.base_classifier(batch, noise)
                    counts += self._count_arr2(predictions.cpu().numpy(), x.size(0))

                return counts

    def _sample_noise1storder(self, x: torch.tensor, num: int, t,pred_class: int, batch_size):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input 
        :param num: number of samples to collect
        :param batch_size:
        :param pred_class : the class expected to be predictied with the highest probability 
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        results_maxC = []
        results_meanC = []
        results_l2 = []
        results_n1 = []
        results_n2 = []
        counts = np.zeros((x.size(0), self.num_classes), dtype=int)
        with torch.no_grad():
            num1 = num
            num2 = num
            while num > 0:
                batch_size_per_example = min(floor(batch_size / x.size(0)), num)
                num -= batch_size_per_example

                batch = x.repeat((batch_size_per_example, 1))
                noise = self.noise_generator.sample_feat(x.size(0) * batch_size_per_example,self.distribution) * t
                # noise = torch.rand_like(batch)*sigma
                predictions = self.base_classifier(batch, noise)
                counts += self._count_arr2(predictions.cpu().numpy(), x.size(0))
            index = 0
            for sample in x:
                num1 = num2
                mean1 = torch.zeros_like(sample)
                mean2 = torch.zeros_like(sample)
                n1, n2 = 0,0
                batch_size = 500
                for i in range(ceil(num1 / batch_size)):
                    this_batch_size = min(batch_size, num1)
                    num1 -= this_batch_size

                    batch = sample.repeat((this_batch_size, 1))
                    noise = self.noise_generator.sample_feat(sample.unsqueeze(0).size(0) * this_batch_size,self.distribution) * t
                    # noise = self.sigma.sample_feat(sample.unsqueeze(0).size(0) * this_batch_size)
                    # noise = torch.rand_like(batch)*sigma
                    predictions = self.base_classifier(batch,noise)
                    predictions = (predictions == pred_class[index].item()).type(torch.double)
                    if i % 2 == 0:
                        mean1 += torch.einsum(
                            'ni,n->i', [noise, (predictions - 0.5).type(torch.FloatTensor).to(self.device)])
                        n1 += this_batch_size
                    else:
                        mean2 += torch.einsum(
                            'ni,n->i', [noise, (predictions - 0.5).type(torch.FloatTensor).to(self.device)])
                        n2 += this_batch_size
                # gradiednt estimation of the Fsmooth
                maxC = max(torch.max((mean1 + mean2)/(n1 + n2)).item(), -torch.min((mean1 + mean2)/(n1 + n2)).item())
                meanC = torch.abs((mean1 + mean2)/(n1 + n2)).sum().item()/np.sqrt(np.prod(sample.shape))
                meanl2 = torch.mul(mean1/n1, mean2/n2).sum().cpu().data.numpy()
                
                results_maxC.append(maxC)
                results_meanC.append(meanC)
                results_l2.append(meanl2)
                results_n1.append(n1)
                results_n2.append(n2)
                index += 1
            return counts, results_maxC, results_meanC, results_l2, results_n1, results_n2
    
    def _confidence_bound(self, maxC : float, meanC : float, meanl2, N1: int, N2: int, D: int, alpha: float) -> float:
        """ Returns a (1 - alpha) confidence bound on the norm values of the gradient

        These values are based on the derivations given in the paper.

        :param maxC: the empirical estimate of the infinity-norm of the gradient 
        :param meanC: the empirical estimate of the 1-norm of the gradient 
        :param N1: the number of total draws used to calculate an independent empirical estimate of the gradient
        :param N1: the number of total draws used to calculate a second independent empirical estimate of the gradient
        :param D: the dimensionality of the input space
        :param alpha: the confidence level
        :return: the relevant bounds on the infinity-norm, 1-norm, 2-norm
                of the gradient each of which holds true w.p at least (1 - alpha) over the samples
        """

        k = (0.25 + (3/np.sqrt(8*np.pi*np.e)))
        t0 = np.sqrt(2*(k/(N1 + N2))*np.log(4*D/alpha))
        maxDir = maxC + t0
        
        t0 = np.sqrt(2*k*(D*np.log(2) - np.log(alpha/2))/(N1 + N2))
        meanDir = meanC + t0
        
        t1 =  k*np.sqrt(-2*D*np.log(alpha/2)/(N1*N2))


        meanA = 3*meanl2 
        epsu = np.sqrt(-k*(N1 + N2)*np.log(alpha/2)/((meanA + t1)*2*N1*N2))
        meanL2 = np.sqrt(meanA + t1)/(np.sqrt(1 + epsu**2) - epsu)
        if ((meanA - t1) <= 0):
            meanOp = 0.0
        else:
            epsl = np.sqrt(-k*(N1 + N2)*np.log(alpha/2)/((meanA - t1)*2*N1*N2))
            meanOp = -np.sqrt(meanA - t1)/(np.sqrt(1 + epsl**2) + epsl)

        

        return maxDir, meanDir, meanL2, meanOp
    
    def bars_certify(self, x: torch.tensor, n0: int, n: int, t: float, alpha: float, batch_size_memory: int,distribution = "norm") -> (np.ndarray, np.ndarray, np.ndarray):
        """ 
        :param x: the input [batch_size_iteration x feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size_memory: maximum batch size in memory for parallel smoothing
        :return: (predicted class: np.ndarray, 
                 certified dimension-wise robustness radius vector (feature space): np.ndarray,
                 certified dimension-heterogeneous robustness radius (feature space): np.ndarray)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.distribution = distribution
        cAHat, radius_norm,radl1 = self._certify2(x, n0, n, t, alpha, batch_size_memory)
        radius_norm_dim = torch.tensor(radius_norm).unsqueeze(1).repeat((1, self.d)).to(self.device)
        radius_feat_dim = self.noise_generator.norm_to_feat(radius_norm_dim).cpu().numpy() * t
        radius_feat = radius_feat_dim.mean(1)
        radius_norm_dim1 = torch.tensor(radl1).unsqueeze(1).repeat((1, self.d)).to(self.device)
        radius_feat_dim1 = self.noise_generator.norm_to_feat(radius_norm_dim1).cpu().numpy() * t
        radius_feat1 = radius_feat_dim1.mean(1)
        return cAHat, radius_feat1, radius_norm

class Noise(object):
    def __init__(self, distribution_transformer: torch.nn.Module, d: int, device: object):
        """
        distribution_transformer: Distribution Transformer model
        d: the number of feature vector dimensions
        device: cpu / cuda
        """
        self.distribution_transformer = distribution_transformer
        self.d = d
        self.device = device

    def sample_norm(self, n: int,distribution = "norm") -> torch.tensor:
        if distribution == "norm":
            return torch.randn((n, self.d), device=self.device)
        elif distribution == "lap":
            return torch.distributions.Laplace(loc=0,scale=1).sample((n, self.d)).to(self.device)
        elif distribution == "uni":
            return torch.distributions.Uniform(-np.sqrt(3),2*np.sqrt(3)).sample((n, self.d)).to(self.device)
            

    def norm_to_feat(self, z: torch.tensor,s = False) -> torch.tensor:
        return self.distribution_transformer(z,s).to(self.device)

    def sample_feat(self, n: int,distribution = "norm") -> torch.tensor:
        z = self.sample_norm(n,distribution)
        return self.norm_to_feat(z)

    def get_weight(self) -> torch.tensor:
        return self.distribution_transformer.get_weight()
    
class Sigma(object):
    def __init__(self, d: int, device: object):
        """
        distribution_transformer: Distribution Transformer model
        d: the number of feature vector dimensions
        device: cpu / cuda
        """
        self.d = d
        self.device = device
        self.sigma = Sigma_Adaptation(d)
    def sample_norm(self, n: int) -> torch.tensor:
        return torch.randn((n, self.d), device=self.device)

    def norm_to_feat(self, z: torch.tensor) -> torch.tensor:
        return self.sigma(z).to(self.device)

    def sample_feat(self, n: int) -> torch.tensor:
        z = self.sample_norm(n)
        return self.norm_to_feat(z)

    def get_weight(self) -> torch.tensor:
        return self.sigma.get_weight()
    
    
    
class Smooth1(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int,device,sigma=1):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [feature_size]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        nA = counts_estimation[np.arange(cAHat.shape[0]), cAHat]
        pABar = self._lower_confidence_bound(nA, n, alpha/2)
        radius_norm = norm.ppf(pABar)*self.sigma
        idx_abstain = np.where(pABar < 0.5)[0]
        cAHat[idx_abstain] = self.ABSTAIN
        radius_norm[idx_abstain] = 0
        
        return cAHat, radius_norm

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]
    
    def _sample_noise(self, x: torch.tensor, num: int, batch_size_memory: int) -> np.ndarray:
            """ Sample the base classifier's prediction under noisy corruptions of the input x.

            :param x: the input [batch_size_iteration x feature_size]
            :param num: number of samples to collect
            :param t: scale factor
            :param batch_size_memory: maximum batch size in memory for parallel smoothing
            :return: an ndarray[int] of length num_classes containing the per-class counts
            """
            # batch size for iteration should be less than or equal to maximum batch size in memory
            assert x.size(0) <= batch_size_memory
            with torch.no_grad():
                counts = np.zeros((x.size(0), self.num_classes), dtype=int)
                while num > 0:
                    batch_size_per_example = min(floor(batch_size_memory / x.size(0)), num)
                    num -= batch_size_per_example

                    batch = x.repeat((batch_size_per_example, 1))
                    noise = torch.randn_like(batch, device=batch.device) * self.sigma
                    predictions = self.base_classifier(batch, noise)
                    counts += self._count_arr(predictions.cpu().numpy(), x.size(0))

                return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros((length, self.num_classes), dtype=int)
        arr = arr.reshape(-1, length).T
        for c in range(self.num_classes):
            counts[:, c] += np.array(arr == c, dtype=int).sum(1)
        return counts
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return np.array([proportion_confint(NA_, N, alpha=2 * alpha, method="beta")[0] for NA_ in NA])
    def _sample_noise1storder(self, x: torch.tensor, num: int, pred_class: int, batch_size):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input 
        :param num: number of samples to collect
        :param batch_size:
        :param pred_class : the class expected to be predictied with the highest probability 
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        results_maxC = []
        results_meanC = []
        results_l2 = []
        results_n1 = []
        results_n2 = []
        counts = np.zeros((x.size(0), self.num_classes), dtype=int)
        with torch.no_grad():
            num1 = num
            num2 = num
            while num > 0:
                batch_size_per_example = min(floor(batch_size / x.size(0)), num)
                num -= batch_size_per_example

                batch = x.repeat((batch_size_per_example, 1))
               
                noise = torch.rand_like(batch)*self.sigma
                predictions = self.base_classifier(batch, noise)
                counts += self._count_arr(predictions.cpu().numpy(), x.size(0))
            index = 0
            for sample in x:
                num1 = num2
                mean1 = torch.zeros_like(sample)
                mean2 = torch.zeros_like(sample)
                n1, n2 = 0,0
                batch_size = 500
                for i in range(ceil(num1 / batch_size)):
                    this_batch_size = min(batch_size, num1)
                    num1 -= this_batch_size

                    batch = sample.repeat((this_batch_size, 1))
                    
                    noise = torch.rand_like(batch)*self.sigma
                    predictions = self.base_classifier(batch,noise)
                    predictions = (predictions == pred_class[index].item()).type(torch.double)
                    if i % 2 == 0:
                        mean1 += torch.einsum(
                            'ni,n->i', [noise, (predictions - 0.5).type(torch.FloatTensor).to(self.device)])
                        n1 += this_batch_size
                    else:
                        mean2 += torch.einsum(
                            'ni,n->i', [noise, (predictions - 0.5).type(torch.FloatTensor).to(self.device)])
                        n2 += this_batch_size
                maxC = max(torch.max((mean1 + mean2)/(n1 + n2)).item(), -torch.min((mean1 + mean2)/(n1 + n2)).item())
                meanC = torch.abs((mean1 + mean2)/(n1 + n2)).sum().item()/np.sqrt(np.prod(sample.shape))
                meanl2 = torch.mul(mean1/n1, mean2/n2).sum().cpu().data.numpy()/(self.sigma**2)
                results_maxC.append(maxC/self.sigma)
                results_meanC.append(meanC/self.sigma)
                results_l2.append(meanl2)
                results_n1.append(n1)
                results_n2.append(n2)
                index += 1
            return counts, results_maxC, results_meanC, results_l2, results_n1, results_n2
    
    def _confidence_bound(self, maxC : float, meanC : float, meanl2, N1: int, N2: int, D: int, alpha: float) -> float:
        """ Returns a (1 - alpha) confidence bound on the norm values of the gradient

        These values are based on the derivations given in the paper.

        :param maxC: the empirical estimate of the infinity-norm of the gradient 
        :param meanC: the empirical estimate of the 1-norm of the gradient 
        :param N1: the number of total draws used to calculate an independent empirical estimate of the gradient
        :param N1: the number of total draws used to calculate a second independent empirical estimate of the gradient
        :param D: the dimensionality of the input space
        :param alpha: the confidence level
        :return: the relevant bounds on the infinity-norm, 1-norm, 2-norm
                of the gradient each of which holds true w.p at least (1 - alpha) over the samples
        """

        k = (0.25 + (3/np.sqrt(8*np.pi*np.e)))
        t0 = np.sqrt(2*(k/(N1 + N2))*np.log(4*D/alpha))
        maxDir = maxC + t0
        
        t0 = np.sqrt(2*k*(D*np.log(2) - np.log(alpha/2))/(N1 + N2))
        meanDir = meanC + t0
        
        t1 =  k*np.sqrt(-2*D*np.log(alpha/2)/(N1*N2))


        meanA = meanl2 
        epsu = np.sqrt(-k*(N1 + N2)*np.log(alpha/2)/((meanA + t1)*2*N1*N2))
        meanL2 = np.sqrt(meanA + t1)/(np.sqrt(1 + epsu**2) - epsu)
        if ((meanA - t1) <= 0):
            meanOp = 0.0
        else:
            epsl = np.sqrt(-k*(N1 + N2)*np.log(alpha/2)/((meanA - t1)*2*N1*N2))
            meanOp = -np.sqrt(meanA - t1)/(np.sqrt(1 + epsl**2) + epsl)

        

        return maxDir, meanDir, meanL2, meanOp
    def certifywithfirst(self, x: torch.tensor, n0: int, n: int,alpha: float, batch_size: int,norm = "l2",distribution = "norm") -> (int, float):
        
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0,batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax(1)
        # draw more samples of f(x + epsilon)
        counts, maxC, meanC, meanl2, n1, n2 = self._sample_noise1storder(
            x, n,cAHat, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts[np.arange(cAHat.shape[0]), cAHat]
        pABar = self._lower_confidence_bound(nA, n, alpha/2)
        idx_abstain = np.where(pABar < 0.5)[0]
        cAHat[idx_abstain] = self.ABSTAIN
        if norm =="l2":
        
            radiusL2list = []
            
            # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
            for i in range(x.size(0)):
                maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                    maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                
                radiusL2 = calculate_radius(pABar[i], meanBarL2,distribution)
                radiusL2list.append(radiusL2)
                

            radiusL2list = np.array(radiusL2list)
            
            radiusL2list[idx_abstain] = 0
            
            
            return cAHat, radiusL2list
        elif norm =="l1":
        
            radiusL1list = []
            
            # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
            for i in range(x.size(0)):
                maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                    maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                
                radius1 = calculate_radius(pABar[i], meanDir,distribution)
                radiusL1list.append(radius1)
                
                
            radiusL1list = np.array(radiusL1list)
            
            radiusL1list[idx_abstain] = 0
            
            
            return cAHat, radiusL1list
        elif norm =="linf":
        
            radiusLInflist = []
            
            # use these samples to get statistical estimates on vaiours norms of gradient of pA wrt x
            for i in range(x.size(0)):
                maxDir, meanDir, meanBarL2, meanBarOp = self._confidence_bound(
                    maxC[i], meanC[i], meanl2[i], n1[i], n2[i], np.prod(x[i].shape), alpha/2)
                
                
                radiusi = calculate_radius(pABar[i], maxDir,distribution)
                radiusi /= np.sqrt(np.prod(x[i].shape))
                
                radiusLInflist.append(radiusi)
            radiusLInflist = np.array(radiusLInflist)
            
            radiusLInflist[idx_abstain] = 0
            return cAHat, radiusLInflist