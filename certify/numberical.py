import numpy as np
from scipy.stats import norm,laplace,uniform
from scipy import integrate
from scipy.special import roots_hermite as herm


roots, weights = herm(1000)
roots = np.array([np.sqrt(2)*x for x in roots])
weights = np.array(weights)/np.sqrt(np.pi)

def meta_calc_fast_p(r,a,b,distribution = "norm"):
    x = -(np.exp(r*roots) - a*roots)/b
    if distribution == "norm":
        return lambda c: np.dot(weights,norm.cdf(x + (c/b)))
    elif distribution == "lap":
        return lambda c: np.dot(weights,laplace.cdf(x + (c/b)))
    elif distribution == "uniform":
        return lambda c: np.dot(weights,uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(x + (c/b)))

def calc_fast_mx(r,a,b,c,distribution = "norm"):
    if distribution == "norm":
        return np.dot(weights,roots*norm.cdf(-(np.exp(r*roots) - (a*roots + c))/b))
    elif distribution == "uni":
        return np.dot(weights,roots*uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(-(np.exp(r*roots) - (a*roots + c))/b))
    elif distribution == "lap":
        return np.dot(weights,roots*laplace.cdf(-(np.exp(r*roots) - (a*roots + c))/b))

def calc_fast_my(r,a,b,c,distribution = "norm"):
    if distribution == "norm" :
        return np.dot(weights,norm.pdf(-(np.exp(r*roots) - (a*roots + c))/b))
    elif distribution == "uni" :
        return np.dot(weights,uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(-(np.exp(r*roots) - (a*roots + c))/b))
    elif distribution == "lap" :
        return np.dot(weights,laplace.pdf(-(np.exp(r*roots) - (a*roots + c))/b))

def calc_fast_pr(r,a,b,c):
    return np.dot(weights,np.exp(-(r**2 - (2*roots*r))/2)*norm.cdf(-(np.exp(r*roots) - (a*roots + c))/b))

def calc(r, a, b, c):
    def calc_limit(x):
        return (np.exp(r*x) - (a*x + c))/b
    
    p = integrate.quad(lambda x: norm.pdf(x) * norm.cdf(- calc_limit(x)), -np.inf, np.inf)
    mx = integrate.quad(lambda x: x * norm.pdf(x) * norm.cdf(- calc_limit(x)), -np.inf, np.inf)
    my = integrate.quad(lambda x: norm.pdf(x) * norm.pdf(calc_limit(x)), -np.inf, np.inf)
    return p[0], mx[0], my[0]

def fast_solve_p(prob, r, a, b, c=1.0, tol=0.001,distribution = "norm"):
    low = 0.0
    high = None
    calc_fast_p =  meta_calc_fast_p(r,a,b,distribution)
    while ((high is None) or (((high - low)/high) > tol)):
        p = calc_fast_p(c)
        if p < prob:
            low = c
            if high is None:
                c = 2*c
            else:
                c = (high + low)/2
        else:
            high = c
            c = (high + low)/2
    return low

def fast_solve_my(prob, my, r, a, b=1.0, tol=0.001,distribution = "norm"):
    low = 0.0
    high = None
    c = 1.0
    while ((high is None) or (((high - low)/high) > tol)):
        c = fast_solve_p(prob, r, a, b, c=c, tol=tol,distribution =distribution)
        m = calc_fast_my(r,a,b,c,distribution)
        if m < my:
            low = b
            if high is None:
                b = 2*b
            else:
                b = (high + low)/2
        else:
            high = b
            b = (high + low)/2
    return low, fast_solve_p(prob, r, a, low,distribution = distribution)

def fast_solve_mx(prob, my, mx, r, a=1.0, tol=0.001,distribution = "norm"):
    low = 0.0 
    high = None
    b = 1.0
    while ((high is None) or ((high - low) > 0.1*tol)):
        b, c = fast_solve_my(prob, my, r, a, b=b, tol=tol,distribution= distribution)
        m = calc_fast_mx(r,a,b,c,distribution)
        if m < mx:
            low = a
            if high is None:
                a = 2*a
            else:
                a = (high + low)/2
        else:
            high = a
            a = (high + low)/2
    if low > 0:
        a = low
    elif high < 0:
        a = high
    else:
        a = 0.0
    return a, fast_solve_my(prob, my, r, a,distribution)

def binary_search(eval_func, high, low, tol):
    x = (high + low)/2
    while ((high - low) > tol):
        if eval_func(x) < 0:
            low = x
        else:
            high = x
        x = (high + low)/2
    return low, high

def calculate_radius(pABar: int, meanBar: int, tol=0.00001,distribution = "norm") -> float:
    if distribution == "norm":
        low = norm.ppf(pABar)
    elif distribution == "lap":
        low = laplace.ppf(pABar,scale = 1)
    elif distribution == "uni":
        low = uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).ppf(pABar)
    if meanBar == 0.0:
        if distribution == "norm":
            high = norm.ppf((1+pABar)/2)
        elif distribution == "lap":
            high = laplace.ppf((1+pABar)/2)
        elif distribution == "uni":
            high = uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).ppf((1+pABar)/2)
    else:
        high = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*abs(meanBar)))
    if distribution == "norm": 
        if meanBar >= norm.pdf(low):
            return low
        elif meanBar <= -norm.pdf(low):
            return np.inf
        elif meanBar == 0:
            z1 = high
            z2 = -high
        elif meanBar > 0:
            def eval_func(x):
                y = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(x) - meanBar)))
                return norm.cdf(x) - (norm.cdf(y) + pABar) 
            l, h = binary_search(eval_func, high, low, tol)
            z1 = l
            z2 = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(z1) - meanBar)))  
        else:
            high, low = -low, -high
            def eval_func(x):
                y = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(x) + meanBar)))
                return pABar - (norm.cdf(y) - norm.cdf(x)) 
            l, h = binary_search(eval_func, high, low, tol)
            z2 = h
            z1 = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(norm.pdf(z2) + meanBar)))
        high = z1
        low = norm.ppf(pABar)
        r = (high+low)/2
        diff = (norm.cdf(z1 - r) - norm.cdf(z2 - r)) - 0.5
        while ((high - low) > tol) :
            if diff > 0:
                low = r
            else:
                high = r
            r = (high+low)/2
            diff = (norm.cdf(z1 - r) - norm.cdf(z2 - r)) - 0.5
        return low
    elif distribution == "lap": 
        if meanBar >= laplace.pdf(low):
            return low
        elif meanBar <= -laplace.pdf(low):
            return np.inf
        elif meanBar == 0:
            z1 = high
            z2 = -high
        elif meanBar > 0:
            def eval_func(x):
                y = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(laplace.pdf(x) - meanBar)))
                return laplace.cdf(x) - (laplace.cdf(y) + pABar) 
            l, h = binary_search(eval_func, high, low, tol)
            z1 = l
            z2 = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(laplace.pdf(z1) - meanBar)))  
        else:
            high, low = -low, -high
            def eval_func(x):
                y = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(laplace.pdf(x) + meanBar)))
                return pABar - (laplace.cdf(y) - laplace.cdf(x)) 
            l, h = binary_search(eval_func, high, low, tol)
            z2 = h
            z1 = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(laplace.pdf(z2) + meanBar)))
        high = z1
        low = laplace.ppf(pABar)
        r = (high+low)/2
        diff = (laplace.cdf(z1 - r) - laplace.cdf(z2 - r)) - 0.5
        while ((high - low) > tol) :
            if diff > 0:
                low = r
            else:
                high = r
            r = (high+low)/2
            diff = (laplace.cdf(z1 - r) - laplace.cdf(z2 - r)) - 0.5
        return low
    elif distribution == "uni": 
        if meanBar >= uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(low):
            return low
        elif meanBar <= -uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(low):
            return np.inf
        elif meanBar == 0:
            z1 = high
            z2 = -high
        elif meanBar > 0:
            def eval_func(x):
                y = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(x) - meanBar)))
                return uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(x) - (uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(y) + pABar) 
            l, h = binary_search(eval_func, high, low, tol)
            z1 = l
            z2 = -np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(z1) - meanBar)))  
        else:
            high, low = -low, -high
            def eval_func(x):
                y = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(x) + meanBar)))
                return pABar - (uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(y) - uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(x)) 
            l, h = binary_search(eval_func, high, low, tol)
            z2 = h
            z1 = np.sqrt(-2*np.log(np.sqrt(2*np.pi)*(uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).pdf(z2) + meanBar)))
        high = z1
        low = uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).ppf(pABar)
        r = (high+low)/2
        diff = (uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(z1 - r) - uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(z2 - r)) - 0.5
        while ((high - low) > tol) :
            if diff > 0:
                low = r
            else:
                high = r
            r = (high+low)/2
            diff = (uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(z1 - r) - uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).cdf(z2 - r)) - 0.5
        return low


def numerical_radius(prob, my, mx, eps=1e-4, a=1.0, tol=0.001,distribution = "norm"):
    distribution = distribution
    if my <= 1e-7:
        return calculate_radius(prob, abs(mx),distribution = distribution)
    prob, my = prob*(1-eps), my*(1-eps)
    mx = mx - eps*abs(mx)
    if distribution == "uni":
        low = uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).ppf(prob)
    elif distribution == "lap":
        low = laplace.ppf(prob)
    elif distribution == "norm":
        low = norm.ppf(prob)
    high = calculate_radius(prob, - np.sqrt(my**2 + mx**2),distribution = distribution)
    r = (high + low)/2
    while (((high - low)/high) > tol):
        r = (high + low)/2
        a, (b, c) = fast_solve_mx(prob, my, mx, r, a=a, tol=tol,distribution=distribution)
        pr = calc_fast_pr(r,a,b,c)
        if pr > 0.5:
            low = r        
        else:
            high = r
    a, (b, c) = fast_solve_mx(prob, my, mx, low, a=a, tol=tol,distribution=distribution)
    return low