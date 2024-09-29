def cmd(X, X_test, K=5):
    r"""
    central moment discrepancy (cmd). objective function for keras models (theano or tensorflow backend). Zellinger, Werner, et al. "Robust unsupervised domain adaptation for
    neural networks via moment alignment.", Zellinger, Werner, et al. "Central moment discrepancy (CMD) for
    domain-invariant representation learning.", ICLR, 2017.

    Args:
        X (Tensor): training instances
        X_test (Tensor): IID samples
        K (int): number of approximation degrees

    Returns (Tensor):
         central moment discrepancy

    """
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = [dm]
    for i in range(K - 1):
        # moment diff of centralized samples
        scms.append(moment_diff(sx1, sx2, i + 2))
        # scms+=moment_diff(sx1,sx2,1)
    return sum(scms)


def l2diff(x1, x2):
    r"""
    standard euclidean norm
    """
    return (x1 - x2).norm(p=2)


def moment_diff(sx1, sx2, k):
    r"""
    difference between moments
    """
    ss1 = sx1.pow(k).mean(0)
    ss2 = sx2.pow(k).mean(0)
    # ss1 = sx1.mean(0)
    # ss2 = sx2.mean(0)
    return l2diff(ss1, ss2)


import torch


def mmd(x, y, width=1):
    x_n = x.shape[0]
    y_n = y.shape[0]

    x_square = torch.sum(x * x, 1)
    y_square = torch.sum(y * y, 1)

    kxy = torch.matmul(x, y.t())
    kxy = kxy - 0.5 * x_square.unsqueeze(1).expand(x_n, y_n)
    kxy = kxy - 0.5 * y_square.expand(x_n, y_n)
    kxy = torch.exp(width * kxy).sum() / x_n / y_n

    kxx = torch.matmul(x, x.t())
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = kxx - 0.5 * x_square.expand(x_n, x_n)
    kxx = torch.exp(width * kxx).sum() / x_n / x_n

    kyy = torch.matmul(y, y.t())
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = kyy - 0.5 * y_square.expand(y_n, y_n)
    kyy = torch.exp(width * kyy).sum() / y_n / y_n

    return kxx + kyy - 2 * kxy

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return ((x1-x2)**2).sum().sqrt()


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1**k).mean(0)
    ss2 = (sx2**k).mean(0)
    return l2diff(ss1, ss2)


def CMD(x1,x2, K=5):
    mx1 = x1.mean(dim=0)
    mx2 = x2.mean(dim=0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = l2diff(mx1, mx2)
    scms = dm

    for i in range(K-1):
        # moment diff of centralized samples
        scms += moment_diff(sx1, sx2, i+2)
    return scms

import math
import numpy as np


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / (sigma * sigma)
    KX = np.exp(KX)
    return KX


def kernel_HSIC(X, Y, sigma):
    return np.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))


def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)


def kernel_CKA(X, Y, sigma=None):
    hsic = kernel_HSIC(X, Y, sigma)
    var1 = np.sqrt(kernel_HSIC(X, X, sigma))
    var2 = np.sqrt(kernel_HSIC(Y, Y, sigma))

    return hsic / (var1 * var2)


class CMD_NEW():
    def mmatch(self, x1, x2, n_moments=5):
        mx1 = x1.mean(0)
        mx2 = x2.mean(0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = self.matchnorm(mx1, mx2)
        # scms = [dm]
        scms = dm
        for i in range(n_moments - 1):
            # moment diff of centralized samples
            # scms.append(self.moment_diff(sx1, sx2, i+2))
            scms += self.moment_diff(sx1, sx2, i + 2)
        # return sum(scms)
        return scms

    def moment_diff(self, sx1, sx2, k):
        """
        difference between moments
        """
        ss1 = sx1.pow(k).mean(0)
        ss2 = sx2.pow(k).mean(0)
        # ss1 = sx1.mean(0)
        # ss2 = sx2.mean(0)
        return self.matchnorm(ss1, ss2)

    def matchnorm(self, x1, x2):
        return (x1 - x2).norm(p=2)

    #    return T.abs_(x1 - x2).sum()# maximum
    #    return 1-T.minimum(x1,x2).sum()/T.maximum(x1,x2).sum()# ruzicka
    #    return kl_divergence(x1,x2)# KL-divergence

    def mmd(self, x1, x2, beta=1.0):
        x1x1 = self.gaussian_kernel(x1, x1, beta)
        x1x2 = self.gaussian_kernel(x1, x2, beta)
        x2x2 = self.gaussian_kernel(x2, x2, beta)
        diff = x1x1.mean() - 2 * x1x2.mean() + x2x2.mean()
        return diff

    def gaussian_kernel(self, x1, x2, beta=1.0):
        # r = x1.dimshuffle(0,'x',1)
        r = x1.view(x1.shape[0], 1, x1.shape[1])
        return torch.exp(-beta * torch.square(r - x2).sum(axis=-1))

    def pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
            y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        x_norm = (x ** 2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y ** 2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # dist = torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        # if y is None:
        #     dist = dist - torch.diag(dist.diag)
        return torch.clamp(dist, 0.0, np.inf)