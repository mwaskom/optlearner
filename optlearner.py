"""Python implementation of a Bayesian probabilistic learner.

This model was originally described in Behrens et al. Nat Neuro 2007.

The code here was adapted from the original C++ code provided by
Tim Behrens.

"""
from __future__ import division
import numpy as np
from numpy import log, exp
from scipy.special import gammaln
import matplotlib.pyplot as plt


class OptimalLearner(object):

    def __init__(self):

        # Set up the parameter grids
        self.p_grid = make_grid(.01, .99, .02)
        self.I_grid = make_grid(log(2), log(10000), .2)
        self.k_grid = make_grid(log(5e-4), log(20), .2)
        self.v_grid = 1 / self.I_grid

        self._p_size = self.p_grid.size
        self._I_size = self.I_grid.size
        self._k_size = self.k_grid.size

        # Set up the transitional distributions
        I_trans = np.vectorize(I_trans_func)(*np.meshgrid(self.I_grid,
                                                          self.I_grid,
                                                          self.k_grid,
                                                          indexing="ij"))
        self._I_trans = I_trans / I_trans.sum(axis=0)

        p_trans = np.vectorize(p_trans_func)(*np.meshgrid(self.p_grid,
                                                          self.p_grid,
                                                          self.I_grid,
                                                          indexing="ij"))
        self._p_trans = p_trans / p_trans.sum(axis=0)

        # Initialize the learner and history
        self.reset()

    def _update(self, y):
        """Perform the Bayesian update for a trial based on y."""

        # Information leak (increase in the variance of the joint
        # distribution to reflect uncertainty of a new trial)
        # -------------------------------------------------------

        pIk = self.pIk

        # Multiply P(I_i+1 | I_i, k) by P(p_i, I_i, k) and
        # integrate out I_i, which gives P(p_i, I_i+1, k)
        I_leaked = np.einsum("jkl,ikl->ijl", self._I_trans, pIk)

        # Multiply P(p_i, I_i+1, k) by P(p_p+1 | p_i, I_i+1) and
        # integrate out p_i, which gives P(p_i+1, I_i+1, k)
        p_leaked = np.einsum("jkl,ijk->ikl", I_leaked, self._p_trans)

        # Set the running joint distribution to the new values
        pIk = p_leaked

        # Update P(p_i+1, I_i+1, k) based on the newly observed data
        # ----------------------------------------------------------

        likelihood = self.p_grid if y else 1 - self.p_grid
        pIk *= likelihood[:, np.newaxis, np.newaxis]

        # Normalize the new distribution
        # ------------------------------

        self.pIk = pIk / pIk.sum()

    @property
    def p_hats(self):
        return np.atleast_1d(self._p_hats)

    @property
    def v_hats(self):
        return np.atleast_1d(self._v_hats)

    @property
    def data(self):
        return np.atleast_1d(self._data)

    def fit(self, data):
        """Fit the model to a sequence of Bernoulli observations."""
        for y in data:
            self._update(y)
            pI = self.pIk.sum(axis=2)
            self.p_dists.append(pI.sum(axis=1))
            self.v_dists.append(pI.sum(axis=0))
            self._p_hats.append(np.sum(self.p_dists[-1] * self.p_grid))
            self._v_hats.append(1 / np.sum(self.v_dists[-1] * self.I_grid))
            self._data.append(y)

    def reset(self):
        """Reset the history of the learner."""
        # Initialize the joint distribution P(p, I, k)
        pIk = np.ones((self._p_size, self._I_size, self._k_size))
        self.pIk = pIk / pIk.sum()

        # Initialize the memory lists
        self.p_dists = []
        self.v_dists = []
        self._p_hats = []
        self._v_hats = []
        self._data = []

    def plot_history(self, ground_truth=None, shifts=None, **kwargs):
        """Plot the data and posterior means from the history."""
        try:
            import seaborn as sns
            palette = sns.husl_palette(3)
        except ImportError:
            palette = [[0.967, 0.441, 0.535],
                       [0.312, 0.692, 0.192],
                       [0.232, 0.639, 0.926]]
        red, green, blue = palette

        f = plt.figure(**kwargs)
        p_ax = f.add_subplot(211, ylim=(-0.1, 1.1))
        p_ax.plot(self.p_hats, c=blue)
        p_ax.plot(self.data, marker="o", c="#444444", ls="", alpha=.5, ms=4)
        if ground_truth is not None:
            p_ax.plot(ground_truth, c="dimgray", ls="--")
        p_ax.set_ylabel("$p$", size=16)
        p_ax.set_xticklabels([])

        v_ax = f.add_subplot(212, ylim=(.1, .4), sharex=p_ax)
        v_ax.plot(self.v_hats, c=green)
        if shifts is not None:
            for trial in shifts:
                v_ax.axvline(trial, c="dimgray", ls=":")
        v_ax.set_ylabel("$v$", size=16)
        f.tight_layout()

    def plot_joint(self, cmap="BuGn"):
        """Plot the current joint distribution P(p, v | y_<=i)."""
        try:
            import seaborn as sns
            pal = sns.color_palette(cmap, 10)
            lc = pal[7]
            bg = pal[0]
        except ImportError:
            pal = "BuGn"
            lc = (0.0, 0.407, 0.164)
            bg = (0.906, 0.964, 0.978)

        pI = self.pIk.sum(axis=2)

        fig = plt.figure(figsize=(7, 7))
        gs = plt.GridSpec(3, 3)

        ax1 = fig.add_subplot(gs[1:, :-1])
        ax1.contourf(self.p_grid, self.v_grid, pI.T, 30, cmap=cmap)

        sns.axlabel("$p$", "$v$", size=16)

        ax2 = fig.add_subplot(gs[1:, -1], axis_bgcolor=bg)
        ax2.plot(pI.sum(axis=0), self.v_grid, c=lc, lw=3)
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax3 = fig.add_subplot(gs[0, :2], axis_bgcolor=bg)
        ax3.plot(self.p_grid, pI.sum(axis=1), c=lc, lw=3)
        ax3.set_xticks([])
        ax3.set_yticks([])


def make_grid(start, stop, step):
    """Define an even grid over a parameter space."""
    count = (stop - start) / step + 1
    return np.linspace(start, stop, count)


def I_trans_func(I_p1, I, k):
    """I_p1 is normal with mean I and std dev k."""
    var = exp(k * 2)
    pdf = exp(-np.power(I_p1 - I, 2)) / (2 * var)
    pdf /= np.sqrt(2 * np.pi * var)
    return pdf


def p_trans_func(p_p1, p, I_p1):
    """p_p1 is beta with mean p and precision I_p1."""
    a = 1 + exp(I_p1) * p
    b = 1 + exp(I_p1) * (1 - p)

    if 0 < p_p1 < 1:
        logkerna = (a - 1) * log(p_p1)
        logkernb = (b - 1) * log(1 - p_p1)
        betaln_ab = gammaln(a) + gammaln(b) - gammaln(a + b)
        return exp(logkerna + logkernb - betaln_ab)
    else:
        return 0
