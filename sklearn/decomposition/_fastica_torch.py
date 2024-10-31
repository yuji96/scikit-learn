"""
Python implementation of the fast ICA algorithms.

Reference: Tables 8.3 and 8.4 page 196 in the book:
Independent Component Analysis, by  Hyvarinen et al.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import torch
from tqdm import tqdm

from ..exceptions import ConvergenceWarning


def _sym_decorrelation(W):
    """Symmetric decorrelation
    i.e. W <- (W * W.T) ^{-1/2} * W
    """
    s, u = torch.linalg.eigh(W @ W.T)
    # Avoid sqrt of negative values because of rounding errors. Note that
    # np.sqrt(tiny) is larger than tiny and therefore this clipping also
    # prevents division by zero in the next step.
    # s = torch.clip(s, min=torch.finfo(W.dtype).tiny)
    s = torch.clip(s, min=np.finfo(np.float32).tiny)

    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    return torch.linalg.multi_dot([u * (1.0 / torch.sqrt(s)), u.T, W])


def _ica_torch(X, tol, g, fun_args, max_iter, w_init):
    """Parallel FastICA.

    Used internally by FastICA --main loop

    """
    X = torch.from_numpy(X).cuda()
    w_init = torch.from_numpy(w_init).cuda()

    W = _sym_decorrelation(w_init)
    del w_init
    p_ = float(X.shape[1])
    for ii in tqdm(range(max_iter)):
        gwtx, g_wtx = g(W @ X, fun_args)
        W1 = _sym_decorrelation((gwtx @ X.T) / p_ - g_wtx[:, np.newaxis] * W)
        del gwtx, g_wtx
        # builtin max, abs are faster than numpy counter parts.
        # np.einsum allows having the lowest memory footprint.
        # It is faster than np.diag(np.dot(W1, W.T)).
        # lim = max(abs(abs(np.einsum("ij,ij->i", W1, W)) - 1))
        lim = (torch.einsum("ij,ij->i", W1, W).abs() - 1).abs().max()
        W = W1
        if lim < tol:
            break
    else:
        warnings.warn(
            (
                "FastICA did not converge. Consider increasing "
                "tolerance or the maximum number of iterations."
            ),
            ConvergenceWarning,
        )

    W = W.cpu().numpy()
    return W, ii + 1
