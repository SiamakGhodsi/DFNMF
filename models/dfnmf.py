import torch
from tqdm import tqdm
import os

eps = torch.tensor(10**-10)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class DFNMF(object):
    """
       Deep Fair Nonnegative Matrix Tri-Factorization (DFNMF)
       This class implements a multi-layer factorization model:
            A ≈ H₁ H₂ ... Hₚ Wₚ (Hₚ)ᵀ ... (H₂)ᵀ (H₁)ᵀ
            + fairness regularization term: λ ||Fᵀ (H₁ ... Hₚ)||_F²
       Pre-training is performed layer by layer using SNMTF.
       Then, fine-tuning alternates between updating each H_i (with fairness)
       and updating the corresponding interaction matrices.
    """
    def __init__(self, A: torch.tensor, F: torch.tensor, args):
        """
                Parameters:
                  A      : (n x n) input data matrix (e.g. an adjacency matrix)
                  F      : fairness matrix (used to build F*Fᵀ)
                  p      : number of layers (pretrain and fine-tune)
                  ft_itr : number of fine-tuning iterations (args.ft_iter)
                  pr_itr : number of iterations for pre-training each layer
                  lg     : fairness (group) regularization parameter λ
                  err    : total model loss
                  dtype  : torch data type
        """
        self.A = A
        self.F = F
        self.args = args
        self.p = len(self.args.layers)
        self.err = torch.zeros(self.args.ft_itr)
        self.lg = self.args.lg
        FFT = F @ F.T
        self.FFT = FFT
        self.FFTn = (torch.abs(FFT) - FFT) / 2  # negative part
        self.FFTp = (torch.abs(FFT) + FFT) / 2  # positive part

    @staticmethod
    def _shallow_nmtf(Z, l, pr_itr):
        """
            Shallow Nonnegative Matrix Tri-Factorization (SNMTF)
            Factorizes Z ≈ H W Hᵀ using multiplicative update rules.
        """
        n = Z.shape[0]
        W = torch.eye(l, l) + torch.ones(l, l) * (1 / l)
        H = torch.rand(n, l)
        for t in range(pr_itr):
            Hn = Z.T @ H @ W + Z @ H @ W.T
            Hd = H @ W.T @ H.T @ H @ W + H @ W @ H.T @ H @ W.T
            H = H * (Hn / torch.maximum(Hd, eps)) ** 0.25

            Wn = H.T @ Z @ H
            Wd = H.T @ H @ W @ H.T @ H
            W = W * (Wn / torch.maximum(Wd, eps))

        return H, W

    def pre_training(self):
        # Pre-training each NMF layer.
        print("\nLayer pre-training started. \n")
        self.Hs = []
        self.Ws = []
        for l in tqdm(range(self.p), desc="Pretraining: ", leave=True):

            if l == 0:
                H, W = self._shallow_nmtf(self.A, self.args.layers[l], self.args.pr_itr)
            else:
                H, W = self._shallow_nmtf(self.Ws[l - 1], self.args.layers[l], self.args.pr_itr)
            self.Hs.append(H)
            self.Ws.append(W)


    def setup_Q(self):
        # Setting up Q matrices.
        self.Q_s = [None for _ in range(self.p + 1)]
        self.Q_s[self.p] = torch.eye(self.args.layers[self.p - 1]).type(self.args.type)
        for i in range(self.p - 1, -1, -1):
            self.Q_s[i] = self.Hs[i] @ self.Q_s[i + 1]

    def update_H(self, i):
        # Updating left hand factors.
        if i == 0:

            Psi = self.Hs[0] @ self.Q_s[1]

            Hn = self.A.T              @ Psi @ self.Ws[self.p - 1]    @ self.Q_s[1].T \
                 + self.A              @ Psi @ self.Ws[self.p - 1].T  @ self.Q_s[1].T \
                 + self.lg * self.FFTn @ Psi                          @ self.Q_s[1].T

            Hd = Psi @ self.Ws[self.p - 1].T @ Psi.T @ Psi @ self.Ws[self.p - 1]   @ self.Q_s[1].T \
               + Psi @ self.Ws[self.p - 1]   @ Psi.T @ Psi @ self.Ws[self.p - 1].T @ self.Q_s[1].T \
               + self.lg * self.FFTp         @ Psi                                 @ self.Q_s[1].T

            self.Hs[0] = self.Hs[0] * (Hn / torch.maximum(Hd, torch.tensor(1e-10))) ** 0.5
        else:

            Psi = self.P @ self.Hs[i] @ self.Q_s[i+1]

            Hn = self.P.T @ self.A.T              @ Psi @ self.Ws[self.p - 1]   @ self.Q_s[i+1].T \
               + self.P.T @ self.A                @ Psi @ self.Ws[self.p - 1].T @ self.Q_s[i+1].T \
               + self.P.T @ (self.lg * self.FFTn) @ Psi                         @ self.Q_s[i+1].T

            Hd = self.P.T @ Psi @ self.Ws[self.p - 1].T @ Psi.T @ Psi @ self.Ws[self.p - 1]   @ self.Q_s[i+1].T \
               + self.P.T @ Psi @ self.Ws[self.p - 1]   @ Psi.T @ Psi @ self.Ws[self.p - 1].T @ self.Q_s[i+1].T \
               + self.P.T @ (self.lg * self.FFTp)       @ Psi                                 @ self.Q_s[i+1].T
            self.Hs[i] = self.Hs[i] * (Hn / torch.maximum(Hd, torch.tensor(1e-10))) ** 0.5

    def update_P(self, i):
        # Setting up P matrices.
        if i == 0:
            self.P = self.Hs[0]
        else:
            self.P = self.P @ self.Hs[i]

    def update_W(self, i):
        # Updating right hand factors.
        # if i == self.p-1:
            Wn = self.P.T @ self.A                         @ self.P
            Wd = self.P.T @ self.P @ self.Ws[i] @ self.P.T @ self.P
            self.Ws[i] = self.Ws[i] * (Wn / torch.maximum(Wd, torch.tensor(1e-10)))

    def calculate_cost(self, i):
        self.err[i] = torch.norm(self.A - self.P @ self.Ws[-1] @ self.P.T) ** 2 \
       + self.lg * torch.norm(self.F.T @ self.P, p='fro') ** 2

    def training(self):
        # Training process after pre-training.
        self.pre_training()

        # for iteration in range(self.args.ft_itr):
        for iteration in tqdm(range(self.args.ft_itr), desc="Fine-tuning: ", leave=True):

            self.setup_Q()

            for i in range(self.p):
                self.update_H(i)
                self.update_P(i)
                self.update_W(i)

            self.calculate_cost(iteration)

        return self.Hs, self.Ws, self.err


"""
## GPU or CPU
GPU = False
if GPU:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("num GPUs", torch.cuda.device_count())
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor
    print("CPU")



args = Namespace()
args.type = dtype
args.ft_itr = 300
args.pr_itr = 50
args.layers = [40, 40, 10]
args.err = torch.zeros(args.ft_itr)
args.p = len(args.layers)

args.lg = 1
args.li = 1
"""