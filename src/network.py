"""Echo State Networkなどを定義する場所"""
import numpy as np
from typing import Dict
from typing import Callable


class EchoStateNetwork(object):
    """Echo State Networkの基本クラス"""
    def __init__(self,
                 d_in: int,
                 d_hidden: int,
                 d_out: int,
                 density: float = 0.2,
                 rho_w: float = 0.95,
                 init_type: Dict[str, Callable] = None,
                 activation: Callable = np.tanh):
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.d_out = d_out
        self._setup_parameters(density, rho_w, init_type)

    def _setup_parameters(self, density, rho_w, init_type):
        if init_type and "input" in init_type:
            input_init_func = init_type["input"]
        else:
            input_init_func = np.random.rand
        self.w_in = input_init_func(self.d_in, self.d_hidden)
        self.w_in -= self.w_in.mean()
        if init_type and "hidden" in init_type:
            init_func = init_type["hidden"]
        else:
            init_func = np.random.rand
        w_hidden = init_func(self.d_hidden, self.d_hidden)
        random_p_hidden = np.random.rand(self.d_hidden, self.d_hidden)
        w_hidden = np.where(random_p_hidden > density, 0, w_hidden)
        w_hidden *= rho_w / np.max(np.abs(np.linalg.eig(w_hidden)[0]))
        self.w_hidden = w_hidden
        self.w_out = np.random.randn(self.d_hidden, self.d_out)