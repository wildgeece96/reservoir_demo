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
        self.activation = activation
        self.X_state = np.random.randn(batch_size, self.d_hidden)

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
        w_hidden = np.where(random_p_hidden > density, 0,
                            w_hidden)  # ランダムに結合を切る
        w_hidden *= rho_w / np.max(np.abs(np.linalg.eig(w_hidden)[0]))
        self.w_hidden = w_hidden
        self.w_out = np.random.randn(self.d_hidden, self.d_out)

    def step_forward(self, input_vector, leak_rate=0.8):
        """ネットワークの状態を1ステップ分すすめる"""
        self.X_state = (1 - leak_rate) * self.X_state + self.activation(
            np.dot(self.input_vector, self.w_in) +
            np.dot(self.X_state, self.w_hidden))
        self.y_output = np.dot(self.X_state, self.w_out)
