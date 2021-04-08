"""Echo State Networkなどを定義する場所"""
from typing import Dict
from typing import Callable
import numpy as np


class EchoStateNetwork():
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
        self.x_state = None

    def _setup_parameters(self, density, rho_w, init_type):
        if init_type and "input" in init_type:
            input_init_func = init_type["input"]
        else:
            input_init_func = np.random.rand
        self.w_in = input_init_func(self.d_in, self.d_hidden)
        self.w_in -= self.w_in.mean()
        self.w_in *= 2.0
        if init_type and "hidden" in init_type:
            init_func = init_type["hidden"]
        else:
            init_func = np.random.rand
        w_hidden = init_func(self.d_hidden, self.d_hidden)
        w_hidden -= w_hidden.mean()
        w_hidden *= 2.0
        random_p_hidden = np.random.rand(self.d_hidden, self.d_hidden)
        w_hidden = np.where(random_p_hidden > density, 0,
                            w_hidden)  # ランダムに結合を切る
        w_hidden *= rho_w / np.max(np.abs(np.linalg.eig(w_hidden)[0]))
        self.w_hidden = w_hidden
        self.w_out = np.random.randn(self.d_hidden, self.d_out)

    def step_forward(self, input_vector, leak_rate=0.8):
        """ネットワークの状態を1ステップ分すすめる"""
        if not isinstance(self.x_state, np.ndarray):
            batch_size = input_vector.shape[0]
            self.x_state = np.random.randn(batch_size, self.d_hidden)
        old_state = self.x_state.copy()
        self.x_state = leak_rate * old_state + (
            1.0 - leak_rate) * self.activation(
                np.dot(input_vector, self.w_in) +
                np.dot(old_state, self.w_hidden))
        return np.dot(self.x_state, self.w_out)

    def take_steps(self, input_vectors, num_steps=None, leak_rate=0.0):
        """指定されたステップ数分だけemulateを行う

        Parameters
        ----------
        input_vectors : np.ndarray
            (num_steps, batch_size, d_in)のshapeを持ったvectors.
        num_steps : int, optional
            emulateしたいステップ数, by default None
        leak_rate : float, optional
            前の状態をどの程度保っておきたいか, by default 0.0

        Returns
        -------
        record_states, record_outputs : np.ndarray
            ネットワークの内部状態の記録と出力の記録
        """
        assert input_vectors.ndim == 3, \
            "input_vectors should have 3 dimensions (num_steps, batch_size, d_in) but got %s" % input_vectors.shape
        assert input_vectors.shape[
            2] == self.d_in, \
                "input_vectors should have the same number of dimensions at input but got %d" % input_vectors.shape[
                2]
        if not num_steps:
            num_steps = input_vectors.shape[0]
        batch_size = input_vectors.shape[1]
        record_states = np.zeros([num_steps, batch_size, self.d_hidden])
        record_outputs = np.zeros([num_steps, batch_size, self.d_out])
        for step in range(num_steps):
            record_outputs[step] = self.step_forward(
                input_vectors[step], leak_rate=leak_rate).copy()
            record_states[step] = self.x_state.copy()
        return record_states, record_outputs

    def update_readout(self, new_weights):
        """読み出し部分の重みを更新する"""
        assert new_weights.shape == self.w_out.shape, "shape doesn't match %s vs %s" % (
            new_weights.shape, self.w_out.shape)
        self.w_out = new_weights
