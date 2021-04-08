"""記憶容量を計測するコードを実装する"""
import numpy as np

from reservoir_network.network import EchoStateNetwork


def determination_coefficient(seq0, seq1):
    """決定係数を計算する

    Parameters
    ----------
    seq0: np.ndarray
        (T,) の1次元配列
    seq1: np.ndarray
        (T,) の1次元配列
    """
    return np.cov(seq0, seq1)[0, 1]**2 / (np.var(seq0) * np.var(seq1))


def train_network(network,
                  input_vectors,
                  desired_outputs,
                  offset=8,
                  total_steps=100):
    """線形回帰を使ってネットワークを訓練させる"""
    record_states, _ = network.take_steps(input_vectors,
                                          num_steps=total_steps,
                                          leak_rate=0.0)
    record_states = record_states[offset:].reshape(-1, network.d_hidden)

    inverse_matrix = np.linalg.inv(np.dot(record_states.T, record_states))
    trained_w_out = np.dot(desired_outputs.T,
                           np.dot(record_states, inverse_matrix)).T

    network.update_readout(trained_w_out)


def calc_memory_capacity(max_k=20, rho_w=0.95, d_hidden=100, total_steps=1000):
    """記憶容量を測定する"""
    params = {
        "d_in": 1,
        "d_hidden": d_hidden,
        "d_out": 1,
        "density": 0.2,
        "rho_w": rho_w
    }
    network = EchoStateNetwork(**params)
    r_value_list = []
    for k in range(max_k):
        input_vectors = np.random.randn(total_steps, 1, 1)
        if k > 0:
            desired_outputs = input_vectors[:-k].reshape(-1, 1)
        else:
            desired_outputs = input_vectors.reshape(-1, 1)
        train_network(network,
                      input_vectors,
                      desired_outputs,
                      offset=k,
                      total_steps=total_steps)

        test_input_vectors = np.random.randn(total_steps, 1, 1)
        _, record_outputs = network.take_steps(test_input_vectors,
                                               num_steps=total_steps,
                                               leak_rate=0.0)

        if k > 0:
            r_2 = determination_coefficient(record_outputs[k:, 0, 0],
                                            test_input_vectors[:-k, 0, 0])
        else:
            r_2 = determination_coefficient(record_outputs[:, 0, 0],
                                            test_input_vectors[:, 0, 0])
        r_value_list.append(r_2)
    return r_value_list
