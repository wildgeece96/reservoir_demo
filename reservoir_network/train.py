"""ネットワークのreadoutを学習させる実装"""
import numpy as np


def _linear_regression(record_states, desired_outputs):
    """線形回帰で重みを計算する"""
    inverse_matrix = np.linalg.inv(np.dot(record_states.T, record_states))
    return np.dot(desired_outputs.T, np.dot(record_states, inverse_matrix)).T


def _ridge_regression(record_states, desired_outputs, alpha=1e-4):
    """リッジ回帰で重みを計算する"""
    inverse_matrix = np.linalg.inv(
        np.dot(record_states.T, record_states) +
        alpha * np.eye(record_states.shape[1]))
    return np.dot(desired_outputs.T, np.dot(record_states, inverse_matrix)).T


def train_network(network,
                  input_vectors,
                  desired_outputs,
                  start_offset=8,
                  end_offset=0,
                  total_steps=None,
                  train_method="linear",
                  ridge_alpha=1e-4,
                  leak_rate=0.0):
    """線形回帰を使ってネットワークを訓練させる

    Parameters
    ----------
    network : EchoStateNetwork
        リザバーネットワーク
    input_vectors : np.ndarray
        (num_steps, batch_size, d_in)のshapeを持った入力ベクトル
    desired_outputs : np.ndarray
        (num_steps - offset, batch_size, d_out)のshapeを持ったほしい出力ベクトル
    start_offset : int, optional
        入力のうち訓練に使わない先頭の要素数, by default 8
    end_offset : int, optional
        入力のうち訓練に使わない最後の要素数, by default 0
    total_steps : int, optional
        エミュレートしたいトータルステップ数, by default 100
    train_method : str, optional
        リードアウトの重みを学習させる手法。"linear", "ridge"の2種類が現状選択できる, by default "linear"
    ridge_alpha : float, optional
        リッジ回帰で重みを学習させるときの正則化の度合い
    leak_rate : float, optional
        エミュレートするときにどれぐらい前回の状態を保持するか, by default 0.0
    """
    if not total_steps:
        total_steps = input_vectors.shape[0]
    assert input_vectors.shape[
        0] - start_offset - end_offset == desired_outputs.shape[
            0], "The number of steps should be equal but found (%s vs %s)" % (
                input_vectors.shape[0] - start_offset - end_offset,
                desired_outputs.shape[0])
    record_states, _ = network.take_steps(input_vectors,
                                          num_steps=total_steps,
                                          leak_rate=leak_rate)
    if start_offset > 0:
        record_states = record_states[start_offset:]
    if end_offset > 0:
        record_states = record_states[:-end_offset]
    record_states = record_states.reshape(-1, network.d_hidden)

    desired_outputs = desired_outputs.reshape(
        -1, network.d_out)  # バッチによる違いは全部無視で(num_steps*batch_size, d_out)に変更
    if train_method == "linear":
        trained_w_out = _linear_regression(record_states, desired_outputs)
    elif train_method == "ridge":
        trained_w_out = _ridge_regression(record_states,
                                          desired_outputs,
                                          alpha=ridge_alpha)
    network.update_readout(trained_w_out)
