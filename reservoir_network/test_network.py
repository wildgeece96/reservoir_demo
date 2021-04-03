"""Echo State Networkの動作を確認する単体テスト"""
from unittest import TestCase
import numpy as np

from reservoir_network.network import EchoStateNetwork


class TestEchoStateNetwork(TestCase):
    """EchoStateNetworkの単体テスト"""
    def test_initialization(self):
        """値の初期化がうまくいっているか確かめる"""
        density = 0.2
        rho_w = 0.95
        network = EchoStateNetwork(5, 100, 2, density=0.2, rho_w=0.95)
        calculated_rho_w = np.abs(np.linalg.eig(network.w_hidden)[0]).max()
        self.assertTrue(abs(calculated_rho_w - rho_w) < 1e-5)
        calculated_density = np.where(network.w_hidden == 0.0, 0.0, 1.0).mean()
        self.assertTrue(abs(calculated_density - density) < 0.01)
