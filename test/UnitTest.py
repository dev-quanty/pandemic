import unittest
import numpy as np
from src.models import sir, seird, seirqhfd


class Test_Models(unittest.TestCase):
    def test_sir(self):
        res1 = sir(np.ones(3), None, np.ones(2))
        res2 = sir(np.array([0, 0, 1]), None, np.ones(2))
        res3 = sir(np.ones(3), None, [0.7, 1.4])

        np.testing.assert_array_equal(res1, np.array([-1, 0, 1]), err_msg='SIR - Allones')
        np.testing.assert_array_equal(res2, np.array([0, 0, 0]), err_msg='SIR - Terminate')
        np.testing.assert_array_almost_equal(res3, np.array([-0.7, -0.7, 1.4]), decimal=2, err_msg='SIR - Parametrs')

    def test_seird(self):
        res1 = seird(np.ones(5), None, np.ones(4))
        res2 = seird(np.array([0, 0, 0, 1, 1]), None, np.ones(4))
        res3 = seird(np.array([1, 0, 1, 0, 0]), None, np.ones(4))
        res4 = seird(np.ones(5), None, np.array([0.7, 1.4, 0.3, 0.9]))
        res5 = sir(np.zeros(3), None, np.zeros(2))

        np.testing.assert_array_equal(res1, np.array([-1, 0, 0, 0, 1]), err_msg='SEIRD - Allones')
        np.testing.assert_array_equal(res2, np.array([0, 0, 0, 0, 0]), err_msg='SEIRD - Terminate')
        np.testing.assert_array_equal(res3, np.array([-1, 1, -1, 0, 1]), err_msg='SEIRD - Initial conditions')
        np.testing.assert_array_almost_equal(res4, np.array([-0.7, -0.7, 1.1, 0.03, 0.27]), decimal=2,
                                             err_msg='SEIRD - Parameters')
        np.testing.assert_array_equal(res5, np.zeros(3), err_msg='SIR - Edge case: All zeros')

    def test_seirqhfd(self):
        res1 = seirqhfd(np.ones(9), None, np.ones(12))
        res2 = seirqhfd(np.array([0, 0, 0, 0, 0, 0, 1, 0, 1]), None, np.ones(12))
        res3 = seirqhfd(np.ones(9), None, np.array([0.7, 1.4, 0.3, 0.9, 0.5, 0.6, 0.4, 0.7, 0.8, 0.6, 1.8, 0.3]))
        res4 = seirqhfd(np.array([1, 0, 1, 0, 0, 0, 0, 0, 0]), None, np.ones(12))
        res5 = seirqhfd(np.zeros(9), None, np.zeros(12))

        np.testing.assert_array_equal(res1, np.array([-3, 1, -2, 0, 0, 1, 0, 2, 1]), err_msg='SEIRQHFD - Allones')
        np.testing.assert_array_equal(res2, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), err_msg='SEIRQHFD - Terminate')
        np.testing.assert_array_almost_equal(res3, np.array([-2.4, 1, -0.8, -0.4, 0.4, 0, -0.08, 1.98, 0.3]), decimal=2,
                                             err_msg='SEIRQHFD - Parameters')
        np.testing.assert_array_equal(res4, np.array([-1, 1, -3, 0, 1, 1, 0, 1, 0]),
                                      err_msg='SEIRQHFD - Initial conditions')
        np.testing.assert_array_equal(res5, np.zeros(9), err_msg='SIR - Edge case: All zeros')


if __name__ == '__main__':
    unittest.main()
