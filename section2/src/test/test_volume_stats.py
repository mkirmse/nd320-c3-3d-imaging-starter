from unittest import TestCase
import numpy as np

from utils.volume_stats import Dice3d, Jaccard3d


class Test(TestCase):

    def setUp(self) -> None:
        self.vol1 = np.zeros((3, 3, 3))
        self.vol1[1, 1, 1] = 1
        self.vol1[1, 1, 2] = 1
        self.vol1[2, 1, 0] = 1

        self.vol2 = np.zeros((3, 3, 3))
        self.vol2[1, 1, 1] = 1
        self.vol2[1, 1, 2] = 1
        self.vol2[2, 0, 0] = 1
        self.vol2[2, 2, 0] = 1

    def test_dice3d(self):
        self.assertEqual(Dice3d(self.vol1, self.vol2), 4 / 7)

    def test_jaccard3d(self):
        self.assertEqual(Jaccard3d(self.vol1, self.vol2), 2 / 5)


