import unittest

from utils.utils import constant_segmentation


class TestConstantSegmentation(unittest.TestCase):

    def test_add_smaller_segment(self):
        ts_size = 11
        window_size = 3
        start, end, num_segments = constant_segmentation(ts_size, window_size)
        self.assertCountEqual(start, [0, 3, 6, 9])
        self.assertCountEqual(end, [3, 6, 9, 11])
        self.assertEquals(num_segments, 4)

    def test_not_add_smaller_segment(self):
        ts_size = 10
        window_size = 3
        start, end, num_segments = constant_segmentation(ts_size, window_size)
        self.assertCountEqual(start, [0, 3, 6])
        self.assertCountEqual(end, [3, 6, 9])
        self.assertEquals(num_segments, 3)
