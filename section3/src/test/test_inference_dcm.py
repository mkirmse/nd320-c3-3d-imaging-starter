import unittest

from section3.src.inference_dcm import run_inference


class TestInference(unittest.TestCase):


    def test_run_inference(self):
        routing_folder = "/home/matthias/projects/udacity/nd320-c3-3d-imaging-starter/section3/src/test/data"
        run_inference(routing_folder = routing_folder, local_test=True)



if __name__ == '__main__':
    unittest.main()
