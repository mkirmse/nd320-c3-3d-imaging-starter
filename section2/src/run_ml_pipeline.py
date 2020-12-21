"""
This file contains code that will kick off training and testing processes
"""
import os
import json
from typing import List, Tuple

from sklearn.model_selection import train_test_split

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import LoadHippocampusData

class Config:
    """
    Holds configuration parameters
    """
    def __init__(self):
        self.name = "Basic_unet"
        self.root_dir = r"/home/matthias/projects/udacity/nd320-c3-3d-imaging-starter/section1/out/TrainingSet"
        self.n_epochs = 10
        self.learning_rate = 0.0002
        self.batch_size = 16
        self.patch_size = 64
        self.test_results_dir = "/home/matthias/projects/udacity/nd320-c3-3d-imaging-starter/section2/out"
        self.val_split = 0.1
        self.test_split = 0.2
        self.eval_only = False
        # use model from path if eval only - for testing purposes
        self.model_path = "/home/matthias/projects/udacity/nd320-c3-3d-imaging-starter/section2/out/2020-12-21_1001_Basic_unet/model.pth"

if __name__ == "__main__":
    # Get configuration

    c = Config()

    # Load data
    print("Loading data...")

    # TASK: LoadHippocampusData is not complete. Go to the implementation and complete it. 
    data = LoadHippocampusData(c.root_dir, y_shape = c.patch_size, z_shape = c.patch_size)


    # Create test-train-val split
    # In a real world scenario you would probably do multiple splits for 
    # multi-fold training to improve your model quality

    keys = range(len(data))

    # Here, random permutation of keys array would be useful in case if we do something like 
    # a k-fold training and combining the results. 

    split = dict()

    # TASK: create three keys in the dictionary: "train", "val" and "test". In each key, store
    # the array with indices of training volumes to be used for training, validation 
    # and testing respectively.
    # <YOUR CODE GOES HERE>
    train_ind, test_ind = train_test_split(list(keys), train_size = 1-c.test_split, test_size = c.test_split)
    train_ind, val_ind = train_test_split(train_ind, train_size = 1-c.val_split, test_size = c.val_split)
    split["train"] = train_ind
    split["val"] = val_ind
    split["test"] = test_ind

    # Set up and run experiment
    
    # TASK: Class UNetExperiment has missing pieces. Go to the file and fill them in
    exp = UNetExperiment(c, split, data)

    # You could free up memory by deleting the dataset
    # as it has been copied into loaders
    del data

    # run training
    if not c.eval_only:
        exp.run()

    # prep and run testing

    # TASK: Test method is not complete. Go to the method and complete it
    results_json = exp.run_test()

    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

