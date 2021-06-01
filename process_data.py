#Thanks to Aladdin Persson for providing us with a simple way to distribute our images
#source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial20-classify-cancer-beginner-project-example/process_data.py

import os
import shutil
import random

seed = 1
random.seed(seed)


directory = "PASTE_DIRECTORY_PATH_TO_DATASET_HERE" + "/" # Modify directory path 
train = "data/train/"
test = "data/test/"
validation = "data/validation/"

# The following commands creates folders for the different folders
os.makedirs(train + "benign/")
os.makedirs(train + "malignant/")
os.makedirs(test + "benign/")
os.makedirs(test + "malignant/")
os.makedirs(validation + "benign/")
os.makedirs(validation + "malignant/")

test_examples = train_examples = validation_examples = 0

# distributing data
for line in open("PASTE_PATH_TO_.CSV_FILE_HERE").readlines()[1:]: # modify .csv path
    split_line = line.split(",")
    img_file = split_line[0]
    benign_malign = split_line[1]

    random_num = random.random()

    if random_num < 0.8:
        location = train
        train_examples += 1

    elif random_num < 0.9:
        location = validation
        validation_examples += 1

    else:
        location = test
        test_examples += 1

    # distributes labelled samples into malignant or benign folders    
    if int(float(benign_malign)) == 0: # benign samples
        shutil.copy(
            directory + img_file + ".jpg",
            location + "benign/" + img_file + ".jpg",
        )

    elif int(float(benign_malign)) == 1: # malignant samples 
        shutil.copy(
            directory + img_file + ".jpg",
            location + "malignant/" + img_file + ".jpg",
        )

# prints number of each sample type
print(f"Number of training examples {train_examples}")
print(f"Number of test examples {test_examples}")
print(f"Number of validation examples {validation_examples}")
