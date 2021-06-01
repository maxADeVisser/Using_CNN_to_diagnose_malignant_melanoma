#Thanks to Aladdin Persson for providing us with a simple way to distribute our images
#source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial20-classify-cancer-beginner-project-example/process_data.py

import os
import shutil
import random

seed = 1
random.seed(seed)

directory = "PASTE_YOUR_DIRECTORY_PATH_HERE" 
train = "data/train/"
test = "data/test/"
validation = "data/validation/"

# The following commands creates the folders
os.makedirs(train + "benign/")
os.makedirs(train + "malignant/")
os.makedirs(test + "benign/")
os.makedirs(test + "malignant/")
os.makedirs(validation + "benign/")
os.makedirs(validation + "malignant/")

test_examples = train_examples = validation_examples = 0

for line in open("ISIC/labels.csv").readlines()[1:]:
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

    if int(float(benign_malign)) == 0:
        shutil.copy(
            "ISIC/images/" + img_file + ".jpg",
            location + "benign/" + img_file + ".jpg",
        )

    elif int(float(benign_malign)) == 1:
        shutil.copy(
            "ISIC/images/" + img_file + ".jpg",
            location + "malignant/" + img_file + ".jpg",
        )

print(f"Number of training examples {train_examples}")
print(f"Number of test examples {test_examples}")
print(f"Number of validation examples {validation_examples}")
