# Data Version Control Tutorial

Example repository for the [Data Version Control With Python and DVC](https://realpython.com/python-data-version-control/) 

To use this repo as part of the tutorial, you first need to get your own copy. Click the _Fork_ button in the top-right corner of the screen, and select your private account in the window that pops up. GitHub will create a forked copy of the repository under your account.

Clone the forked repository to your computer with the `git clone` command

```bash
git clone git@github.com:YourUsername/data-version-control.git
```

Make sure to replace `YourUsername` in the above command with your actual GitHub username.

Machine learning and data science come with a set of problems that are different from what you’ll find in traditional software engineering. Version control systems help developers manage changes to source code. But data version control, managing changes to models and datasets, isn’t so well established.
It’s not easy to keep track of all the data you use for experiments and the models you produce. Accurately reproducing experiments that you or others have done is a challenge. Many teams are actively developing tools and frameworks to solve these problems

In this handson, you’ll learn how to:
•	Use a tool called DVC to tackle some of these challenges
•	Track and version your datasets and models
•	Share a single development computer between teammates
•	Create reproducible machine learning experiments

## Part 1 - Environment Setup

Clone the forked repository to your computer with the git clone command and position your command line inside the repository folder:

```bash
git clone git@github.com:YourUsername/data-version-control.git

cd data-version-control
```
Don’t forget to replace YourUsername in the above command with your actual username. You should now have a clone of the repository on your computer.
Here’s the folder structure for the repository:

data-version-control/
|
├── data/
│   ├── prepared/
│   └── raw/
|
├── metrics/
├── model/
└── src/
    ├── evaluate.py
    ├── prepare.py
    └── train.py

There are six folders in your repository:
1.	src/ is for source code.
2.	data/ is for all versions of the dataset.
3.	data/raw/ is for data obtained from an external source.
4.	data/prepared/ is for data modified internally.
5.	model/ is for machine learning models.
6.	data/metrics/ is for tracking the performance metrics of your models.

The src/ folder contains three Python files:
1.	prepare.py contains code for preparing data for training.
2.	train.py contains code for training a machine learning model.
3.	evaluate.py contains code for evaluating the results of a machine learning model.

The final step in the preparation is to get an example dataset you can use to practice DVC. Images are well suited for this particular tutorial because managing lots of large files is where DVC shines, so you’ll get a good look at DVC’s most powerful features. You’ll use the Imagenette dataset from fastai.
Imagenette is a subset of the ImageNet dataset, which is often used as a benchmark dataset in many machine learning papers. ImageNet is too big to use as an example on a laptop, so you’ll use the smaller Imagenette dataset. Go to the Imagenette GitHub page and click 160 px download in the README.

[https://github.com/fastai/imagenette]

This will download the dataset compressed into a TAR archive. Mac users can extract the files by double-clicking the archive in the Finder. Linux users can unpack it with the tar command. Windows users will need to install a tool that unpacks TAR files, like 7-zip.
The dataset is structured in a particular way. It has two main folders:

1. train/ includes images used for training a model.
2. val/ includes images used for validating a model.

The train/ and val/ folders are further divided into multiple folders. Each folder has a code that represents one of the 10 possible classes, and each image in this dataset belongs to one of ten classes:
1.	Tench
2.	English springer
3.	Cassette player
4.	Chain saw
5.	Church
6.	French horn
7.	Garbage truck
8.	Gas pump
9.	Golf ball
10.	Parachute
For simplicity and speed, you’ll train a model using only two of the ten classes, golf ball and parachute. When trained, the model will accept any image and tell you whether it’s an image of a golf ball or an image of a parachute. This kind of problem, in which a model decides between two kinds of objects, is called binary classification.

Copy the train/ and val/ folders and put them into your new repository, in the data/raw/ folder. Your repository structure should now look like this:
data-version-control/
|
├── data/
│   ├── prepared/
│   └── raw/
│       ├── train/
│       │   ├── n01440764/
│       │   ├── n02102040/
│       │   ├── n02979186/
│       │   ├── n03000684/
│       │   ├── n03028079/
│       │   ├── n03394916/
│       │   ├── n03417042/
│       │   ├── n03425413/
│       │   ├── n03445777/
│       │   └── n03888257/
|       |
│       └── val/
│           ├── n01440764/
│           ├── n02102040/
│           ├── n02979186/
│           ├── n03000684/
│           ├── n03028079/
│           ├── n03394916/
│           ├── n03417042/
│           ├── n03425413/
│           ├── n03445777/
│           └── n03888257/
|
├── metrics/
├── model/
└── src/
    ├── evaluate.py
    ├── prepare.py
    └── train.py


You’ll also use some external libraries in this tutorial:

1.	dvc is the star of this tutorial.
2.	scikit-learn is a machine learning library that allows you to train models.
3.	scikit-image is an image processing library that you’ll use to prepare data for training.
4.	pandas is a library for data analysis that organizes data in table-like structures.
5.	numpy is a numerical computing library that adds support for multidimensional data, like images.


```bash
python -m pip install dvc scikit-learn scikit-image pandas numpy
```

This will install all dependencies.

## Part 2 - Creating Pipeline

Create a new branch and call it sgd-pipeline:


```bash
git checkout -b sgd-pipeline
```

Next, you need to initialize DVC. Make sure you’re positioned in the top-level folder of the repository, then run dvc init:

```bash
dvc init
```
This will create a .dvc folder that holds configuration information, just like the .git folder for Git. In principle, you don’t ever need to open that folder.

Note: DVC has recently started collecting anonymized usage analytics so the authors can better understand how DVC is used. This helps them improve the tool. You can turn it off by setting the analytics configuration option to false:

```bash
dvc config core.analytics false
```

Git gives you the ability to push your local code to a remote repository so that you have a single source of truth shared with other developers. Other people can check out your code and work on it locally without fear of corrupting the code for everyone else. The same is true for DVC.

You need some kind of remote storage for the data and model files controlled by DVC. This can be as simple as another folder on your system. Create a folder somewhere on your system outside the data-version-control/ repository and call it dvc_remote.

Now come back to your data-version-control/ repository and tell DVC where the remote storage is on your system:

```bash
dvc remote add -d remote_storage path/to/your/dvc_remote
```

DVC now knows where to back up your data and models. dvc remote add stores the location to your remote storage and names it remote_storage. You can choose another name if you want. The -d switch tells DVC that this is your default remote storage. You can add more than one storage location and switch between them.

Your repository is now initialized and ready for work. You’ll cover three basic actions:
1.	Tracking files
2.	Uploading files
3.	Downloading files
The basic rule of thumb you’ll follow is that small files go to GitHub, and large files go to DVC remote storage.
 

Tracking Files

Both Git and DVC use the add command to start tracking files. This puts the files under their respective control.
Add the train/ and val/ folders to DVC control:

```bash
dvc add data/raw/train
dvc add data/raw/val
```

Images are considered large files, especially if they’re collected into datasets with hundreds or thousands of files. The add command adds these two folders under DVC control. Here’s what DVC does under the hood:

1.	Adds your train/ and val/ folders to .gitignore
2.	Creates two files with the .dvc extension, train.dvc and val.dvc
3.	Copies the train/ and val/ folders to a staging area
This process is a bit complex and warrants a more detailed explanation.

Finally, DVC copies the data files to a staging area. The staging area is called a cache. When you initialized DVC with dvc init, it created a .dvc folder in your repository. In that folder, it created the cache folder, .dvc/cache. When you run dvc add, all the files are copied to .dvc/cache.



You’ll use this branch to run the experiment as a DVC pipeline. A pipeline consists of multiple stages and is executed using a dvc run command. Each stage has three components:
1.	Inputs
2.	Outputs
3.	Command
DVC uses the term dependencies for inputs and outs for outputs. The command can be anything you usually run in the command line, including Python files. You can practice creating a pipeline of stages while running another experiment. Each of your three Python files, prepare.py, train.py, and evaluate.py will be represented by a stage in the pipeline.

First, you’re going to run prepare.py as a DVC pipeline stage. The command for this is dvc run, which needs to know the dependencies, outputs, and command:
1.	Dependencies: prepare.py and the data in data/raw
2.	Outputs: train.csv and test.csv
3.	Command: python prepare.py
Execute prepare.py as a DVC pipeline stage with the dvc run command:


```bash
dvc run -n prepare \
          -d src/prepare.py -d data/raw \
          -o data/prepared/train.csv -o data/prepared/test.csv \
          python src/prepare.py
```

All this is a single command. The first row starts the dvc run command and accepts a few options:
•	The -n switch gives the stage a name.
•	The -d switch passes the dependencies to the command.
•	The -o switch defines the outputs of the command.

The main argument to the command is the Python command that will be executed, python src/prepare.py. In plain English, the above dvc run command gives DVC the following information:

•	Line 1: You want to run a pipeline stage and call it prepare.
•	Line 2: The pipeline needs the prepare.py file and the data/raw folder.
•	Line 3: The pipeline will produce the train.csv and test.csv files.
•	Line 4: The command to execute is python src/prepare.py.

Great—you’ve automated the first stage of the pipeline.

The First Stage of the Pipeline
You’ll use the CSV files produced by this stage in the following stage.
The next stage in the pipeline is training. The dependencies are the train.py file itself and the train.csv file in data/prepared. The only output is the model.joblib file. To create a pipeline stage out of train.py, execute it with dvc run, specifying the correct dependencies and outputs:


```bash
dvc run -n train \
        -d src/train.py -d data/prepared/train.csv \
        -o model/model.joblib \
        python src/train.py
```

This will create the second stage of the pipeline and record it in the dvc.yml and dvc.lock files.

The Second Stage of the Pipeline
Two down, one to go! The final stage will be the evaluation. The dependencies are the evaluate.py file and the model file generated in the previous stage. The output is the metrics file, accuracy.json. Execute evaluate.py with dvc run

```bash
dvc run -n evaluate \
        -d src/evaluate.py -d model/model.joblib \
        -M metrics/accuracy.json \
        python src/evaluate.py
```

Notice that you used the -M switch instead of -o. DVC treats metrics differently from other outputs. When you run this command, it will generate the accuracy.json file, but DVC will know that it’s a metric used to measure the performance of the model.
You can get DVC to show you all the metrics it knows about with the dvc show command:


```bash
dvc metrics show
```

the output is just like this:

        metrics/accuracy.json:
            accuracy: 0.6996197718631179

You’ve completed the final stage of the pipeline.

The Full Pipeline
You can now see your entire workflow in a single image. Don’t forget to tag your new branch and push all the changes to GitHub and DVC:

```bash
git add --all
git commit -m "Run SGD as pipeline"
dvc commit
git push --set-upstream origin sgd-pipeline
git tag -a sgd-pipeline -m "Trained SGD as DVC pipeline."
git push origin --tags
dvc push
```

This will version and store your code, models, and data for the new DVC pipeline.


Now for the best part!

## Part 3 - Change the Algorithms to Random Forest
For training, you’ll use a random forest classifier, which is a different model that can be used for classification. It’s more complex than the SGDClassifier and could potentially yield better results. Start by creating and checking out a new branch and calling it random_forest:

```bash
git checkout -b "random_forest"
```

The power of pipelines is the ability to reproduce them with minimal hassle whenever you change anything. Modify your train.py to use a RandomForestClassifier instead of the SGDClassifier:

```train.py
from joblib import dump
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.io import imread_collection
from skimage.transform import resize
from sklearn.ensemble import RandomForestClassifier                    #this line
# ...

def main(path_to_repo):
    train_csv_path = repo_path / "data/prepared/train.csv"
    train_data, labels = load_data(train_csv_path)
    rf = RandomForestClassifier()                                      #this line
    trained_model = rf.fit(train_data, labels)                         #this line
    dump(trained_model, repo_path / "model/model.joblib")
```

The only lines that changed were importing the RandomForestClassifier instead of the SGDClassifier, creating an instance of the classifier, and calling its fit() method. Everything else remained the same.
Since your train.py file changed, its MD5 hash has changed. DVC will realize that one of the pipeline stages needs to be reproduced. You can check what changed with the dvc status command:


```bash
dvc status
```

Output is like this:

train:
    changed deps:
        modified:           src/train.py


This will display all the changed dependencies for every stage of the pipeline. Since the change in the model will affect the metric as well, you want to reproduce the whole chain. You can reproduce any DVC pipeline file with the dvc repro command:

```bash
dvc repro evaluate
```

And that’s it! When you run the repro command, DVC checks all the dependencies of the entire pipeline to determine what’s changed and which commands need to be executed again. Think about what this means. You can jump from branch to branch and reproduce any experiment with a single command!
To wrap up, push your random forest classifier code to GitHub and the model to DVC:


```bash
git add --all
git commit -m "Train Random Forrest classifier"
dvc commit
git push --set-upstream origin random_forest
git tag -a random_forest -m "Random Forest classifier accuracy."
git push origin --tags
dvc push
```

Now you can compare metrics across multiple branches and tags.
Call dvc metrics show with the -T switch to display metrics across multiple tags:

```bash
dvc metrics show -T
```
Output is like this:

sgd-pipeline:
    metrics/accuracy.json:
        accuracy: 0.6996197718631179
forest:
    metrics/accuracy.json:
        accuracy: 0.8098859315589354


Awesome! This gives you a quick way to keep track of what the best-performing experiment was in your repository.
When you come back to this project in six months and don’t remember the details, you can check which setup was the most successful with dvc metrics show -T and reproduce it with dvc repro! Anyone else who wants to reproduce your work can do the same. They’ll just need to take three steps:
1.	Run git clone or git checkout to get the code and .dvc files.
2.	Get the training data with dvc checkout.
3.	Reproduce the entire workflow with dvc repro evaluate.


Nice work! You ran multiple experiments and safely versioned and backed up the data and models. What’s more, you can quickly reproduce each experiment by just getting the necessary code and data and executing a single dvc repro command.


## Part 4 - Links


1.   [https://github.com/fastai/imagenette]

2.   [https://github.com/ibrahimkaratas88/data-version-control]

3.    [https://github.com/realpython/data-version-control]
