---
title: Training, Validation, and Test Datasets
date: 2024-08-26
---

You may have come across the terms **"training set"**, **"validation set"**, and **"test set"** in the context of
machine learning, and it might not be immediately clear what distinguishes them, particularly the difference between
the validation and test sets. Fortunately, the distinction is straightforward once you understand the roles each plays
in the model development process.

## Training Dataset

The training dataset is the portion of the dataset used to train the model. This is the data that the model actually
learns from: the model iteratively processes this data, calculating losses and adjusting its internal weights
accordingly to improve its predictions.

## Validation Dataset

The validation dataset is used during the training process, but it serves a different purpose from the training
dataset. The model does not learn from this data. Instead, it is used during training to check if the model is
overfitting (performing poorly only on unseen data). This dataset provides quick feedback, allowing you to rethink
hyperparameters and possibly even the model architecture.

## Test Dataset

Once training is fully completed using both the training and validation datasets, you have a finished model. Now, it's
time to assess how well the model performs. This is where the test dataset comes into play. The test dataset contains
data that the model has never seen during the training process, either for learning or validation. Its primary purpose
is to evaluate the model's performance and generalization ability on completely new data.
Analogy

## TLDR

**The validation set is used during training to monitor how accuracy improves as training progresses. The test set is
used after training is complete to evaluate how accurate the produced model is.**

Think of the validation dataset as analogous to development tests in software engineering, where the goal is to catch
errors and make improvements during the development process. In contrast, the test dataset is like a QA test conducted
after development is complete, ensuring that the final product meets the required standards before it is deployed.

## Suggested Split

![](/images/training-validation-and-test-datasets/ratio.png)

Not unlike many other aspects of machine learning, the split ratio for each of the three datasets is somewhat
arbitrary. There is no definitive right or wrong amount to choose, with the most popular suggestions being
**80/10/10**, **70/15/15**, and **60/20/20**.
