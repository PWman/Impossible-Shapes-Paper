# Impossible-Shapes-Publication

See the paper at https://www.sciencedirect.com/science/article/pii/S0042698921002017

This repository contains the code for obtaining the results in the paper above. The convolutional neural network (CNN) study and associated analysis was performed in Python. The analysis of the human participant data was in MATLAB.

### Repository Structure

The repository is structured in the following way:
- MATLAB code and associated analysis of human participant data can be found in the *Human Participant Study* directory.
- The Python code and analysis (CNN study) is contained in the main directory of this repository.
- The *Results* directory contains the results from the CNN study, as well as some additional analysis of images.
- The above directory contains results titled *Study 0*, *Study 1* and *Study 2*:
  - Study 0 is the circle-square control study
  - Study 1 contains results for the impossible shapes before background size was controlled
  - Study 2 contains results for the main study (impossible controlled for background size)
- The above directories contain some summary tables and figures of DNN performance.
- They also contain sub-directories with the more detailed results for each DNN (e.g. GradCAM results for each network).
- *Results/Shapes* contains the dataset of matched possible and impossible shapes.

### Running the Code

In order to run a copy of our CNN study locally, please pull a copy of the repository and then follow the instructions below. Note that we do not recommend running this study without a GPU as the training the DNNs is an extremely long and computationally intensive process. 
- Ensure necessary Python packages are installed (listed below).
- Run *train_test_utils.py*. This will train the CNNs and save the results and models to *Results/Raw*. This process may take a long time, even with a GPU, as there are a total of 12 networks (6 pretrained and 6 without pretraining) which need to be trained over 20 different seeds.
- If the above fails (returns an  RunTimeError: out of memory), it may be necessary to run the DNN training (and testing) separately for each network. In order to do this, please do the following:
  - Go to *train_test_utils.py* and delete all code below `if __name__ == "__main__":`
  - Write the following instead: `train_test_network(<DNN name>, study_num=<Study Number (i.e. 0/1/2)>)`
  - Run *test_train_utils.py* until completion and repeat with each CNN listed in the *config.py* (and for each study).
- Once this is done, run *process_results.py*. This will aggregate and compile all results, then save them to *Results/Study 1* and *Results/Study 2*.
- If desired, then run *IoU_values.py* to calculate background attention statistics, and *analysis_utils.py* to perform additional analysis (t-tests of possible vs impossible accuracy, etc.), and *gcam_all_layers_alexnet.py* to get gradCAM results for all layers of AlexNet.

#### Required Python Packages:
- NumPy (1.18.4)
- Pillow (7.1.2)
- OpenCV version (4.2.0)
- Matplotlib (3.1.2)
- Pandas (1.0.3)
- PyTorch (1.5.0)
- Seaborn (0.10.1)
- SciPy (1.4.1)
- Scikit-learn (0.23)
- Scikit-image (0.17.1)

Please note that additional packages may be needed in line with requirements of the above. Also note that we used the above package versions for running the results used in our paper, and while it may be possible to run the project with other versions, we cannot gauruntee compatibility.

### Results

#### Validation Accuracies Study 0: Square-Circle Control Study

CNNs (no pretraining) | Pretrained CNNs
------------ | -------------
![DNN Accuracy no pretraining study 0](https://github.com/PWman/Impossible-Shapes-Paper/blob/master/Results/Study%200/Validation%20Accuracies%20(no%20pretraining).png) | ![DNN Accuracy pretrained study 0](https://github.com/PWman/Impossible-Shapes-Paper/blob/master/Results/Study%200/Validation%20Accuracies%20(with%20pretraining).png)

#### Validation Accuracies Study 1: Impossible Shapes (no background control)

CNNs (no pretraining) | Pretrained CNNs
------------ | -------------
![DNN Accuracy no pretraining study 1](https://github.com/PWman/Impossible-Shapes-Paper/blob/master/Results/Study%201/Validation%20Accuracies%20(no%20pretraining).png) | ![DNN Accuracy pretrained study 1](https://github.com/PWman/Impossible-Shapes-Paper/blob/master/Results/Study%201/Validation%20Accuracies%20(with%20pretraining).png)


#### Validation Accuracies Study 2: Impossible Shapes (main study)

CNNs (no pretraining) | Pretrained CNNs
------------ | -------------
![DNN Accuracy no pretraining study 2](https://github.com/PWman/Impossible-Shapes-Paper/blob/master/Results/Study%202/Validation%20Accuracies%20(no%20pretraining).png) | ![DNN Accuracy pretrained study 2](https://github.com/PWman/Impossible-Shapes-Paper/blob/master/Results/Study%202/Validation%20Accuracies%20(with%20pretraining).png) 

