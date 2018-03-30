# iForest

## Introduction

Isolation Forest, also known as iForest, is a data structure for anomaly detection. Traditional model-based methods need to construct a profile of normal instances and identify the instances that do not conform to the profile as anomalies. The traditional methods are optimized for normal instances, so they may cause false alarms. They may not be suitable for large data sets as well, due to high complexities. iForest, different from traditional methods, is more robust in these situations.

This project contains an encapsulation of iForest in `scikit-learn` together with some data prerocessing tools. It is specifically used for **categorical data**. The idea of iForest is from Liu, Ting and Zhou's research.[1]

## Setup

This section contains a short instruction about how to configure and run this program.

### Prequisites

1. `Python (>= 3.3)`
2. `NumPy (>= 1.8.2) and SciPy (>= 0.13.3)`

If you don't have `NumPy` or `SciPy` on your system, you can install them with the following command.
~~~~
python3 -m pip install --user numpy scipy
~~~~
3. `Pandas`

`Pandas` is a data analysis library for Python. To install Pandas, you can use the following command.
~~~~
pip3 install pandas
~~~~
4. `scikit-learn`

The library `scikit-learn` contains a set of simple tools for machine learning.
~~~~
pip3 install scikit-learn
~~~~

5. Other missing packages

If you miss any other packages required by the listed prequisites, please follow the instructions to install them.

### How To Run

The `main` function is in `main.py`. You may run this script directly. You can also import iForest in your own projects.

## Technical Details

### Structure
This project contains a set of files. This section introduces the functions of these files and the core functions within the corresponding files.

#### 1. `main.py`
Script `main.py` contains the main function. It shows a sample procedure of using iForest. This script is also available for you to run the test in the paper [1].

##### i) `CatForest(filename,threshold)`

This function shows the procedure of using iForest to detect anomalies. It returns the `data frame` of anomalies and takes the following arguments as input:

`filename`: String type. The path of the data file. The file should be in **CSV** format.
`threshold`: The threshold of decision boundary in iForest. It ranges from **-1 to 0**. In my test this parameter is usually set to values in **-0.4 to -0.1**.

##### ii) `__main()`
This is the main function to call, but it is mainly used to measure the detection time and accuracy. It calls `CatForest` for all iForest related procedures.

#### 2. `iForest.py`
This script contains an encapsulation of the iForest function in `scikit-learn`. 

##### `iforest(category,user,threshold)`
This function encapsulates the `IsolationForest` class in `scikit-learn`. It returns an `array` with the decision boundaries smaller than `threshold`.

#### 3. `Preprocessing.py`
This script preprocesses the data. It contains the following two functions.

##### i) `preprocessing(data,topnkey,num)`
This function mainly transfers categorical data to numerical data. The original iForest algorithm works for numerical only so categorical data has to be pre-processed. 

Note that the columns are now hard-coded in this script, which is uneasy to fit different types of data. I will write a parser in the future to read the metadata from files.

##### ii) `add_flag(data_frame)`

This function adds flags to existing data. It introduces domain knowledge to the data and is only used for future testing purposes.

#### 4. `readindi.py`
This script is a file loader.

##### `readindividual(filename)`
This function reads the data from `CSV` files. It returns the file contents and top-n-keys. The input parameter `filename` should be the path to a valid **CSV** file.

### Build Your Own Tests
This sections introduces basic steps to apply iForest to your data.
#### 1. Read Data From CSV File
You can follow the following example to read the data.
~~~~python
data, topnkey = readindividual(filename)
~~~~

#### 2. Preprocess Data
The data is preprocessed for each value in the range of `topnkey`.
~~~~python
total_num_examples = 0
for num in range(1,len(topnkey)):
	category, user, predict = preprocessing(data, topnkey, num)
    total_num_examples += len(user)
~~~~

#### 3. Anomaly Detection
Tis process is also done for each value in the range of `topnkey`. It should be coded in the loop in `Step 2`.
~~~~python
u = iforest(category, user, threshold)
anomalies  = predict.iloc[u,:]
~~~~

## Future Work

1. Implement a parser for reading the metadata of the data in the CSV file.
2. Use domain knowledge to verify the classification.

## References
[1] Isolation Forest [https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)

[2] SciPy Official Site [https://www.scipy.org/about.html](https://www.scipy.org/about.html)

[3] Installing scikit-learn [http://scikit-learn.org/stable/install.html](http://scikit-learn.org/stable/install.html)

[4] Pandas [https://pandas.pydata.org](https://pandas.pydata.org)
