# TEMCOIN
Code base for Counting Origamis In Nanostructures in TEM Images

----------------

TEMCOIN provides benchmarking CNN models to predict ligation numbers in DNA origami nanostructures.

This repository includes the following content:
Training and validation datasets in folders: simulation/ and TEM/.
Testing unknown datasets in folders: predictVMD/ and predictTEM/.
The python3 codes are in the folder: DNA_origami_classification_MLAI/.
All the 10 different runs are in the folder: results_plots/ with the python code to do data analysis and plot.

<hr>

# Installation
----------------
Run the script for installation.
```bash
$ curl -fsSL https://github.com/rxhernandez/TEMCOIN/blob/main/TEMCOIN_install.sh | bash
```
You may also install TEMCOIN manually: 
use the download option to download the TEMCOIN.zip then unzip TEMCOIN.zip 
```
$ cd TEMCOIN
$ module load anaconda
$ conda activate environ_xyz
```

TEMCOIN requires Python 3.8 or later to run. You can install the TEMCOIN package from the source distribution (sdist). For this, first ensure you have the build python package.
```
$ pip install --upgrade build
```
Once the build package is installed, run the following command in the terminal.
```
$ python -m build --wheel
```
This command runs the build backend (in this case, setuptools), which copies all the sdist, gets the package dependencies and creates a python wheel (.whl) file. The python wheel is essentially a zip file with a specially formatted name and comes in a ready to install format with pip. Running the below command should install TEMCOIN in your virtual environment. As an example, we upload the wheel file that we have built in /dist/TEMCOIN-0.1.0-py3-none-any.whl 

To run the install the whl file, your system needs to downgrade importlib-metadata to a version compatible with zipp 3.17.0
```
$ pip install importlib-metadata==4.12.0
```
Then install TEMCOIN
```
$ pip install dist/TEMCOIN-0.1.0-py3-none-any.whl
```
<hr>

# Unittest
----------------
The test runs are in the TEMCOIN/test/ folder.

Documentation
----------------

This repository contains code to implement TEMCOIN. 
It is a set of python scripts that runs namd in unix enviroments.

* For details on the underlying theory and methods,
please refer to the following paper:
> DNA Origami Nanostructures Observed in Transmission Electron Microscopy Images can be Characterized through Convolutional Neural Networks (in preparation).

Authors: 
> Xingfei Wei, Qiankun Mo, Chi Chen, Mark Bathe, and Rigoberto Hernandez.

* For details on how to use TEMCOIN please refer to the 
[user_guide.md](https://github.com/rxhernandez/TEMCOIN/blob/main/user_guide.md) in the docs folder.

* Any questions or comments please reach out via email
to the authors of the paper.


<hr>

Authors
----------------

The TEMCOIN codes were debeloped by Xingfei Wei, Qiankun Mo, and Rigoberto Hernandez.

Contributors can be found [here](https://github.com/rxhernandez/TEMCOIN/graphs/contributors).

<hr>

Citing
----------------

If you use database or codes, please cite the paper:

>X. Wei, Q. Mo, C. Chen, M. Bathe and R. Hernandez, "DNA Origami Nanostructures Observed in Transmission Electron Microscopy Images can be Characterized through Convolutional Neural Networks," (in preparation).

and/or this site:

>X. Wei, Q. Mo and R. Hernandez, TEMCOIN, URL, [https://github.com/rxhernandez/TEMCOIN](https://github.com/rxhernandez/TEMCOIN)

<hr>

Acknowledgment
----------------

This work was supported by the National Science Foundation through Grant No.~CHE 1112067.


<hr>

License
----------------

NestedAE code and databases are distributed under terms of the [MIT License](https://github.com/rxhernandez/TEMCOIN/blob/main/LICENSE).

