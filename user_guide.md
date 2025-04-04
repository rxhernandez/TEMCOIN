# Documentation & Installation

Documentation
----------------

This repository contains code to implement TEMCOIN. 
It is a set of python scripts that runs namd in unix enviroments.

* For details on the underlying theory and methods,
please refer to the following paper:
> DNA Origami Nanostructures Observed in Transmission Electron Microscopy Images can be Characterized through Convolutional Neural Networks (in preparation).

Authors: 
> Xingfei Wei, Qiankun Mo, Chi Chen, Mark Bathe, and Rigoberto Hernandez.

* For additional details on how to use TEMCOIN please refer to the [doc folder](https://github.com/rxhernandez/TEMCOIN/blob/main/doc).

* Any questions or comments please reach out via email
to the authors of the paper.


Installation
----------------

Run the script for fast installation.
```bash
$ curl -fsSL https://github.com/rxhernandez/TEMCOIN/blob/main/TEMCOIN_install.sh | bash
```

The two folders build/ and dist/ are not necessary for install TEMCOION.
These two files are just an example of that if you have successfully installed TEMCOION you should have created build/ and dist/  by yourself.

Install TEMCOIN manually: 
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


Unittest
----------------

The testing codes are provided in the folder: 
```
TEMCOIN/tests/212
```


Authors
----------------

The TEMCOIN codes were developed by Xingfei Wei, Qiankun Mo, and Rigoberto Hernandez.

Contributors can be found [here](https://github.com/rxhernandez/TEMCOIN/blob/main/contributors).

<hr>

