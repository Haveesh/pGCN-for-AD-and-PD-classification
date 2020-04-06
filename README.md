# Ensemble learning of population based GCN for AD and PD Classification.


This project is the implementation of population based GCN (pGCN) based on Parisot, S et.Al's  (https://arxiv.org/abs/1703.03020) on DTI data of ADNI and PPMI databases. We further use ensemble learning approach to boost the classification perfromance of pGCN.

We provide an implementation applied to the [ADNI dataset](adni.loni.usc.edu/) for Alzheimer's disease diagnosis.

Implementation on PPMI dataset for Parkinson's disease classification can also be done on similar basis.


#### INSTRUCTIONS TO RUN
The data folder contains the samples of structural connectivity matrices(.npy files) of four subjects with corresponding subject ids These files are generated after the raw DTI data from ADNI repository is pre-processed. It contains the data lists for training, and the MRI and DTI data description files from ADNI and PPMI databases.

There are two subfolders in the code folder. 'pGCN' is the implementation without ensemble learning and 'ensemble_pGCN' is the implementation of ensemble learning on pGCN. 

To run the programme, you will need to install the implementation of graph convolutional networks (GCN) by Kipf et al.

The root folder in train_pGCN.py has to be updated to the folder were the data will be stored. 

To run the programme with default parameters execute the main_ADNI file.

Please note that the file names are to be kept as they are named and not to be changed.

#### REQUIREMENTS 

networkx <br />
nilearn <br />
scikit-learn <br />
joblib <br />
Keras==2.2.4 <br />
Keras-Applications==1.0.8 <br />
Keras-Preprocessing==1.1.0 <br />
kiwisolver==1.1.0 <br />
kneed==0.5.0 <br />
Markdown==3.2.1 <br />
matplotlib==3.0.3 <br />
numpy==1.16.0 <br />
powerlaw==1.4.4 <br />
protobuf==3.11.3 <br />
pyparsing==2.4.6 <br />
python-dateutil==2.8.1 <br />
PyYAML==5.3 <br />
scikit-learn==0.22.1 <br />
scipy==1.2.0 <br />
six==1.14.0 <br />
sklearn==0.0 <br />
tensorboard==1.12.2 <br />
tensorflow-gpu==1.12.0 <br />
termcolor==1.1.0 <br />
Werkzeug==1.0.0 <br />
