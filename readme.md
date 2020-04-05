

This project is implementation of population based GCN (pGCN) proposed by Parisot, S et.Al on DTI data of ADNI and PPMI. We further use ensemble learning approach to boost the classification perfromance of pGCN.

We provide an implementation applied to the [ADNI dataset](adni.loni.usc.edu/) for Alzheimer's disease diagnosis.

Implementation on PPMI dataset for Parkinson's disease classification can also be done on the same lines.


#### INSTRUCTIONS TO RUN
The data folder contains the sample structural connectivity matrices for a few subjects with corresponding ids generated from DTI data(.npy files). These files are generated after the raw DTI data from ADNI repository is pre-processed. It also contains the data lists for training. It also contains the MRI and DTI data description files of ADNI and PPMI databases,

There are two subfolders in the code folder. 'pGCN' is the normal implementation without ensemble learning and 'ensemble_pGCN' is the ensemble learning with pgcn. 

To run the programme, you will need to install the implementation of graph convolutional networks (GCN) by Kipf et al.

The root folder in train_pGCN.py has to be updated to the folder were the data will be stored. 

To run the programme with default parameters execute the main_ADNI file.

The file names are to be kept as they are named and not to be changed.

#### REQUIREMENTS 

tensorflow (>= 1.0) <br />
networkx <br />
nilearn <br />
scikit-learn <br />
joblib



