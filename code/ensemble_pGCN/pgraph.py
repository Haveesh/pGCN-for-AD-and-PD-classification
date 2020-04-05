
""" This file is to generate connectivity matrices (W) i.e. the population graph from the phenotypic information of subjects """


import os
import csv
import numpy as np
import scipy.io as sio
from nilearn import connectome


# Reading and computing the input data
# Input data variables

DTI_info_file = np.genfromtxt('./DTI_info.csv', delimiter=",", dtype = 'unicode')
MRI_info_file = np.genfromtxt('./MRI_info.csv', delimiter=",", dtype = 'unicode')

root_folder = '/path/to/data/'
phenotype = os.path.join(root_folder, './DTI_info.csv')


data_dir = '../'
out_folder = '../'

subjects = os.listdir(data_dir)

T1_img = 'T1-anatomical'
FA_map = 'FA_map-MRI'
DT_img = 'DTI_gated'

subject_condition = dict()

for row in DTI_info_file[1:]:
    subject_condition[row[1]] = row[2]

for subject in subjects:
    DT_img_info = dict()
    MR_img_info = dict()
    FA_img_info = dict()
    subject_available_modalities = os.listdir(os.path.join(data_dir, subject))
    subject_folder_names = []

    '''
    first convert all the dicom files to Nifti files
    This will also generate bval and bvec files for each DTI scan
    '''

    for modality in subject_available_modalities:

        if modality == DT_img or modality == 'DTI_GATED':
            available_sessions = os.listdir(os.path.join(data_dir, subject, modality))
            print ('Available DTI Sessions', available_sessions)
            for session in available_sessions:
                print('Trying session', session)
                date = session[0:10]

                subject_folder_name = subject + '_' + subject_condition[subject] + '_' + date
                
                if subject_folder_name in subject_folder_names:
                    subject_folder_name += session[10:]

                subject_folder_names.append(subject_folder_name)
                out_path = os.path.join(out_folder, subject_folder_name)

                current_path = os.path.join(data_dir, subject, modality, session)
                extra_folder = os.listdir(current_path)
                print (current_path, extra_folder)

                current_path = os.path.join(current_path, extra_folder[0])

                if not os.path.isdir(out_path) and not os.path.isfile(subject_folder_name + '.nii'):
                    os.makedirs(out_path)
                    print('Converting to Nifti from', current_path)
                    os.system('dcm2niix -z n -f ' + subject_folder_name + ' -o ' + str(out_path) + ' ' + str(current_path))
                else:
                    print('Folder/file already exists!')

        else:
            continue

    '''
    Now look for any other modality besides DTI. 
    We will only consider scans for other modalities if the corresponding DTI is present.
    '''
    for modality in subject_available_modalities:

        if modality == T1_img:
            available_sessions = os.listdir(os.path.join(data_dir, subject, modality))
            for session in available_sessions:
                date = session[0:10]
                
                subject_folder_name = subject + '_' + subject_condition[subject] + '_' + date
                
                for existing_DT_folder in subject_folder_names:
                    current_path = os.path.join(data_dir, subject, modality, session)
                    extra_folder = os.listdir(current_path)
                    current_path = os.path.join(current_path, extra_folder[0])
                    if subject_folder_name in existing_DT_folder:
                        copy_files_from_dir(current_path, os.path.join(out_folder, existing_DT_folder))


# Compute connectivity matrices
def subject_connectivity(subject, atlas_name, kind, save=True, save_path=out_folder):
    """
        
        subject      : the subject ID
        atlas_name   : name of the parcellation atlas used
        kind         : the kind of connectivity to be used, e.g. lasso, partial correlation, correlation
        save         : save the connectivity matrix to a file
        save_path    : specify path to save the matrix if different from subject folder

    returns:
        connectivity : connectivity matrix
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    if save:
        subject_file = os.path.join(save_path, subject,
                                    subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


# Get the list of subject IDs
def get_ids(num_subjects=None):
    """

    return:
        subject_IDs    : list of all subject IDs
    """

    subject_IDs = np.genfromtxt(os.path.join(data_folder, 'subject_IDs.txt'), dtype=str)

    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]

    return subject_IDs


# Get phenotype values for a list of subjects
def get_subject_score(subject_list, score):
    scores_dict = {}

    with open(phenotype) as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['SUB_ID'] in subject_list:
                scores_dict[row['SUB_ID']] = row[score]

    return scores_dict


# Make sure gender of each subject is represented in the training set when selecting a subset of the training set
def gender_percentage(train_ind, perc, subject_list):
    """
        train_ind    : indices of the training samples
        perc         : percentage of training set used
        subject_list : list of subject IDs

    return:
        labeled_indices      : indices of the subset of training samples
    """

    train_list = subject_list[train_ind]
    gender = get_subject_score(train_list, score='gender_id')
    unique = np.unique(list(sites.values())).tolist()
    gender = np.array([unique.index(age[train_list[x]]) for x in range(len(train_list))])

    labeled_indices = []

    for i in np.unique(age):
        id_in_gender = np.argwhere(gender == i).flatten()

        num_nodes = len(id_in_gender)
        labeled_num = int(round(perc * num_nodes))
        labeled_indices.extend(train_ind[id_in_gender[:labeled_num]])

    return labeled_indices


# Construct the adjacency matrix of the population from phenotypic scores
def create_affinity_graph_from_scores(scores, subject_list):
    """
        scores       : list of phenotypic information to be used to construct the graph
        subject_list : list of subject IDs

    return:
        graph        : adjacency matrix of the population graph (num_subjects x num_subjects)
    """

    num_nodes = len(subject_list)
    graph = np.zeros((num_nodes, num_nodes))

    for l in scores:
        label_dict = get_subject_score(subject_list, l)

        # quantitative phenotypic scores
        if l in ['AGE_AT_SCAN', 'FIQ']:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    try:
                        val = abs(float(label_dict[subject_list[k]]) - float(label_dict[subject_list[j]]))
                        if val < 2:
                            graph[k, j] += 1
                            graph[j, k] += 1
                    except ValueError:  # missing label
                        pass

        else:
            for k in range(num_nodes):
                for j in range(k + 1, num_nodes):
                    if label_dict[subject_list[k]] == label_dict[subject_list[j]]:
                        graph[k, j] += 1
                        graph[j, k] += 1

    return graph
