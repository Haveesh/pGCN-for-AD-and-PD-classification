"""This is to run the program with default set parameters for ADNI dataset"""


import argparse
import numpy as np
from scipy import sparse
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from sklearn.linear_model import RidgeClassifier
import sklearn.metrics
import scipy.io as sio


import pgraph as Reader
import train_pGCN as Train

# Prepares the training/test data for each cross validation fold and trains the pGCN
def train_fold(train_ind, test_ind, val_ind, graph_feat, features, y, y_data, params, subject_IDs):
    """
        train_ind       : indices of the training samples
        test_ind        : indices of the test samples
        val_ind         : indices of the validation samples
        graph_feat      : population graph computed from phenotypic measures num_subjects x num_subjects
        features        : feature vectors num_subjects x num_features
        y               : ground truth labels (num_subjects x 1)
        y_data          : ground truth labels - different representation (num_subjects x 2)
        params          : dictionary of pGCN parameters
        subject_IDs     : list of subject IDs

    returns:

        test_acc    : average accuracy over the test samples using pGCNs
        test_auc    : average area under curve over the test samples using pGCNs
        lin_acc     : average accuracy over the test samples using the linear classifier
        lin_auc     : average area under curve over the test samples using the linear classifier
        fold_size   : number of test samples
    """

    print(len(train_ind))

    # selection of a subset of data if running experiments with a subset of the training set
    labeled_ind = Reader.site_percentage(train_ind, params['num_training'], subject_IDs)

    fold_size = len(test_ind)

    # Calculate all pairwise distances
    distv = distance.pdist(x_data, metric='correlation')
    # Convert to a square symmetric distance matrix
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    # Get affinity from similarity matrix
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph_feat * sparse_graph

    
    # Classification with pGCN
    test_acc, test_auc = Train.run_training(final_graph, sparse.coo_matrix(x_data).tolil(), y_data, train_ind, val_ind,
                                            test_ind, params)

    print(test_acc)

    
    test_acc = int(round(test_acc * len(test_ind)))

    return test_acc, test_auc , fold_size


def main():
    parser = argparse.ArgumentParser(description='Graph CNNs for population graphs: '
                                                 'classification of the ADNI dataset')
    parser.add_argument('--dropout', default=0.3, type=float,
                        help='Dropout rate (1 - keep probability) (default: 0.3)')
    parser.add_argument('--decay', default=5e-4, type=float,
                        help='Weight for L2 loss on embedding matrix (default: 5e-4)')
    parser.add_argument('--hidden', default=16, type=int, help='Number of filters in hidden layers (default: 16)')
    parser.add_argument('--lrate', default=0.005, type=float, help='Initial learning rate (default: 0.005)')
    parser.add_argument('--atlas', default='ho', help='atlas for network construction (node definition) (default: ho, '
                                                      
    parser.add_argument('--epochs', default=150, type=int, help='Number of epochs to train')
    parser.add_argument('--num_features', default=6786, type=int, help='Number of features to keep for '
                                                                       'the feature selection step (default: 6786)')
    parser.add_argument('--num_training', default=1.0, type=float, help='Percentage of training set used for '
                                                                        'training (default: 1.0)')
    parser.add_argument('--depth', default=0, type=int, help='Number of additional hidden layers in the pGCN. '
                                                             'Total number of hidden layers: 1+depth (default: 0)')
    parser.add_argument('--model', default='gcn_cheby', help='gcn model used (default: gcn_cheby, '
                                                             'uses chebyshev polynomials, '
                                                             'options: gcn, gcn_cheby, dense )')
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation (default: 123)')
    parser.add_argument('--folds', default=0, type=int, help='For cross validation, specifies which fold will be '
                                                             'used. All folds are used if set to 11 (default: 11)')
    parser.add_argument('--save', default=1, type=int, help='Parameter that specifies if results have to be saved. '
                                                            'Results will be saved if set to 1 (default: 1)')
    parser.add_argument('--connectivity', default='correlation', help='Type of connectivity used for network '
                                                                      'construction (default: correlation, '
                                                                      'options: correlation, partial correlation, '
                                                                      'tangent)')

    args = parser.parse_args()
    start_time = time.time()

    # pGCN Parameters
    params = dict()
    params['model'] = args.model                    # pgcn model using chebyshev polynomials
    params['lrate'] = args.lrate                    # Initial learning rate
    params['epochs'] = args.epochs                  # Number of epochs to train
    params['dropout'] = args.dropout                # Dropout rate (1 - keep probability)
    params['hidden'] = args.hidden                  # Number of units in hidden layers
    params['decay'] = args.decay                    # Weight for L2 loss on embedding matrix.
    params['early_stopping'] = params['epochs']     # Tolerance for early stopping (# of epochs). No early stopping if set to param.epochs
    params['max_degree'] = 3                        # Maximum Chebyshev polynomial degree.
    params['depth'] = args.depth                    # number of additional hidden layers in the pGCN. Total number of hidden layers: 1+depth
    params['seed'] = args.seed                      # seed for random initialisation

    # pGCN Parameters
    params['num_features'] = args.num_features      # number of features for feature selection step
    params['num_training'] = args.num_training      # percentage of training set used for training
    atlas = args.atlas                              # atlas for network construction (node definition)
    connectivity = args.connectivity                # type of connectivity used for network construction

    # Get class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Get acquisition site
    sites = Reader.get_subject_score(subject_IDs, score='SITE_ID')
    unique = np.unique(list(sites.values())).tolist()

    num_classes = 2
    num_nodes = len(subject_IDs)

    # Initialise variables for class labels and acquisition sites
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    site = np.zeros([num_nodes, 1], dtype=np.int)

    # Get class labels and acquisition site for all subjects
    for x in xrange(1,10):
         pass i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]])-1] = 1
        y[i] = int(labels[subject_IDs[i]])
        site[i] = unique.index(sites[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    features = Reader.get_networks(subject_IDs, kind=connectivity, atlas_name=atlas)

    # Compute population graph using gender and acquisition site
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)

    # Folds for cross validation experiments
    skf = StratifiedKFold(n_splits=10)

    if args.folds == 11:  # run cross validation on all folds
        scores = Parallel(n_jobs=10)(delayed(train_fold)(train_ind, test_ind, test_ind, graph, features, y, y_data,
                                                         params, subject_IDs)
                                     for train_ind, test_ind in
                                     reversed(list(skf.split(np.zeros(num_nodes), np.squeeze(y)))))

        print(scores)

        scores_acc = [x[0] for x in scores]
        scores_auc = [x[1] for x in scores]
        scores_lin = [x[2] for x in scores]
        scores_auc_lin = [x[3] for x in scores]
        fold_size = [x[4] for x in scores]

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / num_nodes))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / num_nodes))
        print('overall AUC %f' + str(np.mean(scores_auc)))

    else:  # compute results for only one fold

        cv_splits = list(skf.split(features, np.squeeze(y)))

        train = cv_splits[args.folds][0]
        test = cv_splits[args.folds][1]

        val = test

        scores_acc, scores_auc, scores_lin, scores_auc_lin, fold_size = train_fold(train, test, val, graph, features, y,
                                                         y_data, params, subject_IDs)

        print('overall linear accuracy %f' + str(np.sum(scores_lin) * 1. / fold_size))
        print('overall linear AUC %f' + str(np.mean(scores_auc_lin)))
        print('overall accuracy %f' + str(np.sum(scores_acc) * 1. / fold_size))
        #print('overall AUC %f' + str(np.mean(scores_auc)))

    if args.save == 1:
        result_name = 'ADNI_classification.mat'
        sio.savemat('/vol/medic02/users/sparisot/python/graphCNN/results/' + result_name + '.mat',
                    {'lin': scores_lin, 'lin_auc': scores_auc_lin,
                     'acc': scores_acc, 'auc': scores_auc, 'folds': fold_size})

if __name__ == "__main__":
    main()
