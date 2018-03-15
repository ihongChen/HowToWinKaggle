from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from sklearn.preprocessing import normalize
import numpy as np
import scipy.sparse 

class NearestNeighborsFeats(BaseEstimator, ClassifierMixin):
    '''
        This class should implement KNN features extraction 
    '''
    def __init__(self, n_jobs, k_list, metric, n_classes=None, n_neighbors=None, eps=1e-6):
        self.n_jobs = n_jobs 
        self.k_list = k_list # k: # of nearest neighbors, k_list --> list of k
        self.metric = metric ## distance measure
        
        if n_neighbors is None:
            self.n_neighbors = max(k_list) 
        else:
            self.n_neighbors = n_neighbors
            
        self.eps = eps
        self.n_classes_ = n_classes
    
    def fit(self, X, y):
        '''
            Set's up the train set and self.NN object
        '''
        # Create a NearestNeighbors (NN) object. We will use it in `predict` function 
        self.NN = NearestNeighbors(n_neighbors=max(self.k_list), 
                                      metric=self.metric, 
                                      n_jobs=1, 
                                      algorithm='brute' if self.metric=='cosine' else 'auto')
        self.NN.fit(X)
        
        # Store labels 
        self.y_train = y
        
        # Save how many classes we have
        self.n_classes = np.unique(y).shape[0] if self.n_classes_ is None else self.n_classes_
        
        
    def predict(self, X):       
        '''
            Produces KNN features for every object of a dataset X
        '''
        if self.n_jobs == 1:
            test_feats = []
            for i in range(X.shape[0]):
                test_feats.append(self.get_features_for_one(X[i:i+1]))
        else:
            '''
                 *Make it parallel*
                     Number of threads should be controlled by `self.n_jobs`  
                     
                     
                     You can use whatever you want to do it
                     For Python 3 the simplest option would be to use 
                     `multiprocessing.Pool` (but don't use `multiprocessing.dummy.Pool` here)
                     You may try use `joblib` but you will most likely encounter an error, 
                     that you will need to google up (and eventually it will work slowly)
                     
                     For Python 2 I also suggest using `multiprocessing.Pool` 
                     You will need to use a hint from this blog 
                     http://qingkaikong.blogspot.ru/2016/12/python-parallel-method-in-class.html
                     I could not get `joblib` working at all for this code 
                     (but in general `joblib` is very convenient)
                     
            '''
            
            # YOUR CODE GOES HERE
#             test_feats =[]
            pool = Pool(processes = self.n_jobs) 
            test_feats = pool.map(self.get_features_for_one,map(lambda i: X[i:i+1],range(X.shape[0])))
            
#             assert False, 'You need to implement it for n_jobs > 1'
            
            
            
        return np.vstack(test_feats)
        
        
    def get_features_for_one(self, x):
        '''
            Computes KNN features for a single object `x`
        '''

        NN_output = self.NN.kneighbors(x) # distance, indices
        
        # Vector of size `n_neighbors`
        # Stores indices of the neighbors
        neighs = NN_output[1][0] # indices
        
        # Vector of size `n_neighbors`
        # Stores distances to corresponding neighbors
        neighs_dist = NN_output[0][0] # distance

        # Vector of size `n_neighbors`
        # Stores labels of corresponding neighbors
        neighs_y = self.y_train[neighs]  # 
        
        ## ========================================== ##
        ##              YOUR CODE BELOW
        ## ========================================== ##
        
        # We will accumulate the computed features here
        # Eventually it will be a list of lists or np.arrays
        # and we will use np.hstack to concatenate those
        return_list = [] 
        
        
        ''' 
            1. Fraction of objects of every class.
               It is basically a KNNClassifiers predictions.

               Take a look at `np.bincount` function, it can be very helpful
               Note that the values should sum up to one
        '''
        for k in self.k_list:
            # YOUR CODE GOES HERE
#             k_nn_idxs = neighs[0:k] # indices array of k nearest neighbors
#             topk_labels = self.y_train[k_nn_idxs]            
            topk_labels = neighs_y[:k]
            feats = np.bincount(topk_labels,minlength=self.n_classes) 
            feats = feats / (feats.sum()) 
            assert len(feats) == self.n_classes
            return_list += [feats]
        
        
        '''
            2. Same label streak: the largest number N, 
               such that N nearest neighbors have the same label.
               
               What can help you: `np.where`
               ## see comment :https://www.coursera.org/learn/competitive-data-science/discussions/weeks/4/threads/uNzpwAJDEeiZ7AqwKZ_sKA
        '''
        
        not_same_label_idx = np.where(np.diff(neighs_y) != 0 )[0] # not the same label idx
        if np.any(not_same_label_idx):
            feats = [not_same_label_idx[0] + 1] #largest number have the same label
        else:
            feats = [np.size(neighs_y)]
#         feats = []# YOUR CODE GOES HERE
        
        assert len(feats) == 1
        return_list += [feats]
        
        '''
            3. Minimum distance to objects of each class
               Find the first instance of a class and take its distance as features.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.

               `np.where` might be helpful
        '''
        feats = []
        obj_label = neighs_y[0]   
        for c in range(self.n_classes):            
            # YOUR CODE GOES HERE            
#             if c == obj_label and (np.sum(neighs_y==obj_label) > 1): # not only object itself
#                 nn_idx = np.where(neighs_y==obj_label)[0][1] # nearest index in the same class
#                 feats.append(neighs_dist[nn_idx]) # nearest neigh dist in the same class
                
#             elif c!= obj_label and (np.sum(neighs_y == c) > 0):
#                 nn_idx = np.where(neighs_y == c)[0][0]
#                 feats.append(neighs_dist[nn_idx])
            if (np.sum(neighs_y == c) > 0):        
                nn_idx = np.where(neighs_y == c)[0][0]
                feats.append(neighs_dist[nn_idx])
            else:
                feats.append(999)
                
        
        assert len(feats) == self.n_classes
        return_list += [feats]
        
        '''
            4. Minimum *normalized* distance to objects of each class
               As 3. but we normalize (divide) the distances
               by the distance to the closest neighbor.
               
               If there are no neighboring objects of some classes, 
               Then set distance to that class to be 999.
               
               Do not forget to add self.eps to denominator.
        '''
        feats = []
        obj_label = neighs_y[0]
        closet_dist = neighs_dist[0]
        for c in range(self.n_classes):
            # YOUR CODE GOES HERE
#             if c == obj_label and (len(np.where(neighs_y == obj_label)[0]) > 1):
#                 nn_idx = np.where(neighs_y == obj_label)[0][1]
#                 feats.append(normalized_neighs_dist[nn_idx])
                
#             elif c!= obj_label and len(np.where(neighs_y == c)[0]) != 0:
#                 nn_idx = np.where(neighs_y == c)[0][0]
#                 feats.append(normalized_neighs_dist[nn_idx])
            if (np.sum(neighs_y == c) > 0):
                nn_idx = np.where(neighs_y == c)[0][0]
                normalized_dist = neighs_dist[nn_idx] / (self.eps + closet_dist)
                feats.append(normalized_dist)
            else:
                feats.append(999)
        
        assert len(feats) == self.n_classes
        return_list += [feats]
        
        '''
            5. 
               5.1 Distance to Kth neighbor
                   Think of this as of quantiles of a distribution
               5.2 Distance to Kth neighbor normalized by 
                   distance to the first neighbor
               
               feat_51, feat_52 are answers to 5.1. and 5.2.
               should be scalars
               
               Do not forget to add self.eps to denominator.
        '''
        for k in self.k_list:
            
            feat_51 = neighs_dist[k-1]# YOUR CODE GOES HERE
            feat_52 = neighs_dist[k-1]/ (self.eps + neighs_dist[0])# YOUR CODE GOES HERE
            
            return_list += [[feat_51, feat_52]]
        
        '''
            6. Mean distance to neighbors of each class for each K from `k_list` 
                   For each class select the neighbors of that class among K nearest neighbors 
                   and compute the average distance to those objects
                   
                   If there are no objects of a certain class among K neighbors, set mean distance to 999
                   
               You can use `np.bincount` with appropriate weights
               Don't forget, that if you divide by something, 
               You need to add `self.eps` to denominator.
        '''
        
        for k in self.k_list:
            feats = []
            numerator = np.zeros(self.n_classes)
            denominator = np.full(self.n_classes, self.eps)
            t = neighs_y[:k].max() + 1
            numerator[:t] = np.bincount(neighs_y[:k], weights=neighs_dist[:k])
            denominator[:t] = self.eps + np.bincount(neighs_y[:k])
            feats = np.where(numerator>0, numerator/denominator, 999)
            
#             neighs_yk = neighs_y[:k] # k nearest neighs y label
#             neighs_k = neighs[:k] # k nn indx
#             neighs_dist_k = neighs_dist[:k]
            # YOUR CODE GOES IN HERE 
#             ### WRONG !!! WHY???   
#             for c in range(self.n_classes):
#                 nn_idx_in_c = np.where(neighs_yk == c)[0] ## neighbors idxs in class c among k
#                 if np.any(nn_idx_in_c): # not None
#                     mean_dist_in_c = np.sum(neighs_dist_k[nn_idx_in_c]) / (len(nn_idx_in_c) + self.eps)
#                 else:
#                     mean_dist_in_c = 999
#                 feats.append(mean_dist_in_c)
                
            assert len(feats) == self.n_classes
            return_list += [feats]
        
        
        # merge
        knn_feats = np.hstack(return_list)
        
        assert knn_feats.shape == (239,) or knn_feats.shape == (239, 1)
        return knn_feats

if __name__ == '__main__':
    train_path = './KNN_features_data/X.npz'
    train_labels = './KNN_features_data/Y.npy'

    test_path = './KNN_features_data/X_test.npz'
    test_labels = './KNN_features_data/Y_test.npy'

    # Train data
    X = scipy.sparse.load_npz(train_path)
    Y = np.load(train_labels)

    # Test data
    X_test = scipy.sparse.load_npz(test_path)
    Y_test = np.load(test_labels)

    # Out-of-fold features we loaded above were generated with n_splits=4 and skf seed 123
    # So it is better to use seed 123 for generating KNN features as well 
    skf_seed = 123
    n_splits = 4


    # a list of K in KNN, starts with one 
    k_list = [3, 8, 32]

    # Load correct features
    true_knn_feats_first50 = np.load('./KNN_features_data/knn_feats_test_first50.npy')

    # Create instance of our KNN feature extractor
    NNF = NearestNeighborsFeats(n_jobs=4, k_list=k_list, metric='minkowski')

    # Fit on train set
    NNF.fit(X, Y)

    # Get features for test
    test_knn_feats = NNF.predict(X_test[:50])

    # This should be zero
    print ('Deviation from ground thruth features: %f' % np.abs(test_knn_feats - true_knn_feats_first50).sum())
    deviation =np.abs(test_knn_feats - true_knn_feats_first50).sum(0)
    # deviation =np.abs(test_knn_feats - true_knn_feats_first50[44:45]).sum(0)
    for m in np.where(deviation > 1e-3)[0]: 
        p = np.where(np.array([87, 88, 117, 146, 152, 239]) > m)[0][0]
        print ('There is a problem in feature %d, which is a part of section %d.' % (m, p + 1))