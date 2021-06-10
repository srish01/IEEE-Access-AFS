#!/usr/bin python  


#             3. sf_adv_jmi, sf_adv_mim, sf_adv_mrmr, sf_adv_mifs, sf_adv_cmim, sf_adv_disr, sf_adv_icap = run_feature_selection(Xtrk_poisoned, ytrk_poisoned, n_selected_features)
#             4. JMI_adv.append(sf_adv_jmi)
#                MIM_adv.append(sf_adv_mim) ...
#             5. dist_jaccard += jaccard(sf_adv_jmi, af_base_jmi)
#             6. dist_kunch += kuncheva(sf_adv_jmi, af_base_jmi, nf)

#         2. JMI_consis_jacc_clean[0], JMI_consis_kunch_clean[0] = total_consistency(JMI_clean) 
#         3. MIM_consis_jacc_clean[1], JMI_consis_kunch_clean[1] = total_consistency(MIM_clean) 

#         4. JMI_consis_jacc_adv[n, 0], JMI_consis_kunch_adv[n, 0] = total_consistency(JMI_adv)

#         5. dist_jaccard[n, 0], dist_kunch[n, 0] = dist_jaccard/CV, dist_kunch/CV 



# Copyright 2021 Gregory Ditzler 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this 
# software and associated documentation files (the "Software"), to deal in the Software 
# without restriction, including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
# permit persons to whom the Software is furnished to do so, subject to the following 
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies 
# or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE 
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT 
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

import numpy as np 
import pandas as pd 

import argparse

from utils import kuncheva, jaccard, total_consistency
import sys
sys.path.append("./scikit-feature/")
import skfeature as skf
from skfeature.function.information_theoretical_based import JMI, MIM, MRMR, MIFS, CMIM, DISR, ICAP
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from skfeature.function.similarity_based.fisher_score import fisher_score, feature_ranking
from skfeature.function.similarity_based.reliefF import reliefF


# -----------------------------------------------------------------------
# # setup program constants
# Use thread pool executor for feature selection
PARALLEL = True
# percentage of poisoning levels  
POI_RNG = [.01, .025, .05, .075, .1, .125, .15, .175, .2]
# total number of poisoning levels 
NPR = len(POI_RNG)
# percentage of features that we want to select 
SEL_PERCENT = .2
# number of algorithms that we are going to test [JMI, MIM, MRMR, MIFS, CMIM, DISR, ICAP] or [MIM, MIFS, MRMR, DISR, Relief, Fisher]
NALG = 6
# used when we select features 
FEAT_IDX = 0
# number of cross validation runs to perform
CV = 5
# seed for reproducibility
SEED = 1
# dataset names 
# did not run 
#   - miniboone, connect-4, ozone, spambase
DATA = [
        #'hepatitis',
        #'ringnorm',
        'twonorm',
        #'mushroom',
        #'parkinsons',
        #'statlog-german-credit',
        #'oocytes_trisopterus_nucleus_2f',
        #'trains',
        #'breast-cancer-wisc-diag',
        'breast-cancer-wisc-prog',
        'conn-bench-sonar-mines-rocks',
        'ionosphere',
        #'cylinder-bands',
        #'chess-krvkp',
        #'oocytes_merluccius_nucleus_4d',
        'molec-biol-promoter',
        #'spambase',
        #'ozone',
        #'musk-1',
        # 'musk-2'
        
]
BOX = ['0.5', '1', '2', '5', '10']
# -----------------------------------------------------------------------

def similarity_based_FS(algo, X, Y, n_selected_features):
    top_n_feat = []
    top_n_feat.append(feature_ranking(algo(X, Y))[:n_selected_features])
    #print("Algo top n: ", top_n_feat)
    return top_n_feat

def run_feature_selection(X, Y, n_selected_features):
    
    lst = []
    
    if PARALLEL:
        # with multiprocessing.Pool(processes=4) as pool:
        #     lst.append(pool.apply(JMI.jmi, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
        #     lst.append(pool.apply(MIM.mim, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
        #     lst.append(pool.apply(MRMR.mrmr, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
        #     lst.append(pool.apply(MIFS.mifs, args=(X, Y), kwds={'n_selected_features': n_selected_features}))
            
        # lst = [l[FEAT_IDX] for l in lst]
        
        with ProcessPoolExecutor(max_workers=6) as executor:
            #lst.append(executor.submit(JMI.jmi, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(MIM.mim, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(MRMR.mrmr, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(MIFS.mifs, X, Y, n_selected_features=n_selected_features))  
            #lst.append(executor.submit(CMIM.cmim, X, Y, n_selected_features=n_selected_features))
            lst.append(executor.submit(DISR.disr, X, Y, n_selected_features=n_selected_features))
            #lst.append(executor.submit(ICAP.icap, X, Y, n_selected_features=n_selected_features)) 
            lst.append(executor.submit(similarity_based_FS, reliefF, X, Y, n_selected_features))
            lst.append(executor.submit(similarity_based_FS, fisher_score, X, Y, n_selected_features))
        lst = [l.result()[FEAT_IDX] for l in lst]
    else:
       # lst.append(JMI.jmi(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(MIM.mim(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(MRMR.mrmr(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(MIFS.mifs(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        #lst.append(CMIM.cmim(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(DISR.disr(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        #lst.append(ICAP.icap(X, Y, n_selected_features=n_selected_features)[FEAT_IDX])
        lst.append(feature_ranking(reliefF(X, Y))[:n_selected_features][FEAT_IDX])
        lst.append(feature_ranking(fisher_score(X, Y)) [:n_selected_features][FEAT_IDX])


    return lst


def experiment(data, box, cv, output):
    """
    Write the results of an experiment.
        This function will run an experiment for a specific dataset for a bounding box. 
        There will be CV runs of randomized experiments run and the outputs will be 
        written to a file. 

        Parameters
        ----------
        data : string
            Dataset name.
            
        box : string 
            Bounding box on the file name.
        cv : int 
            Number of cross validation runs. 
            
        output : string
            If float or tuple, the projection will be the same for all features,
            otherwise if a list, the projection will be described feature by feature.
                    
        Returns
        -------
        None
            
        Raises
        ------
        ValueError
            If the percent poison exceeds the number of samples in the requested data.
    """
    #data, box, cv, output = 'conn-bench-sonar-mines-rocks', '1', 5, 'results/test.npz'

    # load normal and adversarial data 
    path_adversarial_data = 'Extra_Exp_data/attacks/' + data + '_[xiao][' + box + '].csv'
    df_normal = pd.read_csv('Extra_Exp_data/clean/' + data + '.csv', header=None).values
    df_adversarial = pd.read_csv(path_adversarial_data, header=None).values
    
    # separate out the normal and adversarial data 
    Xn, yn = df_normal[:,:-1], df_normal[:,-1]
    Xa, ya = df_adversarial[:,:-1], df_adversarial[:,-1]

    # change the labels from +/-1 to [0,1]
    ya[ya==-1], yn[yn==-1] = 0, 0

    # calculate the ratios of data that would be used for training and hold out  
    p0, p1 = 1./cv, (1. - 1./cv)
    N, nf = Xn.shape
    # calculate the total number of training and testing samples and set the numbfer of 
    # features that are going to be selected 
    Ntr, Nte = int(p1*N), int(p0*N)
    n_selected_features = int(Xn.shape[1]*SEL_PERCENT)+1

    # zero the results out : err_jaccard and err_kuncheva are 9x4 matrices
    dist_jaccard, dist_kuncheva = np.zeros((NPR, NALG)), np.zeros((NPR, NALG))
    consis_jaccard_adv, consis_kuncheva_adv = np.zeros((NPR, NALG)), np.zeros((NPR, NALG))
    consis_jaccard_clean, consis_kuncheva_clean = np.zeros((1, NALG)), np.zeros((1, NALG))

        
    # set the random seed for reproducibility 
    np.random.seed(SEED)

    for n in range(NPR):
        print("Poisoning Ratio: ", POI_RNG[n])
        # calucate the number of poisoned data that we are going to need to make sure 
        # that the poisoning ratio is correct in the training data. e.g., if you have 
        # N=100 samples and you want to poison by 20% then the 20% needs to be from 
        # the training size. hence it is not 20. 
        Np = int(Ntr*POI_RNG[n]+1)
        if Np >= len(ya): 
            # shouldn't happen but catch the case where we are requesting more poison
            # data samples than are available. NEED TO BE CAREFUL WHEN WE ARE CREATING 
            # THE ADVERSARIAL DATA
            raise ValueError('Number of poison data requested is larger than the available data.')

        # find the number of normal samples (i.e., not poisoned) samples in the 
        # training data. then create the randomized data set that has Nn normal data
        # samples and Np adversarial samples in the training data
        Nn = Ntr - Np

        MIM_fset_adv = []
        MIFS_fset_adv = []
        MRMR_fset_adv = []
        #JMI_fset_adv = []
        #CMIM_fset_adv = []
        DISR_fset_adv = []
        #ICAP_fset_adv = []
        Relief_fset_adv = []
        Fisher_fset_adv = []

        MIM_fset_clean = []
        MIFS_fset_clean = []
        MRMR_fset_clean = []
        #JMI_fset_clean = []
        #CMIM_fset_clean = []
        DISR_fset_clean = []
        #ICAP_fset_clean = []
        Relief_fset_clean = []
        Fisher_fset_clean = []
        
        for c in range(cv):
            print("CV: ", c)
            i = np.random.permutation(N)
            Xtrk, ytrk, Xtek, ytek = Xn[i][:Ntr], yn[i][:Ntr], Xn[i][-Nte:], yn[i][-Nte:]

            idx_normal, idx_adversarial = np.random.permutation(len(ytrk))[:Nn], \
                                        np.random.permutation(len(ya))[:Np]
            Xtrk_poisoned, ytrk_poisoned = np.concatenate((Xtrk[idx_normal], Xa[idx_adversarial])), \
                                        np.concatenate((ytrk[idx_normal], ya[idx_adversarial]))
         


            if n == 0:
                # run feature selection with the original training data containing benign samples
                sf_base_mim, sf_base_mrmr, sf_base_mifs, sf_base_disr, sf_base_relief, sf_base_fisher = run_feature_selection(Xtrk, ytrk, n_selected_features) 
                
                MIM_fset_clean.append(sf_base_mim)
                MIFS_fset_clean.append(sf_base_mifs)
                MRMR_fset_clean.append(sf_base_mrmr)
                #JMI_fset_clean.append(sf_base_jmi)
                #CMIM_fset_clean.append(sf_base_cmim)
                DISR_fset_clean.append(sf_base_disr)
                #ICAP_fset_clean.append(sf_base_icap)
                Relief_fset_clean.append(sf_base_relief)
                Fisher_fset_clean.append(sf_base_fisher)
                #print("Relief clean Feature set", Relief_fset_clean)
                #print("MIM clean feature set", MIM_fset_clean)

            # run feature selection with the training data that has adversarial samples
            sf_adv_mim, sf_adv_mrmr, sf_adv_mifs, sf_adv_disr, sf_adv_relief, sf_adv_fisher = run_feature_selection(Xtrk_poisoned, ytrk_poisoned, n_selected_features)
            
            MIM_fset_adv.append(sf_adv_mim)
            MIFS_fset_adv.append(sf_adv_mifs)
            MRMR_fset_adv.append(sf_adv_mrmr)
            #JMI_fset_adv.append(sf_adv_jmi)
            #CMIM_fset_adv.append(sf_adv_cmim)
            DISR_fset_adv.append(sf_adv_disr)
            #ICAP_fset_adv.append(sf_adv_icap)
            Relief_fset_adv.append(sf_adv_relief)
            Fisher_fset_adv.append(sf_adv_fisher)
            #print("Relief adversarial features: ", Relief_fset_adv)
            #print("MIM adversarial features: ", MIM_fset_adv)

            dist_jaccard[n, 0] += jaccard(sf_adv_mim, sf_base_mim)
            dist_jaccard[n, 1] += jaccard(sf_adv_mifs, sf_base_mifs)
            dist_jaccard[n, 2] += jaccard(sf_adv_mrmr, sf_base_mrmr)
            #dist_jaccard[n, 3] += jaccard(sf_adv_jmi, sf_base_jmi)
            #dist_jaccard[n, 4] += jaccard(sf_adv_cmim, sf_base_cmim)
            dist_jaccard[n, 3] += jaccard(sf_adv_disr, sf_base_disr)
            #dist_jaccard[n, 6] += jaccard(sf_adv_icap, sf_base_icap)
            dist_jaccard[n, 4] += jaccard(sf_adv_relief, sf_base_relief)
            dist_jaccard[n, 5] += jaccard(sf_adv_fisher, sf_base_fisher)

            dist_kuncheva[n, 0] += kuncheva(sf_adv_mim, sf_base_mim, nf)
            dist_kuncheva[n, 1] += kuncheva(sf_adv_mifs, sf_base_mifs, nf)
            dist_kuncheva[n, 2] += kuncheva(sf_adv_mrmr, sf_base_mrmr, nf)
            #dist_kuncheva[n, 3] += kuncheva(sf_adv_jmi, sf_base_jmi, nf)
            #dist_kuncheva[n, 4] += kuncheva(sf_adv_cmim, sf_base_cmim, nf)
            dist_kuncheva[n, 3] += kuncheva(sf_adv_disr, sf_base_disr, nf)
            #dist_kuncheva[n, 6] += kuncheva(sf_adv_icap, sf_base_icap, nf)
            dist_kuncheva[n, 4] += kuncheva(sf_adv_relief, sf_base_relief, nf)
            dist_kuncheva[n, 5] += kuncheva(sf_adv_fisher, sf_base_fisher, nf)
        

        if n == 0:

            consis_jaccard_clean[0, 0], consis_kuncheva_clean[0, 0] = total_consistency(MIM_fset_clean, nf)
            consis_jaccard_clean[0, 1], consis_kuncheva_clean[0, 1] = total_consistency(MIFS_fset_clean, nf)        
            consis_jaccard_clean[0, 2], consis_kuncheva_clean[0, 2] = total_consistency(MRMR_fset_clean, nf)
            #consis_jaccard_clean[0, 3], consis_kuncheva_clean[0, 3] = total_consistency(JMI_fset_clean, nf)
            #consis_jaccard_clean[0, 4], consis_kuncheva_clean[0, 4] = total_consistency(CMIM_fset_clean, nf)
            consis_jaccard_clean[0, 3], consis_kuncheva_clean[0, 3] = total_consistency(DISR_fset_clean, nf)
            #consis_jaccard_clean[0, 6], consis_kuncheva_clean[0, 6] = total_consistency(ICAP_fset_clean, nf)
            consis_jaccard_clean[0, 4], consis_kuncheva_clean[0, 4] = total_consistency(Relief_fset_clean, nf)    
            consis_jaccard_clean[0, 5], consis_kuncheva_clean[0, 5] = total_consistency(Fisher_fset_clean, nf)

        consis_jaccard_adv[n, 0], consis_kuncheva_adv[n, 0] = total_consistency(MIM_fset_adv, nf)
        consis_jaccard_adv[n, 1], consis_kuncheva_adv[n, 1] = total_consistency(MIFS_fset_adv, nf)        
        consis_jaccard_adv[n, 2], consis_kuncheva_adv[n, 2] = total_consistency(MRMR_fset_adv, nf)
        #consis_jaccard_adv[n, 3], consis_kuncheva_adv[n, 3] = total_consistency(JMI_fset_adv, nf)
        #consis_jaccard_adv[n, 4], consis_kuncheva_adv[n, 4] = total_consistency(CMIM_fset_adv, nf)
        consis_jaccard_adv[n, 3], consis_kuncheva_adv[n, 3] = total_consistency(DISR_fset_adv, nf)
        #consis_jaccard_adv[n, 6], consis_kuncheva_adv[n, 6] = total_consistency(ICAP_fset_adv, nf)
        consis_jaccard_adv[n, 4], consis_kuncheva_adv[n, 4] = total_consistency(Relief_fset_adv, nf)
        consis_jaccard_adv[n, 5], consis_kuncheva_adv[n, 5] = total_consistency(Fisher_fset_adv, nf)

    
    dist_jaccard /= CV
    dist_kuncheva /=CV 

    

    np.savez(output, 
            dist_jaccard = dist_jaccard, 
            dist_kuncheva = dist_kuncheva, 
            consis_jaccard_clean = consis_jaccard_clean, 
            consis_kuncheva_clean = consis_kuncheva_clean, 
            consis_jaccard_adv = consis_jaccard_adv, 
            consis_kuncheva_adv = consis_kuncheva_adv, 
            Xtrk_pois = Xtrk_poisoned, 
            ytrk_pois = ytrk_poisoned,
            Xa = Xa,
            ya = ya)
    return None

if __name__ == '__main__': 

    for data in DATA: 
        for box in BOX: 
            print('Running ' + data + ' - box:' + box)
            #try: 
            experiment(data, box, CV, 'IEEE/Final_algs/results/' + data + '_[xiao][' + box + ']_results.npz')
            #except:
            #    print(' ... ERROR ...')


