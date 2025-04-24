import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from tqdm import tqdm
import ast
import sys
import re

class TrialDataset(Dataset):

    def __init__(self, fname, pca = False):
        np.set_printoptions(threshold=sys.maxsize)
        self.dat = pd.read_csv(fname)
        self.transformed_dat = None
        self.results = []
        self.data_dim = 0
        self.avg_res = 0
        self.std_res = 0
        self.num_samples = 0
        self.thresh = 0
        self.relevance_matrix = None
        self.feat_names = []
        self.pca = pca
        # print(np.median(self.dat['Enrollment']))
        # self.plot_age_hist('Minimum Age')
        # self.plot_age_hist('Maximum Age')
        # self.plot_hist('Enrollment')
        # self.plot_hist('Number Locations')
        # self.plot_bar('Organization')
        # self.plot_bar('Primary Purpose')
        # self.plot_bar('Blinding')
        # self.plot_bar('Sex')
        self.load_dataset()

    def __len__(self):
        return np.shape(self.transformed_dat)[0]
    
    def __getitem__(self, idx):
        return self.transformed_dat[idx], self.results[idx]

    def load_dataset(self):
        self.transformed_dat = np.array(pd.get_dummies(self.dat['Conditions'].explode()).groupby(level=0).sum())
        self.feat_names += list(pd.get_dummies(self.dat['Conditions'].explode()).groupby(level=0).sum().columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Collaborators'].explode()).groupby(level=0).sum()), axis=1)
        vals = np.array(pd.get_dummies(self.dat['Collaborators'].explode()).groupby(level=0).sum().columns)
        vals[vals=='[]'] = 'NO COLLABORATORS'
        self.feat_names += list(vals)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Organization'], dtype=int)), axis=1)
        self.feat_names += list(pd.get_dummies(self.dat['Organization'], dtype=int).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Allocation'], dtype=int)), axis=1)
        self.feat_names += list(pd.get_dummies(self.dat['Allocation']).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Intervention Model'], dtype=int)), axis=1)
        self.feat_names += list(pd.get_dummies(self.dat['Intervention Model'], dtype=int).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Primary Purpose'], dtype=int)), axis=1)
        self.feat_names+= list(pd.get_dummies(self.dat['Primary Purpose'], dtype=int).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Blinding'], dtype=int)), axis=1)
        self.feat_names += list(pd.get_dummies(self.dat['Blinding'], dtype=int).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array([self.dat['Enrollment']], dtype=float).T, axis=1)
        self.transformed_dat[:,-1] -= np.mean(self.transformed_dat[:,-1])
        self.transformed_dat[:,-1] /= np.std(self.transformed_dat[:,-1])
        self.feat_names.append('Enrollment')
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Healthy Volunteers'], dtype=int)), axis=1)
        self.feat_names += list(pd.get_dummies(self.dat['Healthy Volunteers'], dtype=int).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array(pd.get_dummies(self.dat['Sex'], dtype=int)), axis=1)
        self.feat_names += list(pd.get_dummies(self.dat['Sex'], dtype=int).columns)
        self.transformed_dat = np.append(self.transformed_dat, np.array([self.dat['Minimum Age'].apply(lambda s: s.split(' ')[0])], dtype=float).T, axis=1)
        self.feat_names.append('Minimum Age')
        self.transformed_dat[:,-1] -= np.mean(self.transformed_dat[:,-1])
        self.transformed_dat[:,-1] /= np.std(self.transformed_dat[:,-1])
        self.transformed_dat = np.append(self.transformed_dat, np.array([self.dat['Maximum Age'].apply(lambda s: s.split(' ')[0])], dtype=float).T, axis=1)
        self.feat_names.append('Maximum Age')
        self.transformed_dat[:,-1] -= np.mean(self.transformed_dat[:,-1])
        self.transformed_dat[:,-1] /= np.std(self.transformed_dat[:,-1])
        self.transformed_dat = np.append(self.transformed_dat, np.array([self.dat['Number Locations']], dtype=float).T, axis=1)
        self.transformed_dat[:,-1] -= np.mean(self.transformed_dat[:,-1])
        self.transformed_dat[:,-1] /= np.std(self.transformed_dat[:,-1])
        self.feat_names.append('Number of Locations')
        deriv_terms = self.dat['Derived terms'].apply(lambda x: [t[0] for t in ast.literal_eval(x)] if type(x) is str else [])
        deriv_terms = pd.get_dummies(deriv_terms.explode(), dtype=float).groupby(level=0).sum()
        for i in range(len(self.dat['Derived terms'])):
            dts = self.dat['Derived terms'].iloc[i]
            if type(dts) is float: continue
            for dt in ast.literal_eval(dts):
                if dt[1] == 'LOW':
                    deriv_terms.at[i, dt[0]] = 0.25
        self.relevance_matrix = np.array(deriv_terms)
        for i in range(np.shape(self.relevance_matrix)[1]):
            self.feat_names.append('deriv terms{}'.format(i))
        self.transformed_dat = np.append(self.transformed_dat, np.array(deriv_terms), axis=1)
        
        treatments = self.dat['Interventions'].apply(lambda i: [t[1] for t in ast.literal_eval(i)])
        self.transformed_dat = np.append(self.transformed_dat, pd.get_dummies(treatments.explode(), dtype=float).groupby(level=0).sum(), axis=1)
        self.feat_names += list(pd.get_dummies(treatments.explode(), dtype=float).columns)

        ph2_res = []
        ph2_aff = []
        for i in tqdm(range(len(self.dat))):
            if pd.isnull(self.dat['Derived terms'].iloc[i]):
                ph2_res.append(0.05)
                ph2_aff.append(0)
                continue
            res, aff = self.nn(ast.literal_eval(self.dat['Derived terms'].iloc[i]), ast.literal_eval(self.dat['PH2 Results'].iloc[i]))
            d = res[1][0]
            if type(d) is str:
                pv = re.findall(r"\d+\.\d+", d)
                if len(pv) == 0:
                    ph2_res.append(0.05)
                    ph2_aff.append(aff)
                    continue
                d = float(pv[0])
            ph2_res.append(d)
            ph2_aff.append(aff)

        ph2_res = (ph2_res - np.mean(ph2_res)) / np.std(ph2_res)
        ph2_aff = (ph2_aff - np.mean(ph2_aff)) / np.std(ph2_aff)
        self.feat_names.append('Phase II Result')
        self.feat_names.append('Phase II Affinity')
        self.transformed_dat = np.append(self.transformed_dat, np.array([ph2_res]).T, axis=1)
        self.transformed_dat = np.append(self.transformed_dat, np.array([ph2_aff]).T, axis=1)
        # print(len(feat_names), np.shape(self.transformed_dat))
        # print(np.shape(self.transformed_dat))
        # pca = PCA(n_components=256)
        # self.transformed_dat = pca.fit_transform(self.transformed_dat)
        # print(len(feat_names), np.shape(self.transformed_dat))
        # print(pca.explained_variance_ratio_)
        
        
            
        
        self.data_dim = np.shape(self.transformed_dat)[1]

        for d in self.dat['Results']:
            if type(d) is str:
                pv = re.findall(r"\d+\.\d+", d)
                if len(pv) == 0:
                    pv = re.findall(r"\d+", d)
                    # if float(pv[0]) > 1:
                    #     self.results.append(1.0)
                    # else:
                    self.results.append(float(pv[0]))
                    continue
                d = float(pv[0])
            self.results.append(d)
            #0 if d>0.05 else 1
        print(max(self.results))
        plt.hist(self.results)
        # plt.xlim((0,1))
        plt.xlabel('Initial p-values')
        plt.ylabel('Count')
        plt.show()

        # pca = PCA(n_components= 3)
        # dat_new = pca.fit_transform(self.transformed_dat, [])
        # dat_neg = dat_new[np.array(self.results)==0]
        # dat_pos = dat_new[np.array(self.results)==1]
        self.results = np.array([self.results]).T
        self.results += 0.00001
        self.thresh = np.log(0.05)
        self.results = np.log(self.results)
        self.avg_res = np.mean(self.results)
        self.std_res = np.std(self.results)
        self.thresh = (self.thresh - self.avg_res)/self.std_res
        self.results = (self.results - self.avg_res)/self.std_res

        plt.hist(self.results)
        plt.xlabel('Transformed p-values')
        plt.ylabel('Count')
        plt.show()
        
        
        # plt.plot(dat_neg[:,0], dat_neg[:,2], 'r.', alpha=0.5)
        # plt.plot(dat_pos[:,0], dat_pos[:,2], 'b.', alpha=0.5)
        # plt.savefig('PCA3.png')
        # plt.show()
        self.num_samples = np.shape(self.results)[0]

    def load(self, train, valid, test, binary = False):
        denom = sum([train, valid, test])
        idxs = np.arange(self.num_samples, dtype=int)
        np.random.shuffle(idxs)
        train_idxs = idxs[:int(self.num_samples*(train/denom))]
        valid_idxs = idxs[int(self.num_samples*(train/denom)):int(self.num_samples*((train+valid)/denom))]
        test_idxs = idxs[int(self.num_samples*((train+valid)/denom)):]
        train_set = self.transformed_dat[train_idxs, :]
        train_res = self.results[train_idxs]
        valid_set = self.transformed_dat[valid_idxs, :]
        valid_res = self.results[valid_idxs]
        test_set = self.transformed_dat[test_idxs, :]
        test_res = self.results[test_idxs]

        neg_idxs = np.arange(len(train_res))[(train_res > self.thresh).flatten()]
        n_samples = len(train_res <= self.thresh) - len(train_res > self.thresh)
        new_negs = np.random.choice(neg_idxs, n_samples, replace = True)
        train_res = np.append(train_res, train_res[new_negs], axis=0)
        train_set = np.append(train_set, train_set[new_negs], axis=0)



        if binary:
            train_res = (train_res <= self.thresh).astype(int)
            valid_res = (valid_res <= self.thresh).astype(int)
            test_res = (test_res <= self.thresh).astype(int)

        return train_set, train_res, valid_set, valid_res, test_set, test_res

    def plot_age_hist(self, var):
        plt.hist(np.array(self.dat[var].apply(lambda s: s.split(' ')[0]), dtype=float))
        plt.xlabel(var)
        # plt.xscale('log')
        plt.ylabel('Count')
        plt.show()
    
    def plot_hist(self, var):
        plt.hist(np.array(self.dat[var], dtype=float))
        plt.xlabel(var)
        # plt.xscale('log')
        plt.ylabel('Count')
        plt.show()

    def plot_bar(self, var):
        cats = np.array(self.dat[var].value_counts().index)
        for i in range(len(cats)):
            if len(cats[i]) > 15:
                cats[i] = cats[i][:15] 
        cts = self.dat[var].value_counts().values
        f = plt.figure(figsize=(20, 6))
        plt.bar(cats, cts)
        plt.ylabel('Counts')
        plt.title(var)
        plt.show()


    def nn(self, query, values, k=1):
        if len(values) == 0:
            return ([], [0.05]), 0
        affinities = np.zeros(len(values))
        for q in query: # Loop over each term in queries (form of [(term, relevance)])
            for i in range(len(values)):    # Loop over each possible neighbor (form of [([(term, relevance)], [pvalue])])
                v = values[i]
                for term in v[0]:
                    if q[0] == term[0]:
                        if q[1] == term[1]:
                            affinities[i] += 1
                        else:
                            affinities[i] += 0.5
        affinities /= len(query)
        return values[np.argsort(affinities)[-1]], np.sort(affinities)[-1]
        

