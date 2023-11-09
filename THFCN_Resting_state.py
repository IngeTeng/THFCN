#Resting-state categorical fMRI data
import numpy as np
from sklearn import linear_model
import copy
from scipy.stats import pearsonr
import os
import networkx as nx
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from mutitask_Lasso_manual import mutitaskLasso
from construct_hyper_graph_KNN import construct_H_with_KNN
from scipy.stats import ks_2samp
from construct_hyper_graph_KNN import construct_H_with_KNN, generate_S_from_H
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score, recall_score, make_scorer,accuracy_score,roc_curve,auc,precision_score,f1_score
from sklearn.model_selection import cross_val_score,cross_val_predict
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import roc_auc_score, recall_score, make_scorer
import time
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")

# import networkx.algorithms.efficiency_measures
# normalize data set into [0, 1] or [-1, 1]
def normalize(ori_data, flag='01'):
    ori_data =  ori_data.T
    data = ori_data.copy()
    if len(data.shape) > 1:   # 2-D
        N , K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data.T, maxV, minV
    else:   # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
        return data, maxV, minV
# Calculate node efficiency
def node_eff(G):
    node_ef = []
    for u in G.nodes():
        sum = 0
        for v in G.nodes():
            if u != v:
                sum += nx.efficiency(G, u, v)
        node_ef.append(sum / (ROI_num - 1))
    node_ef = np.array(node_ef)
    return node_ef

def flatten(nested):
    try:
        for sublist in nested:
            for element in flatten(sublist):
                yield element
    except TypeError:
        yield nested


# Calculate TP TN FP FN
def get_TpTN_FpFn(list1, list2):
    reallabel_list = list(flatten(list1))
    predictlabel_list = list(flatten(list2))
    TP_count = 0
    TN_count = 0
    FP_count = 0
    FN_count = 0
    for i in range(len(reallabel_list)):
        if reallabel_list[i] == 1 and predictlabel_list[i] == 1:
            TP_count += 1
        if reallabel_list[i] == -1 and predictlabel_list[i] == -1:
            TN_count += 1
        if reallabel_list[i] == -1 and predictlabel_list[i] == 1:
            FP_count += 1
        if reallabel_list[i] == 1 and predictlabel_list[i] == -1:
            FN_count += 1
    return TP_count, TN_count, FP_count, FN_count

def reverseFunc(y, belta=1, flag='-01'):
    if flag == '-01':
        if y > 0.99999:
            y = 0.99999
        elif y < -0.99999:
            y = -0.99999
        return np.arctanh(y)
    else:
        if y > 0.999:
            y = 0.999

        elif y < 0.00001:
            y = 0.001
        # elif -0.00001 < y < 0:
        #     y = -0.00001

        x = 1 / belta * np.log(y / (1 - y))
        return x

#Calculate the degree of the hyperedge
def cal_De(H, j):
    row, col = H.shape
    col_sum = 0.0
    for i in range(row):
        col_sum += H[i, j]

    return col_sum


def create_dataset4(ts, belta):
    seq = ts.T
    Nc, L = seq.shape
    Y = np.zeros(shape=(L - 1, Nc))
    # Build Y
    for m in range(1, L):
        for n in range(Nc):
            Y[m - 1, n] = reverseFunc(seq[n, m], belta)

    # Build X
    X = np.zeros(shape=((L - 1), Nc))
    for t in range(L-1):
            for j in range(Nc):
                X[t,j] = seq[j,t]
    return X, Y


# Calculate S=HWDe^-1H^T
def cal_S(H,W):
    # Calculate De
    dim = W.shape[0]
    De = np.zeros(shape=(dim,dim))
    W_mat = np.zeros(shape=(dim, dim))
    for i in range(dim):
        De[i,i] = cal_De(H, i)
        W_mat[i,i] = W[i]
    S_tmp1 = np.dot(H, W_mat)
    S_tmp2 = np.dot(S_tmp1, np.linalg.inv(De))
    S = np.dot(S_tmp2,H.T)
    return S

if __name__ == '__main__':
    start = time.time()
    ##################################Load data####################
    
    ROI_num = 116 # AAL1
    KNN = 10    # Number of neighbors
    belta = 1   # sigmod param

    # hypergraph regularization and multitask params
    alpha = 0.01
    beta = 100
    tol = 1e-24

    # Load run1 data
    dataname = 'run1'
    sub_run1 = 16 # Number of run1 subjects
    path_run1 = 'fMRI/run1/'
    file_list = os.listdir(path_run1)
    pre = dataname
    ts_run1 = []
    ts_run1_L = []
    # Loop all ADNI subjects
    for file_index, file in enumerate(file_list):
        orginal_ts = np.loadtxt(path_run1+file)
        ts_len_run1 = orginal_ts.shape[0]
        new_ts_run1 = np.zeros(shape=(ts_len_run1, ROI_num))
        new_ts_run1 = orginal_ts[:,0:ROI_num]
        # Data normalization
        normalize_style = '-01' #（-1，1）
        dataset_run1, maxV, minV = normalize(new_ts_run1, normalize_style)
        ts_run1.append(dataset_run1)

        # Calculate the hypergraph Laplace L_h for each ADNI subject
        H = construct_H_with_KNN(dataset_run1.T,KNN)
        S, L = generate_S_from_H(H)
        L = np.array(L)
        ts_run1_L.append(L)


    ##Load rest data
    dataname = 'rest'
    sub_rest = 13  # Number of CN subjects
    ROI_num = 116  # AAL1
    path_run1 = 'fMRI/rest/'
    file_list = os.listdir(path_run1)
    pre = dataname

    ts_rest = []
    ts_rest_L = []
    # Loop all rest subjects
    for file_index, file in enumerate(file_list):
        orginal_ts = np.loadtxt(path_run1 + file)
        ts_len_rest = orginal_ts.shape[0]
        new_ts_rest = np.zeros(shape=(ts_len_rest, ROI_num))
        new_ts_rest = orginal_ts[:, 0:ROI_num]
        # Data normalization
        normalize_style = '-01'  # （-1，1）
        dataset_rest, maxV, minV = normalize(new_ts_rest, normalize_style)
        ts_rest.append(dataset_rest)

        # Calculate the hypergraph Laplace L_h for each CN subject
        H = construct_H_with_KNN(dataset_rest.T, KNN)
        S, L = generate_S_from_H(H)
        S = np.array(S)
        L = np.array(L)
        ts_rest_L.append(L)

    ######################Build X,Y######################
    # rest subjects
    x_rest = []
    y_rest = []
    for el in range(sub_rest):
        print('rest sub', el)
        ts = ts_rest[el]
        L = ts_rest_L[el]  # hypergraph Laplace
        Nc = ROI_num
        W = np.zeros(shape=(Nc, Nc))
        # Lasso
        # lasso #Min(am) ||xm-Am*am||2 + lamda*||am||1

        X, Y = create_dataset4(ts, belta)
        W, prediction = mutitaskLasso(X, Y, L,alpha,beta,tol,1)
        res = (prediction - Y) ** 2
        sum = 0.0
        error = 0.0
        for row in range(res.shape[0]):
            for col in range(res.shape[1]):
                sum += res[row,col]
        error = sum/(Nc*ts_len_rest)
        print('error:',error)
        x_rest.append(W.flatten())
        y_rest.append(-1)

    x_run1=[]
    y_run1=[]
    for el in range(sub_run1):
        print('run1 sub', el)
        ts = ts_run1[el]
        L = ts_run1_L[el]
        Nc = ROI_num
        W = np.zeros(shape=(Nc, Nc))
        # Lasso
        # lasso #Min(am) ||xm-Am*am||2 + lamda*||am||1
        X, Y = create_dataset4(ts, belta)
      
        W, prediction = mutitaskLasso(X, Y, L,alpha,beta,tol,1)
        res = (prediction - Y) ** 2
        sum = 0.0
        error = 0.0
        for row in range(res.shape[0]):
            for col in range(res.shape[1]):
                sum += res[row, col]
        error = sum / (Nc * ts_len_run1)
        print('error:', error)
        x_run1.append(W.flatten())
        y_run1.append(1)


    # ########################classification#######################
    # LOO
    dataset = x_rest + x_run1
    datalabels = y_rest + y_run1

    loo = LeaveOneOut()
    loo.get_n_splits(dataset)
    predictlabel_list = []
    reallabel_list = []
    test_predict_label = []
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    clf = SVC(C=5, kernel='rbf', gamma='auto')
    count_right_label = 0
    count = 0  
    for train_index, test_index in loo.split(dataset):
        X_train, X_test = dataset[train_index], dataset[test_index]
        Y_train, Y_test = np.array(datalabels)[train_index], np.array(datalabels)[test_index]
        test_predict_label.append(clf.fit(X_train, Y_train).decision_function(X_test))
        predictlabel_list.append(list(clf.predict(X_test)))
        reallabel_list.append(list(Y_test))
        if Y_test == clf.predict(X_test):
            count_right_label += 1
        count += 1
        
    accurancy = count_right_label / len(datalabels)

    TP_count, TN_count, FP_count, FN_count = get_TpTN_FpFn(reallabel_list, predictlabel_list)
    F1_score = (2 * TP_count) / (2 * TP_count + FP_count + FN_count)
    ACC = (TP_count + TN_count) / (TP_count + FN_count + TN_count + FP_count)
    SEN = TP_count / (TP_count + FN_count)
    SPE = TN_count / (TN_count + FP_count)
    AUC = roc_auc_score(reallabel_list, predictlabel_list)

    print('ACC：%.2f%%' % (ACC * 100))
    print('SEN（：%.2f%%' % (SEN * 100))
    print('SPE：%.2f%%' % (SPE * 100))
    print('F1_SCORE：%.2f%%' % (F1_score * 100))
    print('AUC：%.2f%%' % (AUC * 100))

    end = time.time()
    print(end - start)
