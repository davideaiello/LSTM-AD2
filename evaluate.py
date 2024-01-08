import parser
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging
import dataset
from model import LSTMAD
from tqdm import tqdm
from dataset import return_dataloader

args = parser.parse_arguments()   

def compute_auc_roc(fpr, tpr):
    auc = 0
    for i in range(1, len(fpr)):
        auc += 0.5 * (tpr[i] + tpr[i-1]) * (fpr[i] - fpr[i-1])
    return auc

def compute_auc_pr(sens, prec):
    auc = 0
    for i in range(1, len(sens)):
        auc += 0.5 * (sens[i] - sens[i-1]) * (prec[i] + prec[i-1]) 
    return auc

def compute_auc_prrt(sens, prec, ths):
    auc = 0
    for i in range(1, len(sens)):
        auc += 0.5 * (prec[i] + prec[i-1]) * (sens[i] - sens[i-1]) * (ths[i] - ths[i-1])
    return auc

def compute_metrics(anomaly_scores_norm, df_test, df_collision, tot_anomalies, th=None):
    roc = list()
    sens = list()           # recalls o tpr
    spec = list()
    fpr = list()
    f1 = list()
    f0_1= list()
    prec = list()
    cm_list = list()
    anomlay_indexes_dict = dict()
    acc_with_err = list()
    step = 0.1
    ths = np.arange(0, 1, step)
    if th is None:
        for threshold in tqdm(ths):
            df_anomaly = df_test.loc[np.array(anomaly_scores_norm <= threshold)]
            tp = 0                                                          # true positive per quella threshold
            anomaly_indexes = list()
            for index, row in df_anomaly.iterrows():
                for _, collision_row in df_collision.iterrows():
                    if (row['time'] >= collision_row['start']) and (row['time'] <= collision_row['end']):
                        anomaly_indexes.append(index)
                        tp += 1
    
            cm_anomaly = np.zeros((2,2))
            n_sample = len(df_test)
            n_not_collision = n_sample - tot_anomalies
            n_detected = len(df_anomaly)

            fp = n_detected - tp
            fn = tot_anomalies - tp
            tn = n_not_collision - fp

            cm_anomaly[0, 0] = tn
            cm_anomaly[0, 1] = fp
            cm_anomaly[1, 0] = fn
            cm_anomaly[1, 1] = tp

            cm_list.append(cm_anomaly)
            recall = tp / (tp + fn)
            sens.append(recall)
            fpr.append(1 - tn /(tn + fp))
            precision = tp / (tp + fp)
            prec.append(precision)
            spec.append(tn /(tn + fp))
            f1.append(2 * tp / (2 * tp + fp + fn))
            f0_1.append((1 + 0.1**2) * tp / ((1 + 0.1**2) * tp +  0.1**2*fp + fn))
            cm_anomaly_norm = cm_anomaly.astype('float') / cm_anomaly.sum(axis=1)[:, np.newaxis]
            acc_with_err.append( (np.mean(np.diag(cm_anomaly_norm)), np.std(np.diag(cm_anomaly_norm))) )
            anomlay_indexes_dict[threshold] = anomaly_indexes
        
        f1_max = max(f1)
        f0_1_max = max(f0_1)
        max_index_f1 = f1.index(f1_max)
        max_index_f0_1 = f0_1.index(f0_1_max)
        th_f1_max = max_index_f1 * step
        th_f0_1_max = max_index_f0_1 * step
        logging.info(f"f1: {f1_max} at th: {th_f1_max}")
        logging.info(f"f0.1: {f0_1_max} at th: {th_f0_1_max}")
        auc_roc = compute_auc_roc(fpr, sens)                # Area Under the Receiver Operating Characteristic
        auc_pr = compute_auc_pr(sens, prec)                 # Area Under the Precision-Recall Curve
        auc_ptrt = compute_auc_prrt(sens, prec, ths)        # Area Under the Precision-Recall-Threshold Curve
        logging.info(f"AUC-ROC: {auc_roc}")
        logging.info(f"AUC-PR: {auc_pr}")
        logging.info(f"AUC-PtRt: {auc_ptrt}")
        return sens, fpr, f0_1_max
    else:
        df_anomaly = df_test.loc[np.array(anomaly_scores_norm <= th)]
        tp = 0                                                          # true positive per quella threshold
        anomaly_indexes = list()
        for index, row in df_anomaly.iterrows():
            for _, collision_row in df_collision.iterrows():
                if (row['time'] >= collision_row['start']) and (row['time'] <= collision_row['end']):
                    anomaly_indexes.append(index)
                    tp += 1

        cm_anomaly = np.zeros((2,2))
        n_sample = len(df_test)
        n_not_collision = n_sample - tot_anomalies
        n_detected = len(df_anomaly)

        fp = n_detected - tp
        fn = tot_anomalies - tp
        tn = n_not_collision - fp

        cm_anomaly[0, 0] = tn
        cm_anomaly[0, 1] = fp
        cm_anomaly[1, 0] = fn
        cm_anomaly[1, 1] = tp

        f1 = 2 * tp / (2 * tp + fp + fn)
        f0_1 = (1 + 0.1**2) * tp / ((1 + 0.1**2) * tp +  0.1**2*fp + fn)
        logging.info(f"f1: {f1} at th: {th} for the test set")
        logging.info(f"f0.1: {f0_1} at th: {th} for the test set")
    
def plot_hist(anomaly_scores_norm, df_collision, df):
    logging.info(f"Counting the total number of anomalies...")
    tot_anomalies = 0
    index_anomaly = []
    idx = 0
    for _, row in df.iterrows():
        for _, collision_row in df_collision.iterrows():
            if (row['time'] >= collision_row['start']) and (row['time'] <= collision_row['end']):
                tot_anomalies += 1
                index_anomaly.append(idx)
        idx += 1
    logging.info(f"Anomalies detected: {tot_anomalies}")
    y_true = np.zeros_like(anomaly_scores_norm)
    y_true[index_anomaly] = 1
    anomaly_values = anomaly_scores_norm[index_anomaly]
    normal_values = np.delete(anomaly_scores_norm, index_anomaly)

    plt.hist(anomaly_values, bins=30, color='tab:red', ec="salmon", alpha=0.5, label='Anomalies')
    plt.hist(normal_values, bins=30, color="tab:blue", ec="dodgerblue", alpha=0.5, label='Normal')

    plt.xlabel('Values')
    plt.ylabel('Occurencies')
    plt.legend(loc='upper right')
    plt.title('Distribution')
    plt.show()
    return tot_anomalies

def compute_anomaly_scores(model, dataloader):
    errors = []
    for x, y in tqdm(dataloader):
        if args.device == 'cuda':
            x, y = x.cuda(), y.cuda()
        y_hat = model.forward(x)
        e = torch.abs(y.reshape(*y_hat.shape) - y_hat)
        errors.append(e)
    errors = torch.cat(errors)
    anomaly_scores = model.anomaly_scorer.forward(errors.mean(dim=1))
    anomaly_scores_norm = (anomaly_scores - np.min(anomaly_scores)) / (np.max(anomaly_scores) - np.min(anomaly_scores))
    return anomaly_scores_norm

def evaluation(model, pipeline):
    df_collision, X_collisions, df_test = dataset.read_folder_collisions(args.dataset_folder, args.frequency)
    X_collisions = dataset.preprocess_data(X_collisions, pipeline, train=False)
    if args.test_split == True:
        Dataloader_collisions, DataLoader_val, df_col, df_val = dataset.split_data(X_collisions, args.test_split, df_test)
        logging.info(f"Computing threshold on a test set subset")   
        model.eval()

        anomaly_scores_norm = compute_anomaly_scores(model, DataLoader_val)
        df_val = df_val[-anomaly_scores_norm.shape[0]:] 
        tot_anomalies = plot_hist(anomaly_scores_norm, df_collision, df_val)
        _, _, th = compute_metrics(anomaly_scores_norm, df_val, df_collision, tot_anomalies)
        
        anomaly_scores_norm = compute_anomaly_scores(model, Dataloader_collisions)
        df_col = df_col[-anomaly_scores_norm.shape[0]:] 
        tot_anomalies = plot_hist(anomaly_scores_norm, df_collision, df_col)
        logging.info(f"Computing metrics on test set") 
        compute_metrics(anomaly_scores_norm, df_col, df_collision, tot_anomalies, th)
    else:
        Dataloader_collisions = return_dataloader(X_collisions) 
        anomaly_scores_norm = compute_anomaly_scores(model, Dataloader_collisions)
        df_test = df_test[-anomaly_scores_norm.shape[0]:] 
        tot_anomalies = plot_hist(anomaly_scores_norm, df_collision, df_test)
        logging.info(f"Computing metrics on test set") 
        fpr, tpr, _ = compute_metrics(anomaly_scores_norm, df_test, df_collision, tot_anomalies)
        plt.title("Roc Curve")
        plt.plot(fpr, tpr)
        plt.show()
    
if args.resume == True:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.info(f"Resuming model....")   
    logging.info(f"Arguments: {vars(args)}")   
    checkpoint = torch.load(args.model_path)
    X_train = dataset.read_folder_normal(args.dataset_folder, args.frequency)
    X_train, pipeline = dataset.preprocess_data(X_train)

    model = LSTMAD(X_train.shape[1], args.lstm_layers, args.window_size, args.prediction_length)
    model.load_state_dict(checkpoint["model"])
    anomaly_scorer = checkpoint["anomaly_scorer"]
    model.anomaly_scorer = anomaly_scorer 

    if args.device == 'cuda':
        model = model.to('cuda')

    evaluation(model, pipeline)





