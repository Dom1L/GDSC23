import json

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import get_state_dict, batch_to_device


def inference_k_random(net, state_dict_path, test_dl, test_metadata_df, loss_fn, label_path, k=1):
    with open(label_path, 'r') as infile:
        data = json.load(infile)

    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    sd = get_state_dict(state_dict_path)
    model = net.eval().to(device)
    model.load_state_dict(sd)

    preds = []
    for _ in range(k):
        with torch.no_grad():
            _preds = []
            for batch in tqdm(test_dl):
                batch = batch_to_device(batch, device)
                with torch.cuda.amp.autocast():
                    out = model(batch['wave'])
                    _preds += [out.cpu().numpy()]
            preds.append(np.vstack(_preds))
            
    preds = np.mean(preds, axis=0)
    test_metadata_df['predicted_class_id']  = preds.argmax(axis=-1)
    torch.cuda.empty_cache()
    return test_metadata_df


def inference_all(net, state_dict_path, test_dl, test_metadata_df, loss_fn, label_path, k=1):
    with open(label_path, 'r') as infile:
        data = json.load(infile)

    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    sd = get_state_dict(state_dict_path)
    model = net.eval().to(device)
    model.load_state_dict(sd)

    preds = []
    for i in range(556):
        with torch.no_grad():
            _preds = []
            for batch in tqdm(test_dl):
                batch = batch_to_device(batch, device)
                with torch.cuda.amp.autocast():
                    out = model(batch['wave'])
                    _preds += [out.cpu().numpy()]
            preds.append()
            
    preds = np.mean(preds, axis=0)
    test_metadata_df['predicted_class_id']  = preds.argmax(axis=-1)
    torch.cuda.empty_cache()
    return test_metadata_df




def error_analysis(exp_path):
    df = pd.read_csv(f'{exp_path}/val_predictions.csv')
    df_eval = df[['label', 'predicted_class_id']]
    y_pred = df_eval['predicted_class_id']
    y_true=df_eval['label']

    cm = metrics.confusion_matrix(y_true, y_pred)
    np.save(f"{exp_path}/cm.npy", cm)

    plot_confusion_matrix(cm, exp_path)
    
    report = metrics.classification_report(y_true, y_pred, digits=3,  output_dict=True)
    evaluation = pd.DataFrame(report).transpose()
    evaluation["accuracy"] = ""
    wrong = 0
    for i in range(0,66):
        df_to_eval = df_eval[df_eval['label'] == i]
        for j in df_to_eval['predicted_class_id']:
            if j != i:
                wrong += 1
            else:
                continue
        evaluation['accuracy'][i] = (len(df_to_eval)-wrong)/len(df_to_eval)
        wrong = 0
    pd.options.display.float_format = "{:,.2f}".format
    evaluation.to_csv(f'{exp_path}/val_evaluation.csv') 
    
    
def plot_confusion_matrix(cm, exp_path):
    fig = plt.figure(figsize=(16, 14))
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, fmt = 'g', cmap="magma", mask=cm==0, vmax=10)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.set_ylabel('True', fontsize=20)
    plt.yticks(rotation=0)
    plt.title('Confusion Matrix Validation Set', fontsize=20)
    plt.savefig(f'{exp_path}/conf_matrix_best_model.png')
    plt.close()