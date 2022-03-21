import os
import time
import pandas as pd
from matplotlib import pyplot as plt
import json

def compare_runs(paths=None,mapping=None, out_dir=None,plot_metric_training_bool=True,compare_metric_bool=True):
    if out_dir:
        out_dir=os.path.join(out_dir,time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(name=out_dir, exist_ok=True)

    if paths:
        if isinstance(paths,list):
            paths= [p for p in paths if os.path.exists(p)]
    else:
        paths =[key for key in mapping.keys()]
    # find all history files (this is an indicator for a valid run
    history_path_list=[]
    for p in paths:
        history_path_list.extend([os.path.join(root, f) for root, dirs, files in os.walk(p) for
                                  f in files if os.path.basename(os.path.join(root, f))=="history.csv"])
    # init DataFrame
    df = pd.DataFrame(history_path_list,columns=["history_path"])


    # add run directory
    df["run_path"] = df["history_path"].apply(lambda p: os.path.dirname(p))

    # add folder
    df["run_folder"] = df["run_path"].apply(lambda p: os.path.basename(p))

    # add run directory
    df["bleuhistory_path"] = df["run_path"].apply(lambda p: os.path.join(p, "Bleuhistory.csv"))
    # for index, row in df.iterrows():
    #     # s = select_best_modelCheckPoint(row["history_path"],  metric="val_loss_dice",type="minimize")
    #     s = select_best_modelCheckPoint(row["history_path"],  metric='val_loss',type="minimize")
    #
    #     if index==0:
    #         s = s.to_frame(0)
    #         s=s.T
    #         df= pd.concat([df, s], axis=1)
    #     else:
    #         df.loc[index, 'epoch'] = s['epoch']
    #         df.loc[index, 'train_loss'] = s['train_loss']
    #         df.loc[index, 'val_loss'] = s['val_loss']
    #         # df.loc[index, 'val_loss'] = s['val_loss_dice']
    # df.to_csv(os.path.join(out_dir,"summary.csv"))

    if plot_metric_training_bool:
        plot_metric_training(df=df,out_dir=out_dir,mapping=mapping)
            # print("hello")
        # print("hello")
    # print("saving results to: "+out_dir)
    # df["trials"]=df["run_folder"].str.split('_', 1, expand=True)[1]
    return df
 # def compare_metric(df, out_dir=None):
 #     for index, row in df.iterrows():
 #
def plot_metric_training(df,out_dir,mapping=None):
    # df = df.rename(columns={"val_loss_dice":"val_loss"})
    df1 = pd.read_csv(df["history_path"].iloc[0])
    # df1 = df1.rename(columns={"val_loss_dice":"val_loss"})
    metric_list = df1.columns.to_list()
    metric_list.remove("step")


    for metric in metric_list:
        df_metric = pd.DataFrame()
        for index, row in df.iterrows():

            df_tmp1 = pd.read_csv(row["history_path"])
            # df_tmp1 = df_tmp1.rename(columns={"val_loss_dice": "val_loss"})
            print(row["history_path"])
            #delete irrelevant data
            # if df_tmp["loss"].max()>10000:
            #     #work okay it is the last epoch, otherwise need to debug
            #     df_tmp.drop(index=df_tmp["loss"].argmax(), inplace=True)
            df_tmp = df_tmp1[metric].to_frame(row["run_path"])
            df_metric = pd.concat([df_metric, df_tmp], axis=1)

            # plot per run
            plt.figure()
            plot(df_tmp1[metric_list])
            plt.title(mapping[row["run_path"]])
            plt.savefig(os.path.join(out_dir,mapping[row["run_path"]]+".png"))

        #rename
        df_metric = df_metric.rename(columns=mapping)


        #plot all runs
        plt.figure()
        plot(df=df_metric,mapping=mapping)
        plt.legend()
        plt.title(metric)
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(out_dir,metric+".png"))

        metric ="Bleu_score"
        df_metric = pd.DataFrame()
        for index, row in df.iterrows():
            df_tmp1 = pd.read_csv(row["bleuhistory_path"])
            # df_tmp1 = df_tmp1.rename(columns={"val_loss_dice": "val_loss"})
            # print(row["history_path"])
            df_tmp = df_tmp1[metric].to_frame(row["run_path"])
            df_metric = pd.concat([df_metric, df_tmp], axis=1)

            # plot per run
            plt.figure()
            plot(df_tmp1[metric])
            plt.title(mapping[row["run_path"]])
            plt.savefig(os.path.join(out_dir,mapping[row["run_path"]]+"_bleu"+".png"))

        #rename
        df_metric = df_metric.rename(columns=mapping)
        plt.figure()
        plot(df=df_metric,mapping=mapping)
        plt.legend()
        plt.title(metric)
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(out_dir,metric+".png"))

def select_best_modelCheckPoint(path , metric="val_loss_dice",type="minimize"):
    df = pd.read_csv(path)
    if type=="minimize":
        s = df.iloc[df[metric].argmin()]
    elif type=="maximize":
        s = df.iloc[df[metric].argmax()]
    # print("hello")
    return s
def plot(df,mapping=None):
    if mapping:
        df = df.rename(columns=mapping)
    df.plot(grid=True)

def present_metrics(path,out_dir):

    df = pd.read_csv(path)
    df.groupby(["set"])[["dice_ET", "dice_WT", "dice_TC"]].mean()



if __name__ == '__main__':


    out_dir = "/tcmldrive/project/results/python/compare_runs"
    mapping={
             "/tcmldrive/project/results/python/20220319-170215": "CodeBert restore",
             "/tcmldrive/project/results/python/20220319-170137":"CodeBert RenameParameters",
             "/tcmldrive/project/results/python/20220319-170147": "CodeBert RenameLocalVariables",
             "/tcmldrive/project/results/python/20220319-170157": "CodeBert RenameFields",
             "/tcmldrive/project/results/python/20220319-170208": "CodeBert AddDeadCode",
            }
    compare_runs(mapping=mapping, out_dir=out_dir, plot_metric_training_bool=True, compare_metric_bool=True)
