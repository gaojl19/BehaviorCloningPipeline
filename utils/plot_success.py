from json.tool import main
import json
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter

def plot_success_curve(path1, path2, label1, label2, tag, plot_prefix):
    
    json_path = path1 + "_success.json"
    with open(json_path, "r") as f:
        json_file = json.load(f)
        curve1 = list(json_file.values())
    
    json_path = path2 + "_success.json"
    with open(json_path, "r") as f:
        json_file = json.load(f)
        curve2 = list(json_file.values())
    
    iteration = range(1, max(len(curve1), len(curve2))+1)
    if len(curve1) > len(curve2):
        for _ in range(len(curve1)-len(curve2)):
            curve2.append(curve2[-1])
    else:
        for _ in range(len(curve2)-len(curve1)):
            curve1.append(curve1[-1])
            
            
    # smooth
    curve1 = savgol_filter(curve1, 11, 3)
    curve2 = savgol_filter(curve2, 11, 3)
    
    df = {  label1: curve1,
            label2: curve2 }
    
    sns.set("paper")
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("bright", 2)
    
    data = pd.DataFrame(data=df, index=iteration)
    ax=sns.lineplot(data=data, palette=palette)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(tag + " success")
    ax.set_title(tag + " Mean Success Curve")
    ax.set(ylim=(-0.1, 1.1))
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + tag + "_success_curve.png")
    
    fig.clf()


def plot_all_curve(paths, labels, tag, plot_prefix):
    
    curve = []
    max_len = 0
    for path in paths: 
        json_path = path + "_success.json"
        with open(json_path, "r") as f:
            json_file = json.load(f)
            origin_success = list(json_file.values())
            
            # smooth
            new_curve = savgol_filter(origin_success, 31, 3)
            curve.append(new_curve)
            max_len = len(new_curve) if len(new_curve) > max_len else max_len
        
            
    iteration = range(1, max_len+1)
    print(iteration)
    assert len(labels) == len(curve)
    df = {}
    
    for i in range(len(curve)):
        c = curve[i].tolist()
        padding = max_len - len(c) if max_len >len(c) else 0
        for _ in range(padding):
            c.append(c[-1])
        df[labels[i]] = c
    
    sns.set("paper")
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("bright", len(curve))
    
    data = pd.DataFrame(data=df, index=iteration)
    ax=sns.lineplot(data=data, palette=palette)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(tag + " success")
    ax.set_title(tag + " Mean Success Curve")
    ax.set(ylim=(-0.1, 1.1))
    
    fig = ax.get_figure()
    fig.savefig(plot_prefix + tag + "_success_curve.png")
    
    fig.clf()
    
    
if __name__ == "__main__":
    path1 = "./fig/mt10_hard_2_20_32_fixed/"
    path2 = "./fig/mt10_hard_2_20_32_dp0.5_fixed/"
    label1 = "SM baseline"
    label2 = "SM with random dropout"
    task_name = "SoftModule_2_20_32_0.5dp"
    plot_prefix = "./fig/MT10_Hard_Result/"
    
    plot_success_curve(path1=path1, path2=path2, label1=label1, label2=label2, tag=task_name, plot_prefix=plot_prefix)
    
    # keyword = "_regWeights"
    # path = [ "./fig/mt10_hard_mhsac_fixed/", 
    #          "./fig/mt10_hard_baseline_2_80_8_fixed/",
    #          "./fig/mt10_hard_2_2_256" + keyword + "_fixed/",
    #          "./fig/mt10_hard_2_10_64"+ keyword + "_fixed/",
    #          "./fig/mt10_hard_2_20_32"+ keyword + "_fixed/",
    #          "./fig/mt10_hard_2_40_16"+ keyword + "_fixed/"]
    # label = [ "multi-head", "MLP", "2/2/256", "2/10/64", "2/20/32", "2/40/16"]
    # tag = "regWeights"
    # plot_prefix = "./fig/MT10_Hard_Result/"
    # plot_all_curve(path, label, tag, plot_prefix)
    # path2 = "./fig/mt10_hard_2_20_32_regWeights_fixed/"
    # label1 = "SM baseline"
    # label2 = "SM with L1 Regularization on Weights"
    # task_name = "SoftModule_2_20_32_regWeights"
    # plot_prefix = "./fig/MT10_Hard_Result/"
    
    # plot_success_curve(path1=path1, path2=path2, label1=label1, label2=label2, tag=task_name, plot_prefix=plot_prefix)
    