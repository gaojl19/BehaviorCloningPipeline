from sqlite3 import DataError
from graphviz import Digraph
import json
import numpy as np


def plot_network(n_modules, n_layers, path):
    json_path = path + "_weight.json"
    with open(json_path, "r") as f:
        weights = json.load(f)
    
    for task in weights.keys():
        
        w = weights[task][0]
        max = 0
        for v in w:
            max = v if v > max else max
        for i in range(len(w)):
            w[i] /= max
        print(w)
        
        if len(w) != (n_layers-1) * n_modules * n_modules:
            print(len(w), (n_layers-1) * n_modules * n_modules)
            raise DataError("The length of data is inconsistent with the specified modules and layers")
        
        # 实例化一个Digraph对象(有向图)，name:生成的图片的图片名，format:生成的图片格式
        dot = Digraph(name=task, comment="network", format="png")
        dot.attr(rankdir = "BT")
        dot.edge_attr.update(arrowsize='0.8')

        for i in range(n_layers):
            for j in range(n_modules):
                # 生成图片节点，name：这个节点对象的名称，label:节点名,color：画节点的线的颜色
                dot.node(name=str(i) + "_" + str(j), shape="square")
        idx = 0
        for i in range(n_layers-1):
            for j in range(n_modules):
                for k in range(n_modules):
                    # 在节点之间画线，label：线上显示的文本,color:线的颜色
                    dot.edge(str(i) + "_" + str(j), str(i+1) + "_" + str(k), color="0.010 0.0100 " + str(1-w[idx]), style="bold")
                    idx += 1

       
        # 画图，filename:图片的名称，若无filename，则使用Digraph对象的name，默认会有gv后缀
        # directory:图片保存的路径，默认是在当前路径下保存
        print(path)
        dot.view(filename=task + "_weight", directory=path)
        exit(0)
        
    
if __name__ == "__main__":
    plot_network(n_modules=10, n_layers=2, path="./fig/mt10_similar_fixed/")