# -*- coding: utf-8 -*-
import numpy as np
from vasc import vasc
from helpers import clustering,measure,print_2D
from utils import load_graph, load_data_origin_data
from config import config

if __name__ == '__main__':
    file_path = ""
    dataset = load_data_origin_data(file_path,scaling=True)
    X=dataset.x
    label=dataset.y
    
    expr = dataset.x
    n_cell,_ = expr.shape
    if n_cell > 150:
        batch_size=config['batch_size']
    else:
        batch_size=32 
    #expr = np.exp(expr) - 1 
    #expr = expr / np.max(expr)

#    
#    percentage = [0.5]
#    
#    for j in range(1):
#        print(j)
#        p = percentage[j]
#        samples = np.random.choice( n_cell,size=int(n_cell*p),replace=True )
#        expr_train = expr[ samples,: ]
#        label_train = label[samples]
    
    #latent = 2
    for i in range(1):
        print("Iteration:"+str(i))
        res = vasc( expr,var=False,
                    latent=config['latent'],
                    annealing=False,
                    batch_size=batch_size,
                    label=label,
                    scale=config['scale'],
                    patience=config['patience'] 
                )   
    #print("============SUMMARY==============")
    #k = len(np.unique(label))
    #for r in res:
    #    print("======"+str(r.shape[1])+"========")
    #    pred,si = clustering( r,k=k )
    #    if label is not None:
    #        metrics = measure( pred,label )
    #