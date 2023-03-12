#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:55:42 2023

@author: liuyan
"""
import magic
import scprep

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns 
import anndata as ad
from evaluation import nmi_ari_eva
new_adata=sc.read_h5ad("trj/packer_embryo.h5ad")

#sc.pp.recipe_zheng17(adata)

y=new_adata.obs["cell_type"]

magic_op = magic.MAGIC()
X= magic_op.fit_transform(new_adata.X)
adata=ad.AnnData(X)
adata.write_csvs("MAGIC_imputation.csv")

np.savetxt('MAGIC_imputation.csv', X, delimiter = ',')