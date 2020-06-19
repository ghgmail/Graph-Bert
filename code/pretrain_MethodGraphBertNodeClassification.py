from code.DatasetLoader import DatasetLoader
from code.MethodBertComp import GraphBertConfig
from code.MethodGraphBertNodeClassification import MethodGraphBertNodeClassification
from code.MethodGraphBertGraphClustering import MethodGraphBertGraphClustering
from code.ResultSaving import ResultSaving
from code.Settings import Settings
import numpy as np
import torch


#---- 'cora' , 'citeseer', 'pubmed' ----

dataset_name = 'cora'

np.random.seed(1)
torch.manual_seed(1)

#---- cora-small is for debuging only ----
if dataset_name == 'cora-small':
    nclass = 7
    nfeature = 1433
    ngraph = 10
elif dataset_name == 'cora':
    nclass = 7
    nfeature = 1433
    ngraph = 2708
elif dataset_name == 'citeseer':
    nclass = 6
    nfeature = 3703
    ngraph = 3312
elif dataset_name == 'pubmed':
    nclass = 3
    nfeature = 500
    ngraph = 19717