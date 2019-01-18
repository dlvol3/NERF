# PageRank for scoring ranking of the nodes
# undirected Network to directed by NetworkX .todirect
# Weight of the edges: EI calculated based on Gini gain

#%%

import networkx as nx


#%%

# Get the Garph from the previous twonet()

tbreast
tbreastsub = tbreast.replace(index, feag)

Gbreast = nx.from_pandas_edgelist(tbreast, "feature_i", "feature_j", "EI")

nx.is_directed(DG)

DG = Gbreast.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

rkrank = [featurelist[i] for i in rkrank]
with open('your_file.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)



tbla
tbla = tbla.replace(index, featurelist)

Gbla = nx.from_pandas_edgelist(tbla, "feature_i", "feature_j", "EI")

nx.is_directed(DG)

DG = Gbla.to_directed()

rktest = nx.pagerank(DG, weight='EI')

rkdata = pd.Series(rktest, name='position')
rkdata.index.name = 'PR'
rkdata
rkrank = sorted(rktest, key= rktest.get, reverse=True)

rkrank = [featurelist[i] for i in rkrank]
with open('your_file_bla.txt', 'w') as f:
    for item in rkrank:
        f.write("%s\n" % item)