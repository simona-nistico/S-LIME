import sklearn as sk;
import sklearn.tree;
import numpy as np;
import numpy.random;
import pickle;
import os;
import os.path;
import matplotlib;
import matplotlib.pyplot as plt;
import sys;
if len(sys.argv)>1:
    ds = sys.argv[1];
else:
    print("USAGE: python3 grafici.py <ds>");
    quit();
with open(os.path.join('dataset','score_tree_'+ds+'.pickle'),'rb') as f: scfs, scfsli, scint = np.array(pickle.load(f));
with open(os.path.join('dataset','score_lime_'+ds+'.pickle'),'rb') as f: scli, sclifs, scint = np.array(pickle.load(f));
with open(os.path.join('dataset','score_sents_'+ds+'.pickle'),'rb') as f: sc = np.array(pickle.load(f));
with open(os.path.join('dataset','id_'+ds+'.pickle'),'rb') as f: ids = np.array(pickle.load(f));
algo = '$\mathcal{S}$-LIME';
scfs = scfs[ids];
scli = scli[ids];
scfsli = scfsli[ids];
sclifs = sclifs[ids];
scint = scint[ids];
sc = sc[ids];
ntot = sc.shape[0];
n = 10;
ns = int(ntot/n);
sc_s1 = np.zeros(ns);
scli_s1 = np.zeros(ns);
scfs_s1 = np.zeros(ns);
idsc = np.argsort(sc);
for i in range(ns):
    sc_s1[i] = np.mean(sc[idsc[i*n:(i+1)*n]]);
    scli_s1[i] = np.mean(scli[idsc[i*n:(i+1)*n]]);
    scfs_s1[i] = np.mean(scfs[idsc[i*n:(i+1)*n]]);
plt.figure(1);
plt.plot([*range(ns)], scfs_s1, [*range(ns)], scli_s1, [*range(ns)], sc_s1);
plt.axis([0,ns,0,0.4]);
plt.title(ds.upper()+" dataset.");
plt.legend([r''+algo, r'LIME', r'Sentence']);
plt.xlabel("Sentence id.");
plt.ylabel("Significance of explanations.");
plt.xticks([]);
plt.savefig("detailed_significance_"+ds+".eps");
plt.show();
plt.figure(2);
ni = 3;
plt.plot(sc_s1[ni:ns], scfs_s1[ni:ns]/sc_s1[ni:ns], sc_s1[ni:ns], scli_s1[ni:ns]/sc_s1[ni:ns]);
plt.title(ds.upper()+" dataset.");
plt.legend([r''+algo, r'LIME']);
plt.xlabel("Significance of sentences.");
plt.ylabel("Significance of explanations.");
plt.xscale('log');
plt.savefig("sentence_significance_"+ds+".eps");
plt.show();
plt.figure(3);
plt.boxplot([scfs, scli], labels=[r''+algo, r'LIME'], showfliers=False);
plt.title(ds.upper()+" dataset.");
plt.ylabel("Significance of explanations.");
plt.savefig("comparison_"+ds+".eps");
plt.show();
plt.figure(4);
plt.title(ds.upper()+" dataset.");
plt.boxplot([scfsli, scint, sclifs],
            labels=[r''+algo+"-"+r'LIME', r'Intersection', r'LIME'+"-"+r''+algo],
            showfliers=False);
plt.ylabel("Significance of explanations.");
plt.savefig("mean_significance_"+ds+".eps");
plt.show();