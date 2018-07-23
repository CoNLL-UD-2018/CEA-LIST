#! /usr/bin/python 
# coding: utf-8
#%reset -f

#
#   USAGE 
#   python main.py SOURCE_EMBED TARGET_EMBED DIC_SOURCE_TARGET SIZE
#
#   USAGE EXAMPLE
#   python main.py en.vec fr.vec en-fr-dic.txt 100000
#




import numpy as np
from operator import itemgetter
import collections
import sys
import codecs

#Retourne les embeddings au format matrice np
def get_embeddings(filename,max_n_embeddings):
    nparr = []
    words = []
    sys.stderr.write("Loading "+filename+"...\n")
    with codecs.open(filename,"rb") as f:
        reader = codecs.getreader('utf-8')(f, errors='ignore')
        line = reader.readline()
        line = reader.readline()
        line = line.rstrip().split(' ');
        length = len(line)
        words.append(line[0])
        nparr.append(np.array(line[1:],dtype=np.float32))
        for line_num, line in enumerate(reader):
            line = line.rstrip().split(' ')
            if len(line) == length:
                #print(len(words))
                words.append(line[0])
                nparr.append(np.array(line[1:],dtype=np.float32))
                if len(words) > max_n_embeddings:
                    break
    
    nparr = np.stack(nparr)
    nparr = np.stack(nparr)
    embeds = dict(zip(words,nparr))
    sys.stderr.write(filename+" done\n")
    return embeds,words

#Lit le dictionnaire et retourne la liste des mots l1 et l2 (exclut les mots non présents en embeddings ou les mots présents en inter)
def read_dic(filename,embed1,embed2,inter):
    words1 = []
    words2 = []
    with codecs.open(filename,"rb") as f:
        reader = codecs.getreader('utf-8')(f, errors='ignore')
        for line_num, line in enumerate(reader):
            line = line.rstrip().lower().split("\t");
            if len(line) == 2 and line[0] in embed1 and line[1] in embed2 and line[0] not in inter and line[1] not in inter:
                words1.append(line[0])
                words2.append(line[1])
    return words1, words2

#Crée la structure dictionnaire de python
def build_dic(lw1,lw2):
    d = dict()
    for i in range(0,len(lw1)):
        if d.get(lw1[i]) == None:
            d[lw1[i]] = [lw2[i]]
        else:
             d[lw1[i]].append(lw2[i])
    return d

#https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

#Trois fichiers requis:
# Dico l1 -> l2
# Embeddings l1 (repère source) (th)
# Embeddings l2 (repère destination) (ko)
file1 = sys.argv[1];# 
file2 = sys.argv[2];# 
dic_filename = sys.argv[3];# 
max_n_embeddings = int(sys.argv[4]);#100000


#Récupère les matrices d'embeddings et les mots correspondants
embed1,embeds_w1 = get_embeddings(file1,max_n_embeddings)
embed2,embeds_w2 = get_embeddings(file2,max_n_embeddings)

#l'intersection embeds_w1 et embeds_w2 est à rajouter au dictionnaire
se1 = set(embeds_w1)
se2 = set(embeds_w2)
inter = se1.intersection(se2)
inter = list(inter)
#for w in inter:
#	print(w + " " + w);
sys.stderr.write("colliding embeddings :"+ str(len(inter)) +" done\n")
inter = dict(zip(inter,[[i] for i in inter]))

#Liste les mots sources -> dest du dictionnaire
#les mots destinations doivent avoir un embedding (sinon ils sont inutiles)
dic_w1,dic_w2 = read_dic(dic_filename,embed1,embed2,inter)
l1tol2 = build_dic(dic_w1,dic_w2)

l1tol2 = merge_two_dicts(l1tol2, inter)

#Enlève les mots de dic2 non compris dans les embeddings2
#sdic2 = set([i for l in l1tol2.values() for i in l])
#swords2 = set(embed2.keys())
#sdic2 = sdic2.intersection(swords2)
sdic1 = set(l1tol2.keys())
swords1 = set(embeds_w1)
sdic1 = sdic1.intersection(swords1)
sdic1 = list(sdic1)



#Pour chaque mot1 -> embed1
m = 5
for w1 in embeds_w1:
    #Si on a n mots correspondant en l2, on moyenne les embeddings des n mots de l2
    a1 = embed1[w1]
    w2 = l1tol2.get(w1)
    e = None
    if w2 != None:
        e = np.mean([embed2[w2i] for w2i in w2],axis=0)
    #Si on n'a pas de correspondance
    else:
        #On prend les m mots les plus proches de w1 pour lesquels on a une correspondance
        dists = [np.linalg.norm(embed1[i]-a1) for i in sdic1]
        mmin = np.argpartition(np.array(dists), m)[:m] #index des plus proches en l1
        dists = [dists[i] for i in mmin]
        dists = [1e-10 if x == 0 else x for x in dists]
        dists = [1.0/x for x in dists]
        dists = dists/sum(dists)
        
        #On pondère en fonction de la distance
        w2s = [l1tol2[sdic1[i]] for i in mmin] #mots l2 correspondants à ceux des mmin l1
        w = []
        e = []
        for i in range(0,len(w2s)):
            e.append(np.mean([embed2[w2i] for w2i in w2s[i]],axis=0))
            w.append(dists[i])
        e = np.sum([e[i] * w[i] for i in range(0,len(e))],axis=0)
        
    sys.stdout.write(w1.encode('utf-8')+" ")
    for i in  ['{:.2f}'.format(i) for i in e]:
        sys.stdout.write(i+" ")
    sys.stdout.write("\n")















