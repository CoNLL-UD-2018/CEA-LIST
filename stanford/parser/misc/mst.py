# !/usr/bin/env python
# -*- encoding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

#***************************************************************
#===============================================================
def find_cycles(edges):
  """"""
  
  vertices = np.arange(len(edges))
  indices = np.zeros_like(vertices) - 1
  lowlinks = np.zeros_like(vertices) - 1
  stack = []
  onstack = np.zeros_like(vertices, dtype=np.bool)
  current_index = 0
  cycles = []
  
  #-------------------------------------------------------------
  def strong_connect(vertex, current_index):
    """"""
    
    indices[vertex] = current_index
    lowlinks[vertex] = current_index
    stack.append(vertex)
    current_index += 1
    onstack[vertex] = True
    
    for vertex_ in np.where(edges == vertex)[0]:
      if indices[vertex_] == -1:
        current_index = strong_connect(vertex_, current_index)
        lowlinks[vertex] = min(lowlinks[vertex], lowlinks[vertex_])
      elif onstack[vertex_]:
        lowlinks[vertex] = min(lowlinks[vertex], indices[vertex_])
    
    if lowlinks[vertex] == indices[vertex]:
      cycle = []
      vertex_ = -1
      while vertex_ != vertex:
        vertex_ = stack.pop()
        onstack[vertex_] = False
        cycle.append(vertex_)
      if len(cycle) > 1:
        cycles.append(np.array(cycle))
    return current_index
  #-------------------------------------------------------------
  
  for vertex in vertices:
    if indices[vertex] == -1:
      current_index = strong_connect(vertex, current_index)
  return cycles

#===============================================================
def find_roots(edges):
  """"""
  
  return np.where(edges[1:] == 0)[0]+1
  
#***************************************************************
def argmax(probs):
  """"""
  
  edges = np.argmax(probs, axis=1)
  return edges
  
#===============================================================
def greedy(probs):
  """"""
  
  edges = np.argmax(probs, axis=1)
  cycles = True
  while cycles:
    cycles = find_cycles(edges)
    for cycle_vertices in cycles:
      # Get the best heads and their probabilities
      cycle_edges = edges[cycle_vertices]
      cycle_probs = probs[cycle_vertices, cycle_edges]
      # Get the second-best edges and their probabilities
      probs[cycle_vertices, cycle_edges] = 0
      backoff_edges = np.argmax(probs[cycle_vertices], axis=1)
      backoff_probs = probs[cycle_vertices, backoff_edges]
      probs[cycle_vertices, cycle_edges] = cycle_probs
      # Find the node in the cycle that the model is the least confident about and its probability
      new_root_in_cycle = np.argmax(backoff_probs/cycle_probs)
      new_cycle_root = cycle_vertices[new_root_in_cycle]
      # Set the new root
      probs[new_cycle_root, cycle_edges[new_root_in_cycle]] = 0
      edges[new_cycle_root] = backoff_edges[new_root_in_cycle]
  return edges


#===============================================================
#===============================================================
  
def dfs(graph,marks,node,stack):
    marks[node] = 1
    stack.append(node)
    #print stack
    for i in range(0,len(graph[node])):
        ndi = graph[node][i]
        #print node,'=>',ndi,'(',marks[ndi],')'
        if marks[ndi] == 1:
            #print node,stack
            return stack,True
        else:
            res,boolean = dfs(graph,marks,ndi,stack)
            if boolean:
                return res,True
            
    #print 'stop'
    stack.pop()
    marks[node] = 2
    return stack,False
        

def find_cycle(graph,n):
    #dfs
    marks = np.zeros([n])
    for i in range(0,len(graph)):
        if marks[i] == 0:
            stack, boolean = dfs(graph,marks,i,[])
            if boolean:
                #print 'ret',stack
                if len(stack) > 0:
                    return stack


#maximize the probs
def chu_liu_edmonds_probs(probs):  

    #Remove v(i) -> v(i) edges
    n = np.shape(probs)[0]
    for i in range(0,n):
        probs[i,i] = 0 #no node is going toward itself   
    base_probs = np.array(probs, copy=True)
   
    #Ici probs[u,v] = proba que u <- v dans l'arbre syntaxique (proba que v soit le parent de u)
    best_path = []
    best_score = 0
    n_repeat = 4
    #algorithm test for i = 0 if we have a minimum tree with one node going to 0, if so it gives the result, 
    #otherwise it gives a minimum tree with only the i-th node with the best probability that have 0 as a parent
    i=0
    for i in range(0,min(n_repeat,n)):
        costs = 1-probs #proba => cout
        
        #Ici cost[u,v] = cout de u <- v (proba faible = cout fort => on minimise les couts => on maximise les probas)
        
        costs = costs.T
        #Ici cost[u,v] = cout de v <- u 
        
        path = chu_liu_edmonds(costs)
        path = np.flip(path,axis=1) #flip for the probs
        score = cost(path,base_probs)
        
        path = np.ascontiguousarray(path,dtype=np.integer)
        path = np.sort(path.view('i8,i8'), order=['f0'], axis=0).view(np.int)
        path = path[0:np.size(path),1]
        
        #if the minimum tree has only one node going to 0 (otherwise, greedy on the first 0 node)
        if i == 0 and np.sum(path == 0) == 1:
            best_path = path
            break;
        if i == 0:
            score = -1
            
        probs = np.array(base_probs, copy=True)
        
        #block from the probabilities the possibility for probas [nodeX->node0] to be equal
        for j in range(0,n):
            probs[j,0] += j/100000.0
        
        #exclude already tested solution
        for j in range(0,i):
            probs[probs[0:n,0] == max(probs[0:n,0]),0] = 0
            
        probs[0:n,0] = probs[0:n,0]*(probs[0:n,0] == max(probs[0:n,0]))
        
        if score > best_score and i != 0 and np.sum(path == 0) == 1:
            best_score = score
            best_path = path
    #Pour des cas très particuliers où flottants sont mal gérés (sur cluster) et les probabilités d'être relié à root basses
    #On peut n'avoir aucun noeud sur root, ou plus d'un noeud sur root sur les itérations suivantes
    #Ces cas ne seront pas enregistrés, mais dans ces cas là et si aucune solution ne convient, on fallback sur l'algorithme greedy de stanford qui gère ces cas
    if len(best_path) == 0:
      return nonprojective(base_probs)
    best_path = np.insert(best_path,0,0)
    return best_path

#Implemented following wikipedia article (minimize the cost, complete graph)
def chu_liu_edmonds(costs):
    #Root is 0
    n = np.shape(costs)[0]
    costs[0:n,0] = np.inf #no node is going toward node0
    
    #Remove v(i) -> v(i) edges
    for i in range(0,n):
        costs[i,i] = np.inf #no node is going toward itself
    
    
    #find min incoming
    mins = np.array([],dtype=np.int)
    for i in range(0,n):
        minj_value = np.inf; 
        minj = -1
        for j in range(0,n):
            #print 'test',j,' ',i,' ',costs[j,i],' ',minj_value
            if costs[j,i] < minj_value:
                minj_value = costs[j,i]
                minj = j
                #print 'ok',j,' ',i,' ',minj_value
        if i != minj and minj_value < 1 and minj_value != -1:
            mins = np.append(mins,[minj,i])
    mins = np.reshape(mins,[np.size(mins)//2,2])
    P = mins
    
    
    #find cycles
    graph = dict(np.reshape(np.concatenate((range(0,n),np.repeat(None,n))),[2,n]).T)
    for i in range(0,n):
            graph[i] = list()
    for i in range(0,np.shape(P)[0]):
        if not P[i,1] in graph[P[i,0]]:
            graph[P[i,0]].append(P[i,1])
    #cycles = strongly_connected_components(graph)
    cycle = find_cycle(graph,n)
    
    #cycles
    if cycle != None and len(cycle) > 0 :
        #print("cycle(s)", cycle)
        
        #transform cycle
        mask = np.ones(n, bool)
        mask[cycle] = 0
        oldnodes = costs[np.invert(mask),]
        dprime = costs[mask,][0:len(mask),mask]
        dprime = np.concatenate([dprime,np.ones([1,np.sum(mask)])*np.inf],axis=0)
        dprime = np.concatenate([dprime,np.ones([1,np.sum(mask)+1]).T*np.inf],axis=1)
        
        
        cycle = np.array(cycle)
        vc = np.shape(dprime)[0]-1
        dic = dict()
        for u in range(0,n):
            newu = u - np.sum(u > cycle)
            for v in range(1,n):
                newv = v - np.sum(v > cycle)
                if u != v:
                    if u not in cycle and v in cycle:
                        pi_v = mins[np.where(mins[0:np.shape(mins)[0],1]==v)[0][0],0]
                        #print(u,'->',v,':',u,'-> c (',costs[u,v],'-',costs[pi_v,v],')')
                        if costs[u,v] -  costs[pi_v,v] < dprime[newu,vc]:
                            #print(costs[u,v] -  costs[pi_v,v],'<', dprime[newu,vc])
                            dprime[newu,vc] = costs[u,v] -  costs[pi_v,v]
                            dic[tuple([newu,vc])] = tuple([u,v])
                    elif u in cycle and v not in cycle:
                        if costs[u,v] < dprime[vc,newv]:
                            #print(costs[u,v],'<', dprime[vc,newv])
                            dprime[vc,newv] = min(dprime[vc,newv],costs[u,v])
                            dic[tuple([vc,newv])] = tuple([u,v])
                        
                    elif u not in cycle and v not in cycle:
                        dprime[newu,newv] = costs[u,v]
                        dic[tuple([newu,newv])] = tuple([u,v])


        Aprime = chu_liu_edmonds(dprime)
        u_blocked = Aprime[np.where(Aprime[0:np.shape(Aprime)[0],1]==vc)[0][0],0]
        edge = dic[tuple([u_blocked,vc])]
        v_blocked = edge[1]
        pi_v = mins[np.where(mins[0:np.shape(mins)[0],1]==v_blocked)[0][0],0]
       
        r = np.array([], dtype=np.integer)
        for i in range(0,len(cycle)):
            u = cycle[i]
            v = cycle[(i+1)%len(cycle)]
            if not (u == pi_v and v == v_blocked):
                r = np.concatenate((r,np.array([u,v])))
        r = np.reshape(r,[np.size(r)//2,2])
                
        for tu in Aprime:
            corresponding_edge = dic[tuple([tu[0],tu[1]])]
            edge = np.reshape(np.array([corresponding_edge[0],corresponding_edge[1]]),[2,])
            r = np.concatenate((r,np.reshape(edge,[1,2])))
        r = np.reshape(r,[np.size(r)//2,2])
        
                
        #Todo: erreur si colonne de droite de r plus grande que unique(colonne de droite)
    #no cycles
    else :
        #print("no cycles")
        r = P
    
    
    return r;
#measure the cost of the path
def cost(paths,array):
    s = 0   
    for edges in paths:
        u = edges[0]
        v = edges[1]
        s = s + array[u,v]
        #sys.stdout.write('+'+str(array[u,v]))
    return s
#===============================================================
#===============================================================


#===============================================================
def nonprojective(probs):
  """"""
  
  probs *= 1-np.eye(len(probs)).astype(np.float32)
  probs[0] = 0
  probs[0,0] = 1
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  #edges = chu_liu_edmonds(probs)
  edges = greedy(probs)
  roots = find_roots(edges)
  best_edges = edges
  best_score = -np.inf
  if len(roots) > 1:
    for root in roots:
      probs_ = make_root(probs, root)
      #edges_ = chu_liu_edmonds(probs_)
      edges_ = greedy(probs_)
      score = score_edges(probs_, edges_)
      if score > best_score:
        best_edges = edges_
        best_score = score
  return best_edges

#===============================================================
def make_root(probs, root):
  """"""
  
  probs = np.array(probs)
  probs[1:,0] = 0
  probs[root,:] = 0
  probs[root,0] = 1
  probs /= np.sum(probs, axis=1, keepdims=True)
  return probs

#===============================================================
def score_edges(probs, edges):
  """"""
  
  return np.sum(np.log(probs[np.arange(1,len(probs)), edges[1:]]))

#***************************************************************
if __name__ == '__main__':
  def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=1, keepdims=True)
  probs = softmax(np.random.randn(100,100))
  probs *= 1-np.eye(len(probs)).astype(np.float32)
  probs[0] = 0
  probs[0,0] = 1
  probs /= np.sum(probs, axis=1, keepdims=True)
  
  edges = nonprojective(probs)
  roots = find_roots(edges)
  best_edges = edges
  best_score = -np.inf
  if len(roots) > 1:
    for root in roots:
      probs_ = make_root(probs, root)
      edges_ = nonprojective(probs_)
      score = score_edges(probs_, edges_)
      if score > best_score:
        best_edges = edges_
        best_score = score
  edges = best_edges
  print(edges)
  print(np.arange(len(edges)))
  print(find_cycles(edges))
  print(find_roots(edges))
