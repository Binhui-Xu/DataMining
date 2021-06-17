#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os 
from collections import Counter
from datetime import datetime
from itertools import combinations
import networkx as nx
import seaborn as sns1
import matplotlib.pyplot as plt
import random
color = sns.color_palette()

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None  # default='warn'


aisles_path='./aisles.csv'
departments_path='./departments.csv'
opp_path='./order_products__prior.csv'
opt_path='./order_products__train.csv'
orders_path='./instacart/orders.csv'
products_path='./instacart/products.csv'


# In[2]:


def read_data():
    global orders,aisles,departments,order_products_prior,order_products_train,products
    aisles=pd.read_csv(aisles_path)
    departments=pd.read_csv(departments_path)
    order_products_prior=pd.read_csv(opp_path)
    order_products_train=pd.read_csv(opt_path)
    orders=pd.read_csv(orders_path)
    products=pd.read_csv(products_path)


# In[3]:


def data_exploary(transactions_df):
    product_frequency = transactions_df.product_id.value_counts() / n_orders
    plt.hist(product_frequency, bins = 100)
    plt.xlim((0,0.01))
    my_x_ticks = np.arange(0,0.01, 0.001)
    plt.title('Number of times each product frequency occurs')
    plt.xlabel('product frequency')
    plt.ylabel('number of times');
    
    plt.hist(product_frequency, bins = 100)
    plt.xlim((0,0.02))
    my_x_ticks = np.arange(0,0.01, 0.001)
    plt.title('Number of times each product frequency occurs')
    plt.xlabel('product frequency')
    plt.ylabel('number of times')
    plt.ylim([0, 1000])


# In[4]:


def get_transactionDf():
    transactionDf = pd.merge(order_products_prior, products, on='product_id', how='left')
    transactionDf = pd.merge(transactionDf, aisles, on='aisle_id', how='left')
    transactionDf = pd.merge(transactionDf, departments, on='department_id', how='left')
    return transactionDf


# In[5]:


def get_itemset(transactions_df,max_length):
        transactions_by_order = transactions_df.groupby('order_id')['product_id']
        max_length_reference = max_length
        for order_id, order_list in transactions_by_order:
            max_length = min(max_length_reference, len(order_list))
            order_list = sorted(order_list)
            for l in range(2, max_length + 1):
                product_combinations = combinations(order_list, l)
                for combination in product_combinations:
                    yield combination


# In[6]:


def get_product_name(product_ids):
    if type(product_ids) == int:
        return products_id_to_name[product_ids]
    names = []
    for prod in product_ids:
        name = products_id_to_name[prod]
        names.append(name)
    names = tuple(names)
    return names


# In[7]:


def association_rules(order_products, min_support, min_length = 2, max_length = 5,min_confidence = 0.2, min_lift = 1.0):
    #Loading dataset
    data = order_products[['order_id', 'product_id']]
    #compute product and order supports
    n_orders = len(set(data.order_id))
    product_freq = data.product_id.value_counts()/n_orders
    candidate_products = product_freq[product_freq >= min_support]
    freqDf = data[data.product_id.isin(candidate_products.index)]
    order_sizes = freqDf.order_id.value_counts()
    orders_freq = order_sizes[order_sizes >= min_length]
    freqDf = freqDf[freqDf.order_id.isin(orders_freq.index)]
    #find product itemsets and compute supports
    itemsets = get_itemsets(freqDf,max_length)
    counter = Counter(itemsets).items()
    itemsets_count = pd.Series([x[1] for x in counter], index = [x[0] for x in counter])
    itemsets_freq = itemsets_count/n_orders
    candidate_itemsets = itemsets_freq[itemsets_freq >= min_support]
    candidate_itemsets = candidate_itemsets[candidate_itemsets.index.map(len) >= min_length]
    #create rules table
    rulesDf=buildDF(candidate_itemsets)
    #fill evaluation measure in rules table
    support = {**{k: v for k, v in candidate_products.items()},
               **{k: v for k, v in itemsets_freq.items()}}
    rulesDf[['support_A', 'support_B', 'support_AB']] = rulesDf[['A', 'B', 'AB']].applymap(lambda x: support[x])
    rulesDf.drop('AB', axis = 1, inplace = True)
    rulesDf['confidence'] = rulesDf.support_AB/rulesDf.support_A
    rulesDf['lift'] = rulesDf.confidence / rulesDf.support_B
    rulesDf = rulesDf[rulesDf.confidence >= min_confidence]
    rulesDf = rulesDf[rulesDf.lift >= min_lift]
    rulesDf = rulesDf.sort_values(by = 'lift', ascending = False).reset_index(drop = True)
    #convert product id to name for recommendation analysis
    rulesDf[['A', 'B']] = rulesDf[['A', 'B']].applymap(get_product_name)
#     print('{} rules were generated'.format(len(apriori_df)))
    return rulesDf


# In[ ]:


def buildDF(candidate_itemsets):
    print('Populating dataframe...')
    A = []
    B = []
    AB = []
    for c in candidate_itemsets.index:
        c_length = len(c)
        for l in range(1, c_length):
            comb = itemsets(c, l)
            for a in comb:
                AB.append(c)
                b = list(c)
                for e in a:
                    b.remove(e)
                b = tuple(b)
                if len(a) == 1:
                    a = a[0]
                A.append(a)
                if len(b) == 1:
                    b = b[0]
                B.append(b)
    rulesDf = pd.DataFrame({'A': A,'B': B,'AB': AB})
    return rulesDf


# In[8]:


def rules_visualization(rules,rules_num):
    graph = nx.DiGraph()
    color_map=[]
    N = 50
    colors = np.random.rand(N)    
    strs=['R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11'] 
    for i in range (rules_num):      
        graph.add_nodes_from(["R"+str(i)])
        graph.add_nodes_from([rules.iloc[i]['A']])
        graph.add_edge(rules.iloc[i]['A'], "R"+str(i), color=colors[i] , weight = 2)
        graph.add_nodes_from([rules.iloc[i]['B']])
        graph.add_edge("R"+str(i), rules.iloc[i]['B'], color=colors[i],  weight=2)
    for node in graph:
        found_a_string = False
        for item in strs: 
            if node==item:
                found_a_string = True
        if found_a_string:
            color_map.append('yellow')
        else:
            color_map.append('green')       
    edges = graph.edges()
    colors = [graph[u][v]['color'] for u,v in edges]
    weights = [graph[u][v]['weight'] for u,v in edges]
    pos = nx.spring_layout(graph, k=16, scale=1)
    nx.draw(graph, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=16, with_labels=False)            
    for p in pos:  # raise text positions
           pos[p][1] += 0.07
    nx.draw_networkx_labels(graph, pos)
    plt.show()


# In[9]:


read_data()
order_products=get_priorDf()
# data_exploary(order_products)
products_id_to_name = {k: v for k, v in zip(products.product_id, products.product_name)}
start = datetime.now()
rules = association_rules(order_products, min_support = 0.002,max_length=3)
print('Execution time: ', datetime.now() - start)
rules_visualization(rules, 11)
rules.head(11)

