# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 22:25:28 2021

@author: peptr
"""
import os
from Preprocess import get_filter_datas,merge_datas_communes,add_distances_education,add_distances_commerces,compute_NN_price, add_distance_transports


df_ini = get_filter_datas(['2014','2015','2016','2017','2018','2019','2020'])

idf =  [91,92,93,77,78,94,95]
specific_models = [75,13,69]

#for dep in df_ini.code_departement.unique() :
for dep in df_ini.code_departement.unique():
    
    if not os.path.exists('../Datas/Processed_data/{}.csv'.format(dep)) :
    
        try :
            dep = int(dep)
            
            if dep not in specific_models :
        
                df = df_ini[df_ini.code_departement == dep]
                df.reset_index(drop=True,inplace=True)
        
                df = merge_datas_communes(df)
                df = add_distances_education(df)
                df = add_distances_commerces(df)
                
                
                df = compute_NN_price(df)
                
                if dep in idf : 
                    print("transports")
                    df = add_distance_transports(df)
                
                
            if dep == 75 :
                
                df = df_ini[df_ini.code_departement == dep]
                df.reset_index(drop=True,inplace=True)
                df = df[(df.Prix_m2_bati < 30000) & (df.Prix_m2_bati > 4000)].reset_index(drop=True)
                
                df['arrondissement'] = df['code_commune'].astype('str').str[-2:]
                df.code_commune = 75056 #real INSEE code
        
                df = add_distances_education(df)
                df = add_distances_commerces(df)
                
                
                df = compute_NN_price(df)
                df = add_distance_transports(df)
                
            if dep == 13 :
                
                df = df_ini[df_ini.code_departement == dep]
                
                df_marseille = df[(df.code_commune >= 13201)&(df.code_commune < 13217)]
                df = df[~df.index.isin(df_marseille.index)].reset_index(drop=True)
                df_marseille.reset_index(drop=True,inplace=True)
                df_marseille['arrondissement'] = df_marseille['code_commune'].astype('str').str[-2:]
                df_marseille = add_distances_education(df_marseille)
                df_marseille = add_distances_commerces(df_marseille)
                
                
                df_marseille = compute_NN_price(df_marseille)
                df_marseille.to_csv('../Datas/Processed_data/marseille.csv',index=False)
                
                
                
                df = merge_datas_communes(df)
                df = add_distances_education(df)
                df = add_distances_commerces(df)
                
                
                df = compute_NN_price(df)
                
            if dep == 69 : 
                
                
                df = df_ini[df_ini.code_departement == dep]
                
                df_lyon = df[(df.code_commune >= 69381)&(df.code_commune <= 69389)]
                df = df[~df.index.isin(df_lyon.index)].reset_index(drop=True)
                df_lyon.reset_index(drop=True,inplace=True)
                df_lyon['arrondissement'] = df_lyon['code_commune'].astype('str').str[-2:]
                df_lyon = add_distances_education(df_lyon)
                df_lyon = add_distances_commerces(df_lyon)
                
                
                df_lyon = compute_NN_price(df_lyon)
                df_lyon.to_csv('../Datas/Processed_data/lyon.csv',index=False)
                
                
                
                df = merge_datas_communes(df)
                df = add_distances_education(df)
                df = add_distances_commerces(df)
                
                
                df = compute_NN_price(df)
                
                    
            df.to_csv('../Datas/Processed_data/{}.csv'.format(dep),index=False)
            
            
        
        except : 
            print("Problème avec le département {}".format(dep))
        