# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 20:57:45 2021

@author: peptr

Fonctions utilisées pour réaliser le pré-process des données / merge tous les 
datasets externes. 

Output : dataset utilisé par le modèle prédictif

"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree
from pyproj import Proj, transform
from math import radians 
from sklearn.preprocessing import OrdinalEncoder



def get_filter_datas(year) : 

    if type(year) == int : 
        year = [year]
    
    to_stack = []
    
    for i in range(len(year)) : 
        print("Year {}".format(year[i]))
        df = pd.read_csv('../Datas/Raw_data/DVF_datasets/{}.csv'.format(year[i]))
        print("Original number of lines : {}, {} features".format(len(df),len(df.columns)))
        df = df[np.logical_or(df.nature_mutation == 'Vente',df.nature_mutation == "Vente en l'état futur d'achèvement")]
        df = df[np.logical_or(df.type_local == 'Appartement',df.type_local=='Maison')]
        df = df[pd.notnull(df.surface_reelle_bati)]
        df = df[pd.notnull(df.valeur_fonciere)]
        df.surface_terrain.fillna(0,inplace=True)
        df.nature_culture.fillna('sols',inplace=True)


        type_à_garder = ["terrains d'agrément",'jardins','sols',np.nan]
        to_delete = df[~df.nature_culture.isin(type_à_garder)].id_mutation.unique()
        df = df.drop(df[df.id_mutation.isin(to_delete)].index)

        # On considère par défault que le terrain lié à un appartement est nul
        df.loc[df.type_local == 'Appartement','surface_terrain'] = 0


        # Tri de certains outlier, peut être modifié, solution de facilité ici :

        df['detect_outlier'] = 1
        df_2 = df[['detect_outlier','id_mutation']].groupby('id_mutation').sum().reset_index()


        to_treat = df_2[df_2.detect_outlier >= 2]
        df_tt = df[df.id_mutation.isin(to_treat.id_mutation)]
        df_2 = df_2[df_2.detect_outlier < 2]

        print("Removing outliers, shape before : {}".format(len(df)))
        df = df[df.id_mutation.isin(df_2.id_mutation)]
        print("Done. Shape after : {}".format(len(df)))

# =============================================================================
#         #test to recover datas, bad in practice, recommand to let that as it is
#
#         df_tt.loc[df_tt.nature_culture == 'jardins','surface_reelle_bati'] = 0
#         df_tt.loc[df_tt.nature_culture == "terrains d'agrément",'surface_reelle_bati'] = 0
#         df_tt.drop(df_tt[(df_tt.type_local == 'Appartement') & (df_tt.nature_culture == "terrains d'agrément")].index,inplace = True)
#         df_tt.loc[df_tt.type_local == 'Appartement','surface_terrain'] = 0
#         total = df_tt[['id_mutation','surface_terrain','surface_reelle_bati']].groupby('id_mutation').sum().rename(
#                     columns = {'surface_terrain' : 'surface_terrain_totale',
#                               'surface_reelle_bati' : 'surface_reelle_bati_totale'})
# 
#         new_df = df_tt.drop(df_tt[df_tt.nature_culture != 'sols'].index).merge(total,on='id_mutation',how='left')
#         new_df.valeur_fonciere = new_df.valeur_fonciere *(new_df.surface_reelle_bati + new_df.surface_terrain)/(new_df.surface_reelle_bati_totale + new_df.surface_terrain_totale)
#         new_df.drop_duplicates(subset=['id_mutation','longitude','latitude'],inplace=True)
# 
#         df = pd.concat([df,new_df[df.columns]]).reset_index(drop=True)
# 
#         print("Shape after data recuperation : {}".format(len(df)))
# =============================================================================

        df['type_local_str'] = df.type_local.copy()
        df.type_local = OrdinalEncoder().fit_transform(df.type_local.values.reshape(-1,1))

        df.date_mutation = pd.to_datetime(df.date_mutation)

        df['year'] = df.date_mutation.dt.year
        df['quarter'] = df.date_mutation.dt.quarter
        df['month'] = df.date_mutation.dt.month

        df.drop(columns = ['nature_culture','detect_outlier', 'numero_disposition',
           'adresse_numero', 'adresse_suffixe', 'date_mutation',
           'adresse_nom_voie', 'adresse_code_voie',
           'nom_commune', 'ancien_code_commune',
           'ancien_nom_commune', 'id_parcelle', 'ancien_id_parcelle',
           'numero_volume', 'lot1_numero', 'lot1_surface_carrez', 'lot2_numero',
           'lot2_surface_carrez', 'lot3_numero', 'lot3_surface_carrez',
           'lot4_numero', 'lot4_surface_carrez', 'lot5_numero',
           'lot5_surface_carrez', 'nombre_lots', 'code_type_local', 
           'code_nature_culture', 'code_nature_culture_speciale',
           'nature_culture_speciale'],inplace=True)

        df = df[pd.notnull(df.longitude)]

        df['Prix_m2_bati'] = df.valeur_fonciere/df.surface_reelle_bati
        df['Prix_m2_total'] = df.valeur_fonciere/(df.surface_reelle_bati + df.surface_terrain)
        df['nombre_pieces_principales'] = df['nombre_pieces_principales'].astype(int)


        df.latitude = df.latitude.apply(radians)
        df.longitude = df.longitude.apply(radians)

        df.reset_index(inplace=True,drop=True)
        
        print("Final number of lines : {}, {} features".format(len(df),len(df.columns)))
        to_stack.append(df)
        del(df)
        
    return pd.concat(to_stack,ignore_index=True)



def merge_datas_communes(df):
    
    df_communes = pd.DataFrame(df[['code_commune','year','quarter']])
    df_communes.code_commune = df_communes.code_commune.astype(str)
    df_communes['nb_transac_quarter'] = 1
    df_communes = df_communes.groupby(by = ['code_commune','year','quarter']).sum().reset_index()
    df_communes['evol_quarter'] = 0
    
    
    df_communes_2 = df_communes[['code_commune']]
    df_communes_2['nb_transac_total'] = df_communes['nb_transac_quarter']
    df_communes_2 = df_communes_2.groupby(by = ['code_commune']).sum().reset_index()
    df_communes = df_communes.merge(df_communes_2,on=['code_commune'],how='left')
    
    
    for com in df_communes.code_commune.unique() : 
        serie = df_communes[df_communes.code_commune == com].nb_transac_quarter
        diff = serie.diff()
        percentage = (diff / serie.shift(1) * 100).fillna(0)
        df_communes.loc[df_communes.code_commune == com,'evol_quarter'] = percentage
    
    merge = pd.read_csv('../Datas/Raw_data/INSEE_datasets/données_éducation_commune.csv',delimiter=';')
    merge.rename(columns={'Code' : 'code_commune'},inplace=True)
    merge.drop(columns=['Libellé'],inplace=True)
    
    df_merge = df_communes.merge(merge,on = ['code_commune'])
    
    merge = pd.read_csv('../Datas/Raw_data/INSEE_datasets/données_activité_éco_commune.csv',delimiter=';')
    merge.rename(columns={'Code' : 'code_commune'},inplace=True)
    merge.drop(columns=['Libellé'],inplace=True)
    
    df_merge = df_merge.merge(merge,on = ['code_commune'])

    merge = pd.read_csv('../Datas/Raw_data/INSEE_datasets/données_logements_commune.csv',delimiter=';')
    merge.rename(columns={'Code' : 'code_commune'},inplace=True)
    merge.drop(columns=['Libellé'],inplace=True)
    
    df_merge = df_merge.merge(merge,on = ['code_commune'])
    
    
    merge = pd.read_csv('../Datas/Raw_data/INSEE_datasets/données_nb_commerces_commune.csv',delimiter=';')
    merge.rename(columns={'Code' : 'code_commune'},inplace=True)
    merge.code_commune = merge.code_commune.astype(str)
    merge.drop(columns=['Libellé'],inplace=True)
    
    df_merge = df_merge.merge(merge,on = ['code_commune'])
    
    merge = pd.read_csv('../Datas/Raw_data/INSEE_datasets/données_entreprises_commune.csv',delimiter=';').dropna(axis=1)
    merge.rename(columns={'Code' : 'code_commune'},inplace=True)
    merge.code_commune = merge.code_commune.astype(str)
    merge.drop(columns=['Libellé'],inplace=True)
    
    df_merge = df_merge.merge(merge,on = ['code_commune'])
    
    
    
    
    
    df_merge.code_commune = df_merge.code_commune.astype(int)
    #df_merge.rename(columns={'Code' : 'code_commune'},inplace=True)
    
    return df.merge(df_merge,on= ['code_commune','year','quarter'])



def add_distances_education(df):
    
    df_education = pd.read_csv('../Datas/Raw_data/BPE_datasets/bpe19_enseignement_xy.csv',delimiter=';',usecols=['SECT','CL_PGE','LAMBERT_X','LAMBERT_Y','TYPEQU'])
    inProj = Proj('epsg:2154')
    outProj = Proj('epsg:4326')

    df_education.LAMBERT_X,df_education.LAMBERT_Y = transform(inProj,outProj,df_education.LAMBERT_X,df_education.LAMBERT_Y)
    df_education.rename(columns={'LAMBERT_X' : 'latitude','LAMBERT_Y' : 'longitude'},inplace=True) 
    df_education.longitude = df_education.longitude.apply(radians)
    df_education.latitude = df_education.latitude.apply(radians)
    df_education.dropna(inplace=True)

    lycees = df_education[((df_education.TYPEQU == 'C301')|(df_education.TYPEQU == 'C302'))|(df_education.TYPEQU == 'C303')].reset_index(drop=True)

    print("Calcul des lycées les plus proches...")

    model = BallTree(lycees[['latitude', 'longitude']].values, metric='haversine')#, func=distanceMetric)
    dist, indices = model.query(df[['latitude', 'longitude']].values,5)
    dist *= 6371

    df['dist_min_lycee'] = dist[:,0]
    df['type_plus_proche_lycee'] = lycees.loc[indices[:,0]].TYPEQU.reset_index(drop=True)
    df['dist_min_5_lycee'] = np.mean(dist,axis=1)

    print("Calcul des maternelles les plus proches...")

    maternelles = df_education[(df_education.TYPEQU == 'C101') | (df_education.TYPEQU == 'C102')].reset_index(drop=True)


    model = BallTree(maternelles[['latitude', 'longitude']].values, metric='haversine')#, func=distanceMetric)
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_maternelle'] = dist

    print("Calcul des écoles élémentaires les plus proches...")
    ecoles_elementaires = df_education[(df_education.TYPEQU == 'C104')|(df_education.TYPEQU == 'C105')].reset_index(drop=True)


    model = BallTree(ecoles_elementaires[['latitude', 'longitude']].values, metric='haversine')#, func=distanceMetric)
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_ecoles_elementaires'] = dist


    print("Calcul des collèges les plus proches...")

    colleges = df_education[df_education.TYPEQU == 'C201'].reset_index(drop=True)
    model = BallTree(colleges[['latitude', 'longitude']].values, metric='haversine')#, func=distanceMetric)
    dist, indices = model.query(df[['latitude', 'longitude']].values,5)
    dist *= 6371

    df['dist_min_colleges'] = dist[:,0]
    df['dist_min_5_colleges'] = np.mean(dist,axis=1)


    print("Calcul des Etablissements supérieurs les plus proches...")

    sup = df_education[((((df_education.TYPEQU == 'C402')|(df_education.TYPEQU == 'C403'))|((df_education.TYPEQU == 'C409')|(df_education.TYPEQU == 'C501')))|((df_education.TYPEQU == 'C502')|(df_education.TYPEQU == 'C503')))|(df_education.TYPEQU == 'C504')].reset_index(drop=True)
    model = BallTree(sup[['latitude', 'longitude']].values, metric='haversine')#, fnc=distanceMetric)
    dist, indices = model.query(df[['latitude', 'longitude']].values,5)
    dist *= 6371

    df['dist_min_sup'] = dist[:,0]
    df['dist_min_5_sup'] = np.mean(dist,axis=1)
    
    return df



def add_distances_commerces(df):
    
    df_all = pd.read_csv('../Datas/Raw_data/BPE_datasets/bpe19_ensemble_xy.csv',delimiter=';',usecols=['LAMBERT_X','LAMBERT_Y','TYPEQU'])
    inProj = Proj('epsg:2154')
    outProj = Proj('epsg:4326')

    df_all.LAMBERT_X,df_all.LAMBERT_Y = transform(inProj,outProj,df_all.LAMBERT_X,df_all.LAMBERT_Y)
    df_all.rename(columns={'LAMBERT_X' : 'longitude','LAMBERT_Y' : 'latitude'},inplace=True)
    
    df_all.longitude = df_all.latitude.apply(radians)
    df_all.longitude = df_all.latitude.apply(radians)
    
    df_all.dropna(inplace=True)

    supermarches = df_all[(df_all.TYPEQU == 'B101')|(df_all.TYPEQU == 'B102')].reset_index(drop=True)

    print("Calcul du super/hyper marché le plus proche...")
    
    model = BallTree(supermarches[['latitude', 'longitude']].values, metric='haversine')#, func=distanceMetric)
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_supermarche'] = dist
    
    
    print("Calcul de la boulangerie la plus proche...")
    
    boulangeries = df_all[df_all.TYPEQU == 'B203'].reset_index(drop=True)
    
    model = BallTree(boulangeries[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_boulangerie'] = dist
    
    print("Calcul de la boucherie ou poissonerie la plus proche...")
    
    boucheries = df_all[(df_all.TYPEQU == 'B204') | (df_all.TYPEQU == 'B206')].reset_index(drop=True)
    
    model = BallTree(boucheries[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_boucherie'] = dist
    
    
    print("Calcul de la poste la plus proche...")
    
    poste = df_all[((df_all.TYPEQU == 'A206') | (df_all.TYPEQU == 'A207')) | (df_all.TYPEQU == 'A208')].reset_index(drop=True)
    
    model = BallTree(poste[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_poste'] = dist
    
    
    print("Calcul de la pharmacie la plus proche...")
    
    pharmacie = df_all[df_all.TYPEQU == 'D301'].reset_index(drop=True)
    
    model = BallTree(pharmacie[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_pharmacie'] = dist
    
    
    print("Calcul de l'aéroport le plus proche...")
    
    aero = df_all[df_all.TYPEQU == 'E102'].reset_index(drop=True)
    
    model = BallTree(aero[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_aeroport'] = dist
    
    print("Calcul du musée, cinéma ou conservatoire le plus proche...")
    
    culture = df_all[((df_all.TYPEQU == 'F303') | (df_all.TYPEQU == 'F304'))|(df_all.TYPEQU == 'F305')].reset_index(drop=True)
    
    model = BallTree(culture[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_culture'] = dist
    
    
    print("Calcul du généraliste le plus proche...")
    
    gene = df_all[df_all.TYPEQU == 'D201'].reset_index(drop=True)
    
    model = BallTree(gene[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_generaliste'] = dist
    
    
    
    print("Calcul du centre de santé le plus proche...")
    
    sante = df_all[df_all.TYPEQU == 'D108'].reset_index(drop=True)
    
    model = BallTree(sante[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_generaliste'] = dist
    
    
    print("Calcul de la banque la plus proche...")
    
    banque = df_all[df_all.TYPEQU == 'A203'].reset_index(drop=True)
    
    model = BallTree(banque[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_banque'] = dist
    
    
    print("Calcul du service d'urgences le plus proche...")
    
    urgence = df_all[df_all.TYPEQU == 'D106'].reset_index(drop=True)
    
    model = BallTree(urgence[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_urgence'] = dist
    
    
    print("Calcul du spécialiste le plus proche...")
    
    specialiste = df_all[(df_all.TYPEQU == 'D202') |  
                     (df_all.TYPEQU == 'D203') |
                     (df_all.TYPEQU == 'D204') |
                     (df_all.TYPEQU == 'D205') |
                     (df_all.TYPEQU == 'D206') |
                     (df_all.TYPEQU == 'D207') |
                     (df_all.TYPEQU == 'D208') |
                     (df_all.TYPEQU == 'D209') |
                     (df_all.TYPEQU == 'D210') |
                     (df_all.TYPEQU == 'D211') |
                     (df_all.TYPEQU == 'D212') |
                     (df_all.TYPEQU == 'D213') |
                     (df_all.TYPEQU == 'D214') ].reset_index(drop=True)
    
    model = BallTree(specialiste[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_specialiste'] = dist
    
    
    
    print("Calcul du dentise le plus proche...")
    
    dent = df_all[df_all.TYPEQU == 'D221'].reset_index(drop=True)
    
    model = BallTree(dent[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_dent'] = dist
    
    
    print("Calcul du service infirmier le plus proche...")
    
    inf = df_all[df_all.TYPEQU == 'D232'].reset_index(drop=True)
    
    model = BallTree(inf[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_infirimier'] = dist
    
    
    print("Calcul de la crèche la plus proche...")
    
    creche = df_all[df_all.TYPEQU == 'D502'].reset_index(drop=True)
    
    model = BallTree(creche[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_creche'] = dist
    
    
    
    print("Calcul de la gare la plus proche...")
    
    gare = df_all[(df_all.TYPEQU == 'E107')|
                    (df_all.TYPEQU == 'E108')|
                    (df_all.TYPEQU == 'E109')].reset_index(drop=True)
    
    model = BallTree(gare[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    dist *= 6371

    df['dist_min_gare'] = dist  
    
    return df


def add_demographic_evolution(df):
    print("Adding demographic evolution")
    print("Shape before : {}".format(df.shape))
    evol = pd.read_csv('Evol_pop.csv').rename(columns={'code_commune_INSEE' : 'code_commune'})
    df.code_commune = df.code_commune.astype(str)
    df = df.merge(evol,how='left').fillna(0)
    df['evolution_demo'] = df.apply(lambda row : row['mean_3_year_tot_{}'.format(row.year)],axis=1)
    evol.drop(columns=['code_commune'],inplace=True)
    df.drop(columns = evol.columns,inplace=True)
    print("Shape end : {}".format(df.shape))
    return df
    


def compute_NN_price(df):

    estim = []
    estim_total = []

    estim_median = []
    estim_median_total = []

    mean_dist_5_NN = []
    mean_dist_10_NN = []
    mean_dist_20_NN = []
    mean_dist_25_NN = []
    mean_dist_50_NN = []
    mean_dist_75_NN = []
    mean_dist_100_NN = []


    std_dist_5_NN = []
    std_dist_10_NN = []
    std_dist_20_NN = []
    std_dist_25_NN = []
    std_dist_50_NN = []
    std_dist_75_NN = []
    std_dist_100_NN = []

    k = 30


    model = BallTree(df[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,k)

    print("Début du calcul des habitations les plus proches, calcul du prix du voisinage...")

    for element in tqdm(df.index):


        target = df[df.index == element]

        try :
            # Delete it if element in X_train ie answer here
            df_neigh = df[df.index.isin(indices[element])].drop(index=target.index).reset_index(drop=True)
        except :
            df_neigh =df[df.index.isin(indices[element])].reset_index(drop=True)

        df_neigh_type = df_neigh[(df_neigh.type_local == target.type_local.values[0]) & (df_neigh.code_commune == target.code_commune.values[0])]

        if len(df_neigh_type) >= 5: 
            df_neigh = df_neigh_type.reset_index(drop=True)

        n = min(len(df_neigh),5)

        dm = np.argsort(distance_matrix(target[['surface_reelle_bati','surface_terrain','nombre_pieces_principales']].values,
                                        df_neigh[['surface_reelle_bati','surface_terrain','nombre_pieces_principales']].values,p=2))[0][:n]
        estim.append(df_neigh[df_neigh.index.isin(dm)].Prix_m2_bati.mean())
        estim_total.append(df_neigh[df_neigh.index.isin(dm)].Prix_m2_total.mean())

        estim_median.append(df_neigh[df_neigh.index.isin(dm)].Prix_m2_bati.median())
        estim_median_total.append(df_neigh[df_neigh.index.isin(dm)].Prix_m2_total.median())

        mean_dist_5_NN.append(np.mean(dist[element][:5]))
        mean_dist_10_NN.append(np.mean(dist[element][:10]))
        mean_dist_20_NN.append(np.mean(dist[element][:20]))
        mean_dist_25_NN.append(np.mean(dist[element][:25]))
        mean_dist_50_NN.append(np.mean(dist[element][:50]))
        mean_dist_75_NN.append(np.mean(dist[element][:75]))
        mean_dist_100_NN.append(np.mean(dist[element][:100]))

        std_dist_5_NN.append(np.std(dist[element][:5]))
        std_dist_10_NN.append(np.std(dist[element][:10]))
        std_dist_20_NN.append(np.std(dist[element][:20]))
        std_dist_25_NN.append(np.std(dist[element][:25]))
        std_dist_50_NN.append(np.std(dist[element][:50]))
        std_dist_75_NN.append(np.std(dist[element][:75]))
        std_dist_100_NN.append(np.std(dist[element][:100]))


    df['voisinage'] = estim
    df['voisinage_total'] = estim_total

    df['mean_dist_5_NN'] = mean_dist_5_NN
    df['mean_dist_10_NN'] = mean_dist_10_NN
    df['mean_dist_20_NN'] = mean_dist_20_NN
    df['mean_dist_25_NN'] = mean_dist_25_NN
    df['mean_dist_50_NN'] = mean_dist_50_NN
    df['mean_dist_75_NN'] = mean_dist_75_NN
    df['mean_dist_100_NN'] = mean_dist_100_NN

    df['std_dist_5_NN'] = std_dist_5_NN
    df['std_dist_10_NN'] = std_dist_10_NN
    df['std_dist_20_NN'] = std_dist_20_NN
    df['std_dist_25_NN'] = std_dist_25_NN
    df['std_dist_50_NN'] = std_dist_50_NN
    df['std_dist_75_NN'] = std_dist_75_NN
    df['std_dist_100_NN'] = std_dist_100_NN

    print('Correlation prix estimé vs vrai prix au m^2 : {}'.format(df.voisinage.corr(df.Prix_m2_bati)))
    print('Correlation prix surface bâtie + terrain estimé vs vrai prix au m^2 : {}'.format(df.voisinage_total.corr(df.Prix_m2_total)))

    return df




def evaluate_model(y_test,y_pred):
    error= np.abs((y_test-y_pred)/y_test)*100
    print('Mean absolute percentage error = {}'.format(np.mean(error)))
    print('Median absolute percentage error = {}'.format(np.median(error)))
    


def add_distance_transports(df):
    
    """
    Only for Ile de France, ie departements : 
        Paris (75)
        Seine-et-Marne (77)
        Yvelines (78)
        Essonne (91)
        Hauts-de-Seine (92)
        Seine-Saint-Denis (93)
        Val-de-Marne (94) 
    """
    
    df_metro = pd.read_csv('../Datas/Raw_data/DIVERS_datasets/emplacement-des-gares-idf.csv',delimiter=';',usecols = 
                            ['Geo Point','mode_','ligne','cod_ligf'])
    df_metro['latitude'] = df_metro['Geo Point'].str.split(',',expand=True)[0].astype(float).apply(radians)
    df_metro['longitude'] = df_metro['Geo Point'].str.split(',',expand=True)[1].astype(float).apply(radians)

    df_metro.drop(columns=['Geo Point'],inplace=True)

    model = BallTree(df_metro[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,10)
    dist *= 6371
    df['Distance_plus_proche_ferré'] = dist[:,0]
    df['Distance_moyenne_5_plus_proches_ferrés'] = np.mean(dist[:,:5],axis=1)
    df['Distance_moyenne_10_plus_proches_ferrés'] = np.mean(dist,axis=1)
    df['Type_plus_proche_ferré'] = df_metro.loc[indices[:,0]].mode_.reset_index(drop=True)
    df['ligne_plus_proche_ferré'] = df_metro.loc[indices[:,0]].ligne.reset_index(drop=True)

    df_bus = pd.read_csv('../Datas/Raw_data/DIVERS_datasets/arrets-par-lignes-de-transport-en-commun-en-ile-de-france.csv',delimiter=';',usecols = 
                                ['agency_name','stop_lon','stop_lat','route_long_name'])

    df_bus.stop_lon = df_bus.stop_lon.apply(radians)
    df_bus.stop_lat = df_bus.stop_lat.apply(radians)

    nocti = df_bus[df_bus.agency_name == 'Noctilien'].reset_index()


    model = BallTree(nocti[['stop_lat', 'stop_lon']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,1)
    df['Noctilien_plus_proche'] = dist* 6371

    df_bus.route_long_name =  df_bus.route_long_name.apply(lambda s: int(s) if s.isdigit() else np.nan)
    df_bus = df_bus.dropna().reset_index(drop=True)

    model = BallTree(df_bus[['stop_lat', 'stop_lon']].values, metric='haversine')
    dist, indices = model.query(df[['latitude', 'longitude']].values,10)
    dist *= 6371
    df['Distance_plus_proche_bus'] = dist[:,0]
    df['Distance_moyenne_5_plus_proches_bus'] = np.mean(dist[:,:5],axis=1)
    df['Distance_moyenne_10_plus_proches_bus'] = np.mean(dist,axis=1)

    df['std_distance_5_plus_proches_bus'] = np.std(dist[:,:5],axis=1)
    df['std_distance_10_plus_proches_bus'] = np.std(dist,axis=1)
    
    return df