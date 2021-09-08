# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 19:01:42 2021

@author: peptr
"""

from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.neighbors import BallTree
from pyproj import Proj, transform
from math import radians
from pandas.io.json import json_normalize
import json

def rectify_code_commune(x) : 
    return int(x)  + 100


def merge_datas_communes(df_test,department = None):
    
    
    df_train = pd.read_csv('./Datas/Processed_data/{}.csv'.format(df_test.code_departement.astype(int).values[0]))
    
    if department == 75 : 
        df_train.code_commune = df_train.code_postal.apply(rectify_code_commune)
    
    if df_test.code_commune.values[0] not in df_train.code_commune.unique():
        return df_test
    
    df_communes = pd.DataFrame(df_train[['year','quarter']][df_train.code_commune == df_test.code_commune.values[0]])

    df_communes['nb_transac_quarter'] = 1
    df_communes = df_communes.groupby(by = ['year','quarter']).sum().reset_index()
    df_communes['evol_quarter'] = 0
    
    
    serie = df_communes.nb_transac_quarter
    diff = serie.diff()
    percentage = (diff / serie.shift(1) * 100).fillna(0)
    df_communes.evol_quarter = percentage
    
    test = df_test.merge(df_communes,on= ['year','quarter'])
    
    if len(test) == 0 :
        print("No transaction during this quarter... We take the last evolution instead.")
        year  = max(df_communes.year.values)
        df_year = df_communes[df_communes.year == year]
        quarter = max(df_year.quarter.values)
        df_test['nb_transac_quarter'] = df_year[df_year.quarter == quarter].nb_transac_quarter.values
        df_test['evol_quarter'] = df_year[df_year.quarter == quarter].evol_quarter.values
    else : 
        df_test = test.copy()
        
    
    return df_test



def create_dataset_initial(adresse,code_postal,type_local,surface_reelle_bati,nombre_pieces_principales,surface_terrain,department):
    
    ref = pd.read_csv("./Datas/Raw_data/DIVERS_datasets/communes-departement-region.csv")
    geolocator = Nominatim(user_agent = 'thomas_b')
    
    location = geolocator.geocode(adresse)
    
    print(location)
    
    if location is None : 
        print("Erreur d'adresse")
    
    
    
    df = pd.DataFrame(pd.Series(radians(location.latitude)),columns=['latitude'])
    
    df['latitude'] = pd.Series(radians(location.latitude))
    

    df['longitude'] =  radians(location.longitude)
    
    df['surface_reelle_bati'] = surface_reelle_bati
    df['surface_terrain'] = surface_terrain
    df['nombre_pieces_principales'] = nombre_pieces_principales
    
    df['nature_mutation'] = 'Vente'
    
    df['year'] = 2019
    df['quarter'] = 1
    df['month'] = 1
    
    df['type_local_str'] = type_local
    df['code_postal'] = code_postal
    
    
    infos = ref.loc[ref.code_postal == code_postal]

    df['code_departement'] = int(infos.code_departement.unique()[0])
    df['code_commune'] = int(infos.code_commune_INSEE.unique()[0])
    
    if department in [75,'lyon','marseille'] : 
        arr = int(str(code_postal)[-2:])
        df['arrondissement'] = arr
    
    return df

    
    
def add_datas_NN(df_test,department):
    df_train = pd.read_csv('./Datas/Processed_data/{}.csv'.format(df_test
                                                                   .code_departement.astype(int).values[0]))
    
    df_train.dropna(inplace=True)
    df_train.reset_index(drop=True,inplace=True)
    
    
    if department == 75 : 
        df_train.code_commune = df_train.code_postal.apply(rectify_code_commune)
    
    
    df_test['type_local'] = df_train[df_train.type_local_str == df_test.type_local_str.values[0]].type_local
    
    datas = df_train[df_train.code_commune == df_test.code_commune.values[0]]
    no_com = False
    
    if len(datas) == 0:
        print("Commune jamais vue, l'estimation peut être mauvaise...")
        no_com = True

    k = 30

    model = BallTree(df_train[['latitude', 'longitude']].values, metric='haversine')
    dist, indices = model.query(df_test[['latitude', 'longitude']].values,k)

    print("Début du calcul des habitations les plus proches, calcul du prix du voisinage...")


    df_neigh = df_train[df_train.index.isin(indices[0])].reset_index(drop=True)

    df_neigh_type = df_neigh[(df_neigh.type_local == df_test.type_local.values[0]) & (df_neigh.code_commune == df_test.code_commune.values[0])]

    if len(df_neigh_type) >= 5: 
        df_neigh = df_neigh_type.reset_index(drop=True)

    n = min(len(df_neigh),5)

    dm = np.argsort(distance_matrix(df_test[['surface_reelle_bati','surface_terrain','nombre_pieces_principales']].values,
                                        df_neigh[['surface_reelle_bati','surface_terrain','nombre_pieces_principales']].values,p=2))[0][:n]
    df_test['voisinage'] = df_neigh[df_neigh.index.isin(dm)].Prix_m2_bati.mean()
    df_test['voisinage_total'] = df_neigh[df_neigh.index.isin(dm)].Prix_m2_total.mean()


    df_test['mean_dist_5_NN'] = np.mean(dist[0][:5])
    df_test['mean_dist_10_NN'] = np.mean(dist[0][:10])
    df_test['mean_dist_20_NN'] = np.mean(dist[0][:20])
    df_test['mean_dist_25_NN'] = np.mean(dist[0][:25])
    df_test['mean_dist_50_NN'] = np.mean(dist[0][:50])
    df_test["mean_dist_75_NN"]= np.mean(dist[0][:75])
    df_test['mean_dist_100_NN'] = np.mean(dist[0][:100])

    df_test['std_dist_5_NN'] = np.std(dist[0][:5])
    df_test['std_dist_10_NN'] = np.std(dist[0][:10])
    df_test['std_dist_20_NN'] = np.std(dist[0][:20])
    df_test['std_dist_25_NN'] = np.std(dist[0][:25])
    df_test['std_dist_50_NN'] = np.std(dist[0][:50])
    df_test['std_dist_75_NN'] = np.std(dist[0][:75])
    df_test['std_dist_100_NN'] = np.std(dist[0][:100])
    
    # Nearest Neighbor
    NN = df_train[df_train.index == indices[0][0]]
    
    type_str = df_neigh[df_neigh.type_local_str.values == df_test.type_local_str.values[0]]
    if len(type_str) == 0 :
        df_test['type_local'] = df_neigh.type_local.unique()[0]
    else : 
        df_test['type_local'] = df_neigh.type_local.unique()[0]
    
    
    # All of those informations are stricly the same than the neighrest neighbor 
    
    if department not in [75,'lyon','marseille'] : 
        df_test[['nb_transac_total',
     'Nb de pers. non scolarisées de 15 ans ou + 2017',
     'Part des non ou peu diplômés dans la pop. non scolarisée de 15 ans ou + 2017',
     'Part des pers., dont le diplôme le plus élevé est le bepc ou le brevet, dans la pop. non scolarisée de 15 ans ou + 2017',
     'Part des pers., dont le diplôme le plus élevé est un CAP ou un BEP, dans la pop. non scolarisée de 15 ans ou + 2017',
     'Part des pers., dont le diplôme le plus élevé est le bac, dans la pop. non scolarisée de 15 ans ou + 2017',
     "Part des diplômés d'un BAC+2 dans la pop. non scolarisée de 15 ans ou + 2017",
     "Part des diplômés d'un BAC+3  ou BAC+4 dans la pop. non scolarisée de 15 ans ou + 2017",
     "Taux d'activité par tranche d'âge 2017",
     "Nb d'emplois au lieu de travail (LT) 2017",
     'Part des emplois sal. dans le nb d’emplois au LT 2017',
     'Part des agriculteurs expl. dans le nb d’emplois au LT 2017',
     'Part des artisans, commerçants, chefs d’ent. dans le nb d’emplois au LT 2017',
     'Part des cadres et prof. intellectuelles sup. dans le nb d’emplois au LT 2017',
     'Part des prof. intermédiaires dans le nb d’emplois au LT 2017',
     'Part des employés dans le nb d’emplois au LT 2017',
     'Part des ouvriers dans le nb d’emplois au LT 2017',
     "Taux d'activité par tranche d'âge 2017.1",
     "Taux d'activité par tranche d'âge 2017.2",
     'Résidences principales 2017',
     'Part des logements vacants dans le total des logements 2017',
     'Part des rés. principales dans le total des logements 2017',
     'Part des rés. secondaires (y compris les logements occasionnels) dans le total des logements 2017',
     'Part des locataires dans les rés. principales 2017',
     'Part des locataires HLM dans les rés. principales 2017',
     'Part des rés. principales construites avant 1946 2017',
     'Police - Gendarmerie 2019',
     'Pôle emploi 2019',
     'Banques 2019',
     'Point de contact postal 2019',
     'Hypermarché - Supermarché 2019',
     'Supérette - Épicerie 2019',
     'Boulangerie 2019',
     'Boucherie charcuterie 2019',
     'Poissonnerie 2019',
     'École maternelle 2019',
     'École élémentaire 2019',
     'Collège 2019',
     'Lycée 2019',
     "Service d'urgences 2019",
     'Médecin généraliste 2019',
     'Chirurgien dentiste 2019',
     'Infirmier 2019',
     'Masseur kinésithérapeute 2019',
     'Pharmacie 2019',
     'Hébergement des personnes âgées 2019']] = NN[['nb_transac_total',
     'Nb de pers. non scolarisées de 15 ans ou + 2017',
     'Part des non ou peu diplômés dans la pop. non scolarisée de 15 ans ou + 2017',
     'Part des pers., dont le diplôme le plus élevé est le bepc ou le brevet, dans la pop. non scolarisée de 15 ans ou + 2017',
     'Part des pers., dont le diplôme le plus élevé est un CAP ou un BEP, dans la pop. non scolarisée de 15 ans ou + 2017',
     'Part des pers., dont le diplôme le plus élevé est le bac, dans la pop. non scolarisée de 15 ans ou + 2017',
     "Part des diplômés d'un BAC+2 dans la pop. non scolarisée de 15 ans ou + 2017",
     "Part des diplômés d'un BAC+3  ou BAC+4 dans la pop. non scolarisée de 15 ans ou + 2017",
     "Taux d'activité par tranche d'âge 2017",
     "Nb d'emplois au lieu de travail (LT) 2017",
     'Part des emplois sal. dans le nb d’emplois au LT 2017',
     'Part des agriculteurs expl. dans le nb d’emplois au LT 2017',
     'Part des artisans, commerçants, chefs d’ent. dans le nb d’emplois au LT 2017',
     'Part des cadres et prof. intellectuelles sup. dans le nb d’emplois au LT 2017',
     'Part des prof. intermédiaires dans le nb d’emplois au LT 2017',
     'Part des employés dans le nb d’emplois au LT 2017',
     'Part des ouvriers dans le nb d’emplois au LT 2017',
     "Taux d'activité par tranche d'âge 2017.1",
     "Taux d'activité par tranche d'âge 2017.2",
     'Résidences principales 2017',
     'Part des logements vacants dans le total des logements 2017',
     'Part des rés. principales dans le total des logements 2017',
     'Part des rés. secondaires (y compris les logements occasionnels) dans le total des logements 2017',
     'Part des locataires dans les rés. principales 2017',
     'Part des locataires HLM dans les rés. principales 2017',
     'Part des rés. principales construites avant 1946 2017',
     'Police - Gendarmerie 2019',
     'Pôle emploi 2019',
     'Banques 2019',
     'Point de contact postal 2019',
     'Hypermarché - Supermarché 2019',
     'Supérette - Épicerie 2019',
     'Boulangerie 2019',
     'Boucherie charcuterie 2019',
     'Poissonnerie 2019',
     'École maternelle 2019',
     'École élémentaire 2019',
     'Collège 2019',
     'Lycée 2019',
     "Service d'urgences 2019",
     'Médecin généraliste 2019',
     'Chirurgien dentiste 2019',
     'Infirmier 2019',
     'Masseur kinésithérapeute 2019',
     'Pharmacie 2019',
     'Hébergement des personnes âgées 2019']].values
                                                
                                                
    if no_com : 
        df_evol = df_neigh[df_neigh.year == max(df_neigh.year)]
        df_evol = df_evol[df_evol.quarter == max(df_evol.year)]
        
        
        df_test['nb_transac_quarter'] = df_evol.nb_transac_quarter.values[0]
        df_test['evol_quarter']= df_evol.evol_quarter.values[0]
        
    else : 
        df_test = merge_datas_communes(df_test)
    
    return df_test

    
def add_distances_education(df,evaluation=False):
    
    if evaluation:
        path = './Datas/Raw_data/BPE_datasets/bpe19_enseignement_xy.csv'
    else : 
        path = '../Datas/Raw_data/BPE_datasets/bpe19_enseignement_xy.csv'
    
    df_education = pd.read_csv(path,delimiter=';',usecols=['SECT','CL_PGE','LAMBERT_X','LAMBERT_Y','TYPEQU'])
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



def add_distances_commerces(df,evaluation=False):
    
    if evaluation:
        path = './Datas/Raw_data/BPE_datasets/bpe19_ensemble_xy.csv'
    else : 
        path = '../Datas/Raw_data/BPE_datasets/bpe19_ensemble_xy.csv'
    
    df_all = pd.read_csv(path,delimiter=';',usecols=['LAMBERT_X','LAMBERT_Y','TYPEQU'])
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



def add_distance_transports(df,evaluation=False):
    
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
    
    if not evaluation : 
        path = '../Datas/Raw_data/DIVERS_datasets/emplacement-des-gares-idf.csv'
    else : 
        path = './Datas/Raw_data/DIVERS_datasets/emplacement-des-gares-idf.csv'
    
    df_metro = pd.read_csv(path,delimiter=';',usecols = 
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
    
    if not evaluation : 
        path = '../Datas/Raw_data/DIVERS_datasets/arrets-par-lignes-de-transport-en-commun-en-ile-de-france.csv'
    else : 
        path = './Datas/Raw_data/DIVERS_datasets/arrets-par-lignes-de-transport-en-commun-en-ile-de-france.csv'

    df_bus = pd.read_csv(path,delimiter=';',usecols = 
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
    
    
    
def add_specific_datas(df,department) : 
    
    if department == 'lyon' : 
    
        print("Adding info about public transportation...")
        
        with open('./Datas/Raw_data/lyon_datasets/tcl_sytral.json') as f:
            data = json_normalize(json.load(f)['features'])
        data['geometry.coordinates'] = data['geometry.coordinates'].astype(str).str.split('[',expand=True)[1].str.split(']',expand=True)[0]
        df_transports = data['geometry.coordinates'].str.split(',',expand=True).rename(columns = {0:'longitude',1:'latitude'})
        df_transports.longitude = df_transports.longitude.astype(float).apply(radians)
        df_transports.latitude = df_transports.latitude.astype(float).apply(radians)
        
        
        
        model = BallTree(df_transports[['latitude', 'longitude']].values, metric='haversine')
        dist, indices = model.query(df[['latitude', 'longitude']].values,10)
        dist *= 6371
        
    
        df['dist_min_tc'] = dist[:,0]
        df['dist_mean_5_tc'] = np.mean(dist[:,:5],axis=1)
        df['dist_mean_10_tc'] = np.mean(dist[:,:10],axis=1)
        
        df['dist_std_5_tc'] = np.std(dist[:,:5],axis=1)
        df['dist_std_10_tc'] = np.std(dist[:,:10],axis=1)
        
        
    
        print("Now, adding economical datas about employement and demography per arrondissement...")
        # Adding pauverty for each arrondissement
        
        df_arr = pd.read_excel('./Datas/Raw_data/FILOSOFI_datasets/pauvrete_2018.xlsx')
        df_arr = df_arr[df_arr['Code géographique'].isin(df.code_commune.astype(str))].reset_index(drop=True).drop(
                        columns = ['Libellé géographique']).rename(columns = {'Code géographique' : 'code_commune'})
        
        df.code_commune = df.code_commune.astype(str)
        df = df.merge(df_arr,on='code_commune',how='left')

        print("Done.")
        
    elif department == 'marseille' : 
    
        df_plages = pd.read_csv('./Datas/Raw_data/Marseille_datasets/marseille_bases_nautiques_plages_2018_vsohc0e.csv',
                    usecols=['Nom du site','Longitude','Latitude','Categorie'])
    
        df_plages.Latitude = df_plages.Latitude.apply(radians)
        df_plages.Longitude = df_plages.Longitude.apply(radians)
        
        plages = df_plages[df_plages.Categorie == 'Plages'].reset_index(drop=True)
        
        
    
        print("Calcul des plages les plus proches...")
        
        model = BallTree(plages[['Latitude', 'Longitude']].values, metric='haversine')
        dist, indices = model.query(df[['latitude', 'longitude']].values,5)
        dist *= 6371
        
    
        df['dist_min_plage'] = dist[:,0]
        df['mean_dist_2_plages'] = np.mean(dist[:,:2],axis=1)
        df['mean_dist_3_plages'] = np.mean(dist[:,:3],axis=1)
        df['mean_dist_5_plages'] = np.mean(dist[:,:5],axis=1)
        
        df['nom_plage_plus_proche'] = plages.loc[indices[:,0].squeeze(),'Nom du site'].values
        
        
        bn = df_plages[df_plages.Categorie == 'Bases nautiques'].reset_index(drop=True)
        
        
    
        print("Calcul de la base nautique la plus proche...")
        
        
        model = BallTree(bn[['Latitude', 'Longitude']].values, metric='haversine')
        dist, indices = model.query(df[['latitude', 'longitude']].values,1)
        dist *= 6371
        
    
        df['dist_min_bn'] = dist[:,0]
        
        df_parcs = pd.read_csv('./Datas/Raw_data/Marseille_datasets/marseille_parcs_jardins_2018.csv',
                        delimiter = '\t',usecols=['Nom du site','Longitude','Latitude'])
        
        df_parcs.Latitude = df_parcs.Latitude.apply(radians)
        df_parcs.Longitude = df_parcs.Longitude.apply(radians)
        
        
        print("Calcul de l'espace vert le plus proche")
        
        
        model = BallTree(df_parcs[['Latitude', 'Longitude']].values, metric='haversine')
        dist, indices = model.query(df[['latitude', 'longitude']].values,5)
        dist *= 6371
        
        df['dist_min_espace_vert'] = dist[:,0]
        df['mean_dist_5_espaces_verts'] = np.mean(dist,axis=1)
        
        
        df_monum = pd.read_csv('./Datas/Raw_data/Marseille_datasets/marseille_monuments_historiques_2018.csv',
                           delimiter = ';',usecols=['Longitude','Latitude']).dropna(subset=['Longitude','Latitude'])
        
        df_monum.Latitude = df_monum.Latitude.apply(radians)
        df_monum.Longitude = df_monum.Longitude.apply(radians)
        
        print("Calcul du monument le plus proche")
        
        
        model = BallTree(df_monum[['Latitude', 'Longitude']].values, metric='haversine')
        dist, indices = model.query(df[['latitude', 'longitude']].values,5)
        dist *= 6371
        
        df['dist_min_monument_historique'] = dist[:,0]
        df['mean_dist_5_monuments_historiques'] = np.mean(dist,axis=1)
        
        
        df_culture = pd.read_csv('./Datas/Raw_data/Marseille_datasets/marseille_lieux_culturels_2018_jrvozrd.csv',
                           delimiter = '\t',usecols=['Longitude','Latitude']).dropna(subset=['Longitude','Latitude'])
        
        df_culture.Latitude = df_culture.Latitude.apply(radians)
        df_culture.Longitude = df_culture.Longitude.apply(radians)
        
        
        print("Calcul du lieu culturel le plus proche")
        
        
        model = BallTree(df_culture[['Latitude', 'Longitude']].values, metric='haversine')
        dist, indices = model.query(df[['latitude', 'longitude']].values,5)
        dist *= 6371
        
        df['dist_min_culture'] = dist[:,0]
        df['mean_dist_5_lieux_culturels'] = np.mean(dist,axis=1)
        
        print("Now, adding economical datas about employement and demography per arrondissement...")
        # Adding pauverty for each arrondissement
        
        df_arr = pd.read_excel('./Datas/Raw_data/FILOSOFI_datasets/pauvrete_2018.xlsx')
        df_arr = df_arr[df_arr['Code géographique'].isin(df.code_commune.astype(str))].reset_index(drop=True).drop(
                        columns = ['Libellé géographique']).rename(columns = {'Code géographique' : 'code_commune'})
        
        df.code_commune = df.code_commune.astype(str)
        df = df.merge(df_arr,on='code_commune',how='left')

        print("Done.")
        
    elif department == 75 : 
        print("Now, adding economical datas about employement and demography per arrondissement...")
    
        df_arr = pd.read_excel('./Datas/Raw_data/FILOSOFI_datasets/pauvrete_2018.xlsx')
        df_arr = df_arr[df_arr['Code géographique'].isin(df.code_commune.astype(str))].reset_index(drop=True).drop(
                        columns = ['Libellé géographique']).rename(columns = {'Code géographique' : 'code_commune'})
        
        df.code_commune = df.code_commune.astype(str)
        df = df.merge(df_arr,on='code_commune',how='left')
        
        print("Done.")
        
        
    
    return df
        
    
