# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:53:12 2021

@author: peptr
"""


import geopy as geopy
import pickle


from geopy.geocoders import Nominatim
from add_datas import create_dataset_initial,merge_datas_communes,add_datas_NN,add_distances_education,add_distances_commerces



adresse = 'Ecole polytechnique, 91120 Palaiseau'
code_postal = 91120

surface_reelle_bati = 110
nombre_pieces_principales = 5
surface_terrain = 1100 
type_local = 'Appartement'



print('------------------------')
print()
print('Ajout des données...')
print()
print('------------------------')

df = create_dataset_initial(adresse,code_postal,type_local,surface_reelle_bati,nombre_pieces_principales,surface_terrain)
df = merge_datas_communes(df)
df = add_datas_NN(df)
df = add_distances_education(df)
df = add_distances_commerces(df)

model = pickle.load(open('./Models_departements/91_catboost.sav', 'rb'))




quantitative_features =  ['surface_reelle_bati',
 'surface_terrain',
 'nb_transac_quarter',
 'evol_quarter',
 'nb_transac_total',
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
 'Hébergement des personnes âgées 2019',
 'dist_min_lycee',
 'dist_min_5_lycee',
 'dist_min_maternelle',
 'dist_min_ecoles_elementaires',
 'dist_min_colleges',
 'dist_min_5_colleges',
 'dist_min_sup',
 'dist_min_5_sup',
 'dist_min_supermarche',
 'dist_min_boulangerie',
 'dist_min_boucherie',
 'dist_min_poste',
 'dist_min_pharmacie',
 'dist_min_aeroport',
 'dist_min_culture',
 'dist_min_generaliste',
 'dist_min_banque',
 'dist_min_urgence',
 'dist_min_specialiste',
 'dist_min_dent',
 'dist_min_infirimier',
 'dist_min_creche',
 'dist_min_gare',
 'voisinage',
 'voisinage_total',
 'mean_dist_5_NN',
 'mean_dist_10_NN',
 'mean_dist_20_NN',
 'mean_dist_25_NN',
 'mean_dist_50_NN',
 'mean_dist_75_NN',
 'mean_dist_100_NN',
 'std_dist_5_NN',
 'std_dist_10_NN',
 'std_dist_20_NN',
 'std_dist_25_NN',
 'std_dist_50_NN',
 'std_dist_75_NN',
 'std_dist_100_NN']


categorical_features = ['nature_mutation','nombre_pieces_principales','type_local_str','year','type_plus_proche_lycee']


df = df[categorical_features + quantitative_features]


print('------------------------')
print()
print('Done..')
print("Prediction de valeur foncière : {} euros.".format(model.predict(df)[0]))
print('------------------------')
print()






