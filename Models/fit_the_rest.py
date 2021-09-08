# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:01:47 2021

@author: peptr
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import pickle
import os


dep  = [i+1 for i in range(95)]
dom = [971, 972, 973, 974]

target = dep + dom

def create_network():
    model = Sequential()
    model.add(Dense(16, activation='relu',input_shape = (X_keras.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(32, activation='selu'))
    model.add(Dense(16, activation='selu'))
    model.add(Dense(8, activation='selu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer = 'adam', loss = 'mean_absolute_error')

    return model



for department in target : 
    
    try : 
    
        if not os.path.exists('./models/Pipeline_LGBM_{}.sav'.format(department)) : 
            print("-"*80)
            print("Fitting keras and LGBM for department {}".format(department))
            print("-"*80)
            
            df = pd.read_csv('../Datas/Processed_data/{}.csv'.format(department))
            df.dropna(inplace=True)
            df = df[(df.Prix_m2_bati < 10000)].reset_index(drop=True)
            
            categorical_features = ['nombre_pieces_principales','type_local','year','quarter']
            quantitative_features = [
                                     'surface_reelle_bati',
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
            
            
            features = quantitative_features + categorical_features
            
    
    
            X = df[features]
            Y = df.Prix_m2_bati
            
            preprocessor = ColumnTransformer(
        [
            ("preprocessor_cat",OneHotEncoder(handle_unknown = 'ignore'),categorical_features),
            ("preprocessor_quant",StandardScaler(),quantitative_features)
            
        ], remainder = 'drop')
            
            
            print("Fitting LGBM ...")
            
            
            MODEL = Pipeline(
        [
            ("preprocessor",preprocessor),
            ("regressor",LGBMRegressor(learning_rate=0.2,n_estimators=1000))
        ])
            MODEL.fit(X,Y)
            X_keras = preprocessor.transform(X)
            callback = EarlyStopping(patience=15,restore_best_weights = True)
            
            print("Fitting Neural Network ...")
            
            
            keras_regressor = KerasRegressor(build_fn=create_network, epochs=1500,verbose=1,batch_size = 128)
            keras_regressor.fit(X_keras,Y,callbacks=[callback],validation_split=0.2)
            
            print("Saving models ...")
            
            
            filename = './models/Pipeline_LGBM_{}.sav'.format(department)
            pickle.dump(MODEL, open(filename, 'wb'))
            
            filename = './models/preprocessor_keras_{}.sav'.format(department)
            pickle.dump(preprocessor, open(filename, 'wb'))
            
            filename = './models/keras_{}.h5'.format(department)
            keras_regressor.model.save(filename)
            
            print("Done.")
            
            
            
            
    except : 
        print("Problem with department {}".format(department))
            
        
        
        
        
        

        
        
categorical_features = ['nombre_pieces_principales','type_local','year','quarter']
quantitative_features = [
                         'surface_reelle_bati',
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


features = quantitative_features + categorical_features


with open('./models/features_others.txt', "wb") as fp:  
    pickle.dump([categorical_features + quantitative_features], fp)
        
        
        
    
    













