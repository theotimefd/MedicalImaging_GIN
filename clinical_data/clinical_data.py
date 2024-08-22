import os
import numpy as np
import pandas as pd
import csv

directory = "/home/fehrdelt/data_ssd/data/clinical_data/"
MISSING_VALUE = np.nan
#MISSING_VALUE = -1

def get_IPP(row):
    try:
        ipp = int(row[6])
    except:
        return MISSING_VALUE
    
    return ipp

def get_age(row):
    # returns -1 if age can't be computed
    
    if len(row[3])>0 and len(row[8])>0:
        birth = row[3].split('/')[2]
        entry = row[8].split(' ')[0].split('/')[2]
        return int(entry)-int(birth)
    
    else:
        return MISSING_VALUE

    
def get_hemocue(row):
    # hemoglobine
    try:
        hemocue = float(row[23])
    except:
        return MISSING_VALUE
    
    if hemocue < 200: #    <------- vérif
            return hemocue
    else: return MISSING_VALUE


def get_fracas_bassin(row):
    try:
        fracas = int(row[27])
    except:
        return MISSING_VALUE
    
    return fracas


def get_catecholamines(row):
    try:
        catecholamines = int(row[35])
    except:
        return MISSING_VALUE
    
    return catecholamines


def get_PAS(row):
    try:
        PAS = int(row[16])
    except:
        return MISSING_VALUE
    
    if PAS>0:return PAS
    else:return MISSING_VALUE


def get_PAD(row): #pression_arterielle_systolique_PAS_arrivee_du_smur
    try:
        PAD = int(row[17])
    except:
        return MISSING_VALUE
    
    if PAD>0:return PAD
    else:return MISSING_VALUE


def get_glasgow(row): #pression_arterielle_diastolique_PAD_arrivee_du_smur
    try:
        glasgow = int(row[21])
    except:
        return MISSING_VALUE
    
    return glasgow


def get_glasgow_moteur(row):
    try:
        glasgow_moteur = int(row[22])
    except:
        return MISSING_VALUE
    
    return glasgow_moteur


def get_anomalie_pupille(row):
    try:
        anomalie_pupille = int(row[26])
    except:
        return MISSING_VALUE
    
    return anomalie_pupille


def get_freq_cardiaque(row):
    try:
        freq_cardiaque = int(row[18])
    except:
        return MISSING_VALUE
    
    if freq_cardiaque>0: return freq_cardiaque
    else: return MISSING_VALUE


def get_ACR(row): # arret_cardio_respiratoire_massage
    try:
        ACR = int(row[29])
    except:
        return MISSING_VALUE
    
    return ACR


def get_penetrant(row):
    try:
        penetrant = int(row[14])
    except:
        return MISSING_VALUE
    
    return penetrant


def get_ischemie(row): # ischemie_du_membre
    try:
        ischemie = int(row[31])
    except:
        return MISSING_VALUE
    
    return ischemie


def get_hemorragie(row): # hemorragie_externe
    try:
        hemorragie = int(row[30])
    except:
        return MISSING_VALUE
    
    return hemorragie


def get_amputation(row): # hemorragie_externe
    try:
        amputation = int(row[28])
    except:
        return MISSING_VALUE
    
    return amputation

def get_neurochir(row):
    try:
        neurochir = int(row[85])
    except:
        return MISSING_VALUE

    return neurochir

def get_pic(row):
    try:
        pic = int(row[62])
    except:
        return MISSING_VALUE

    return pic

# hemocue = hemoglobine
# catecholamines = vasopresseur
# arret cardio respiratoire = ACR

def get_clinical_data(filepath):
    """
    Returns a pandas DataFrame containing IPP and the relevant clinical data
    """

    df = pd.DataFrame(columns=['IPP', 'age', 'hemocue_initial', 'fracas_du_bassin', 'catecholamines', 
                                    'pression_arterielle_systolique_PAS_arrivee_du_smur', 
                                    'pression_arterielle_diastolique_PAD_arrivee_du_smur', 
                                    'score_glasgow_initial', 'score_glasgow_moteur_initial', 
                                    'anomalie_pupillaire_prehospitalier', 'frequence_cardiaque_FC_arrivee_du_smur', 
                                    'arret_cardio_respiratoire_massage', 'penetrant_objet', 'ischemie_du_membre', 
                                    'hemorragie_externe', 'amputation'])

    with open(filepath) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            
            if line_count>1:
                
                IPP = get_IPP(row)
                if not np.isnan(IPP):

                    age = get_age(row)
                    hemocue = get_hemocue(row)
                    fracas_bassin = get_fracas_bassin(row)
                    catecholamines = get_catecholamines(row)
                    PAS = get_PAS(row)
                    PAD = get_PAD(row)
                    glasgow = get_glasgow(row)
                    glasgow_moteur = get_glasgow_moteur(row)
                    anomalie_pupille = get_anomalie_pupille(row)
                    freq_cardiaque = get_freq_cardiaque(row)
                    ACR = get_ACR(row)
                    penetrant = get_penetrant(row)
                    ischemie = get_ischemie(row)
                    hemorragie = get_hemorragie(row)
                    amputation = get_amputation(row)
                    #print(amputation)
                    
                    df.loc[len(df.index)] = [IPP, age, hemocue, fracas_bassin, catecholamines, PAS, PAD, glasgow, 
                                            glasgow_moteur, anomalie_pupille, freq_cardiaque, ACR, penetrant,
                                            ischemie, hemorragie, amputation]

            line_count+=1
    return df


def get_clinical_data_outcome(filepath):
    """
    Returns a pandas DataFrame containing IPP and the outcome (neurosurgery OR IPC)
    """

    df_outcome = pd.DataFrame(columns=['IPP', 'neurochir+pic'])

    with open(filepath) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            
            if line_count>1:

                ipp = get_IPP(row)
                if not np.isnan(ipp):
                    
                    neurochir = get_neurochir(row)
                    pic = get_pic(row)

                    combined = int(0)

                    if neurochir==int(1) or pic == int(1):
                        combined = int(1)
                    
                    if np.isnan(neurochir) and np.isnan(pic):
                        combined = MISSING_VALUE
                    
                    df_outcome.loc[len(df_outcome.index)] = [ipp, combined]

            line_count+=1
    
    return df_outcome
            