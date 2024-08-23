import os
import pandas as pd
import numpy as np


from clinical_data.clinical_data import get_clinical_data, get_clinical_data_outcome
from mega_CT_TIQUA.volumes_csv import get_volumes_csv



def make_row(id, volumes_df, shanoir_import_df, clinical_data_df, outcome):
    volumes = list(volumes_df.loc[id])
    try:
        ipp = (shanoir_import_df[shanoir_import_df[5] == volumes[0]])[0].values[0]
                
    except:
        return []
    
    try:
        clinical_data = clinical_data_df.loc[clinical_data_df["IPP"]==float(ipp)]
    except:
        return []

    volumes = np.array(volumes)
    clinical_data = clinical_data.values[0][1:]

    try:
        outcome_patient = list(outcome.loc[outcome["name"]==volumes[0]]["neurochir+pic"])[0]
    except:
        return []
    
    combined = np.concatenate([volumes, clinical_data, [outcome_patient]])


    return combined



def get_training_csv(brain_extraction_method, registration_method, output_folder):
    """
    arguments:
    brain_extraction_method: "custom_nn", "matlab", "TTS"
    registration_method: "ANTS", "ANTS_hist_match", "LDDMM"
    output_folder: where the csv file is saved

    returns:
    A csv file containing the segmentation volumes, clinical data and outcome for each patient.
    """

    configuration = brain_extraction_method+"_"+registration_method

    clinical_data_df = get_clinical_data("/home/fehrdelt/data_ssd/data/clinical_data/Full/cleaned_data_full.csv")
    shanoir_import_df = pd.read_csv("/home/fehrdelt/data_ssd/data/clinical_data/Full/shanoir_import_full.csv", header=None)

    ct_tiqua_temp_directory = "/home/fehrdelt/data_ssd/data/mega_CT_TIQUA_temp/"

    volumes = {}

    volumes_df = pd.DataFrame(columns=["name", 'supratentorial_IPH', 
                                'supratentorial_SAH', 'supratentorial_Petechiae', 
                                'supratentorial_Edema','infratentorial_IPH', 
                                'infratentorial_SAH', 'infratentorial_Petechiae', 
                                'infratentorial_Edema', 'brainstem_IPH', 
                                'brainstem_SAH', 'brainstem_Petechiae', 
                                'brainstem_Edema','extracerebral_SDH', 'extracerebral_EDH'])

    for file in os.listdir(ct_tiqua_temp_directory):
        if os.path.isdir(ct_tiqua_temp_directory+file) and file[0:2] == "P0" and len(file)==5: # takes only the first image in case 2 were taken the same day: P0016 and P0016a
            volumes = get_volumes_csv(ct_tiqua_temp_directory+file+"/"+file+"_"+configuration+"_Volumes.csv")
            volumes_df.loc[len(volumes_df)] = [file]+list(volumes.values())

    outcome_csv = get_clinical_data_outcome("/home/fehrdelt/data_ssd/data/clinical_data/Full/cleaned_data_full.csv")

    outcome = pd.DataFrame(columns=["name", "neurochir+pic"])

    for i in range(len(outcome_csv)):

        ipp = outcome_csv.loc[i]["IPP"]


        temp_list = list(shanoir_import_df.loc[shanoir_import_df[0]==int(ipp)][5])

        if len(temp_list)>0: # sometimes the patient isn't on shanoir import because i deleted it afterwards (minor, etc) -> skip this patient
            name = list(shanoir_import_df.loc[shanoir_import_df[0]==int(ipp)][5])[0]

            outcome.loc[len(outcome)] = [name, outcome_csv.loc[i]["neurochir+pic"]]

    combined_volumes_clinical_df = pd.DataFrame(columns=["name", 'supratentorial_IPH', 
                            'supratentorial_SAH', 'supratentorial_Petechiae', 
                            'supratentorial_Edema','infratentorial_IPH', 
                            'infratentorial_SAH', 'infratentorial_Petechiae', 
                            'infratentorial_Edema', 'brainstem_IPH', 
                            'brainstem_SAH', 'brainstem_Petechiae', 
                            'brainstem_Edema','extracerebral_SDH', 'extracerebral_EDH',
                            'age', 'hemocue_initial', 'fracas_du_bassin', 'catecholamines', 
                                    'pression_arterielle_systolique_PAS_arrivee_du_smur', 
                                    'pression_arterielle_diastolique_PAD_arrivee_du_smur', 
                                    'score_glasgow_initial', 'score_glasgow_moteur_initial', 
                                    'anomalie_pupillaire_prehospitalier', 'frequence_cardiaque_FC_arrivee_du_smur', 
                                    'arret_cardio_respiratoire_massage', 'penetrant_objet', 'ischemie_du_membre', 
                                    'hemorragie_externe', 'amputation', 'outcome_neurochir_pic'])



    for row_id in range(len(volumes_df)):
        
        row = make_row(id=row_id, volumes_df=volumes_df, shanoir_import_df=shanoir_import_df, clinical_data_df=clinical_data_df, outcome=outcome)
        if len(row)>0:
            combined_volumes_clinical_df.loc[len(combined_volumes_clinical_df)] = row

    print(combined_volumes_clinical_df.head)

    combined_volumes_clinical_df.to_csv(os.path.join(output_folder, f"combined_clinical_data_volumes_outcome_{configuration}.csv"))