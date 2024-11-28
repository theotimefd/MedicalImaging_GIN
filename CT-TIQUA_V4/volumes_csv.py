import os
import numpy as np
import pandas as pd
import csv

directory = "/home/fehrdelt/data_ssd/scripts/mega_CT_TIQUA/volumes_output/"

# atlas regions not used:
"""
Third_ventricle
Fourth_ventricle
Right_lateral_ventricle
Left_lateral_ventricle
Right_frontal_CSF
Left_frontal_CSF
Right_sylvian_CSF
Left_sylvian_CSF
Right_parietal_CSF
Left_parietal_CSF
Right_temporal_CSF
Left_temporal_CSF
Right_occipital_CSF
Left_occipital_CSF
Basal_cisterns
Perimesencephalic_space
Right_cerebellar_CSF
Left_cerebellar_CSF
"""

supratentorial = ["Right_frontal_cortex","Left_frontal_cortex","Right_temporal_cortex","Left_temporal_cortex","Right_insular_cortex","Left_insular_cortex",
                  "Right_parietal_cortex","Left_parietal_cortex","Right_occipital_cortex","Left_occipital_cortex","Right_basal_ganglia","Left_central_ganglia",
                  "Right_hypothalamus","Left_hypothalamus","Right_frontal_white_matter","Left_frontal_white_matter","Right_temporal_white_matter",
                  "Left_temporal_white_matter","Right_occipital_white_matter","Left_occipital_white_matter","Right_parietal_white_matter","Left_parietal_white_matter",
                  "Right_insular_white_matter","Left_insular_white_matter"]

brainstem = ["Brainstem"]

infratentorial = ["Right_cerebellum_lobe", "Left_cerebellum_lobe","Cerebellar_vermis"]

extracerebral = ["Right_posterior_inferior_extra_cerebral_space","Left_posterior_inferior_extra_cerebral_space","Right_anterior_inferior_extra_cerebral_space",
                 "Left_anterior_inferior_extra_cerebral_space","Right_posterior_superior_extra_cerebral_space","Left_posterior_superior_extra_cerebral_space",
                 "Right_Superior_Anterior_Extra_Brain_Space","Left_Superior_Anterior_Extra_Brain_Space", "Rest"]

columns=['name','supratentorial_IPH', 'supratentorial_SAH', 'supratentorial_Petechiae', 'supratentorial_Edema',
                           'infratentorial_IPH', 'infratentorial_SAH', 'infratentorial_Petechiae', 'infratentorial_Edema',
                           'brainstem_IPH', 'brainstem_SAH', 'brainstem_Petechiae', 'brainstem_Edema', 
                           'SDH', 'EDH']

volumes_dict = {}

def get_volumes_csv(filepath):
    """
    Returns dictionnary with the volume of each lesion in its corresponding zone
    """

    with open(filepath) as csv_file:
        
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        zones_number_row = []
        zones_name_row = []

        patient_volumes =  {'supratentorial_IPH':0, 
                            'supratentorial_SAH':0, 
                            'supratentorial_Petechiae':0, 
                            'supratentorial_Edema':0,
                            'infratentorial_IPH':0, 
                            'infratentorial_SAH':0, 
                            'infratentorial_Petechiae':0, 
                            'infratentorial_Edema':0,
                            'brainstem_IPH':0, 
                            'brainstem_SAH':0, 
                            'brainstem_Petechiae':0, 
                            'brainstem_Edema':0,
                            'SDH':0, 
                            'EDH':0}

        for row in csv_reader:
            
            if line_count == 0:
                zones_number_row = row
            if line_count == 1:
                zones_name_row = row

            if line_count>1:
                
                if line_count==3: # IPH
                    for region in supratentorial:
                        try:
                            patient_volumes['supratentorial_IPH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in infratentorial:
                        try:
                            patient_volumes['infratentorial_IPH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in brainstem:
                        try:
                            patient_volumes['brainstem_IPH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass


                if line_count==4: # SDH
                    for region in extracerebral:
                        try:
                            patient_volumes['SDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in supratentorial:
                        try:
                            patient_volumes['SDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in infratentorial:
                        try:
                            patient_volumes['SDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in brainstem:
                        try:
                            patient_volumes['SDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass

                if line_count==5: # EDH
                    for region in extracerebral:
                        try:
                            patient_volumes['EDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in supratentorial:
                        try:
                            patient_volumes['EDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in infratentorial:
                        try:
                            patient_volumes['EDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in brainstem:
                        try:
                            patient_volumes['EDH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass

                if line_count==6: # IVH
                    pass

                if line_count==7: # SAH
                    for region in supratentorial:
                        try:
                            patient_volumes['supratentorial_SAH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in infratentorial:
                        try:
                            patient_volumes['infratentorial_SAH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in brainstem:
                        try:
                            patient_volumes['brainstem_SAH'] += int(row[zones_name_row.index(region)])
                        except:
                            pass

                if line_count==8: # Petechiae
                    for region in supratentorial:
                        try:
                            patient_volumes['supratentorial_Petechiae'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in infratentorial:
                        try:
                            patient_volumes['infratentorial_Petechiae'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in brainstem:
                        try:
                            patient_volumes['brainstem_Petechiae'] += int(row[zones_name_row.index(region)])
                        except:
                            pass

                if line_count==9: # Edema
                    for region in supratentorial:
                        try:
                            patient_volumes['supratentorial_Edema'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in infratentorial:
                        try:
                            patient_volumes['infratentorial_Edema'] += int(row[zones_name_row.index(region)])
                        except:
                            pass
                    for region in brainstem:
                        try:
                            patient_volumes['brainstem_Edema'] += int(row[zones_name_row.index(region)])
                        except:pass



            line_count += 1
    
    return patient_volumes
            

"""
with open(directory+"cleaned_volumes.csv", 'w') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerow(patient_volumes)
"""