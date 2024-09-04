# Integrating Medical Imaging and Deep Learning for Analysis in Traumatic Brain Injuries Clinical Data 


This study is divided into two parts:
## Improving the CT-TIQUA [1] brain CT scan lesion analysis tool
- New brain extraction technique using TotalSemgentator [2]
- New registration technique using ANTS SyN with histogram matching or LDDMM

## Implementing a classification tool to predict the need for neurosurgery or intra-cranial pressure control
- Using gradient boosting (scikit-learn)
- Comparing classification using prehospital clinical data only, brain lesion volumes only and combination of both



This isn't a ready-to use tool but rather the materials to reproduce results from my master thesis.


## Structure of the repository
- brain_extraction: Comparison of different brain extraction methods on CT scans
- clinical_data: scripts to extract and format prehospital clinical data
- gradient_boosting: classification pipelines
- mega_CT_TIQUA: evaluating modifications of the CT-TIQUA tool
- registration: comparing registration methods


[1] Wasserthal, J., Breit, H. C., Meyer, M. T., Pradella, M., Hinck, D., Sauter, A. W., ... & Segeroth, M. (2023). TotalSegmentator: robust segmentation of 104 anatomic structures in CT images. Radiology: Artificial Intelligence, 5(5).

[2] Brossard, C., Grèze, J., de Busschère, J. A., Attyé, A., Richard, M., Tornior, F. D., ... & Lemasson, B. (2023). Prediction of therapeutic intensity level from automatic multiclass segmentation of traumatic brain injury lesions on CT-scans. Scientific Reports, 13(1), 20155.
