# Detect traffic anomaly to predict accidents
 
Full code for running on Google Colab (with visualization): final full code.ipynb

extract_motion_mask.py: extract motion mask

detection (for local running).ipynb: extract average frames, detect vehicles that are visibile in the mask (in white part of the mask)

detect_anomaly.py: detect anomaly based on list of vehicles detected in the previous stage
