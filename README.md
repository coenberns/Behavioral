This repository combines the EGG data from the recordings using MiGUT duty-cycling and behavioral characterization. 
-> Unfortunately, the behavioral characterization pipeline of the lab has not been finalized but it utilizes Vitpose+ and GROUNDING_DINO to generate key-points of the animal. 
-> Subsequently, a pre-trained LSTM is used for the generation of behavioral characterization. Once the accuracy is properly good, this pipeline can be used in combination with the complete EGG recordings. 

For now, this repo was focused on generating a pipeline based on manually labeled video data (and thus behav. charact.) and combining it with the synchronized EGG data into the Behavioral trained models (CEBRA-Behavior). 
Additionally, CEBRA-Time, self-supervised contrastive learning based solely on the EGG data and it's evenly spaced time-variable, was used here. 

It should be noted that the results here were merely used as a Proof-of-Concept for this thesis. The main results of the findings are in the "initial_multi_approach.py" script.
Additionally, a script was generated for multi-session training for future endeavors. 


