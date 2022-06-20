1. DESCRIPTION
This is the implementation of STAPI pipeline. It is recommended to run STAPI under Python3.7 and a virtual environment.



2. INSTALLATION
To install required packages, run the following command in terminal.

pip install -r requirements.txt


To use our model, replace the Data folder with the downloaded dataset (../Data folder).



3. EXECUTION
To train STAPI from scratch, run the following command. The trained models (both filter and typographic classifier) will be saved in output folder. Confusion matrices and ROC curves will be generated.

python train.py


