import os
import subprocess


# ---------------------- Overall -----------------------
competition = 'competitive-data-science-predict-future-sales'

# ------------------------ PATH ------------------------
ROOT_DIR = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).rstrip().decode('utf-8')

DATA_DIR = os.path.join('~/.kaggle/competitions', competition)
