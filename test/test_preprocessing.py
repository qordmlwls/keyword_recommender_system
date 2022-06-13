import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.module_for_data_preparation import DataPreparation

import pandas as pd

if __name__ == '__main__':
    try:
        behavioral_data = pd.read_csv(os.path.join('..','mydataset','behavioral_data.csv'))
        path = '..'
    except FileNotFoundError:
        behavioral_data = pd.read_csv(os.path.join(os.getcwd(),'mydataset','behavioral_data.csv'))
        path = os.getcwd()

    preparationer = DataPreparation(path)
    preparationer.run(behavioral_data)