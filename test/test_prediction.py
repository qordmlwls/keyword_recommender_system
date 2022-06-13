import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join('..', 'RecNN')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'RecNN')))
from modules.module_for_real_time_prediction import ContentRank

if __name__ == '__main__':
    try:
        prediction_data_sample = pd.read_csv(os.getcwd() + '\\mydataset\\prediction_data_sample.csv')
    except FileNotFoundError:
        prediction_data_sample = pd.read_csv('..' + '\\mydataset\\prediction_data_sample.csv')
    rank_producer = ContentRank()
    #rank_df = rank_producer.run(prediction_data_sample)
    shap_value = rank_producer.run_xai(prediction_data_sample)

    print('end')