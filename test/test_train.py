import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from modules.module_for_train_recommendation_model import *


if __name__ == '__main__':

    try:
        keyword_meta = pd.read_csv('../mydataset/mapping_df.csv')
    except:
        keyword_meta = pd.read_csv(os.path.join(os.getcwd(), 'mydataset', 'mapping_df.csv'))
    args = {
        'frame_size': 10,
        'batch_size': 10,
        'n_epochs': 100,
        'plot_every': 30,
        'num_items': len(keyword_meta['book_id'])  # n items to recommend. Can be adjusted for your vram
    }
    train_model = TrainRecommendationModel(**args)
    train_model.run()