import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'RecNN')))

sys.path.append(os.path.abspath(os.path.join(os.getcwd(),'RecNN')))
import recnn
import recnn
import torch
import torch.nn as nn
import json
import torch_optimizer as optim
import time
import datetime
import pickle
import torch.nn.functional as F
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from sklearn.preprocessing import LabelEncoder
import dateutil.parser
import shap
import numpy as np
# import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use
class ContentRank():
    """
    사용자 로그를 input으로 받아서 사용자별 추천리스트를 return해주는 class
    """
    def __init__(self):
        try:
            with open(os.path.join(os.getcwd(), "model", "model_config.json"), "r") as jsonfile:
                model_config = json.load(jsonfile)
                print("Read successful")
        except FileNotFoundError:
            with open(os.path.join("..", "model", "model_config.json"), "r") as jsonfile:
                model_config = json.load(jsonfile)
                print("Read successful")
        self.input_size = model_config['input_size']
        self.num_items = model_config['num_items']
        self.frame_size = model_config['frame_size']
        # self.db = Postgresql_DB()
        try:
            self.mapping_df = pd.read_csv(os.path.join('..', 'mydataset', 'mapping_df.csv'))
        except FileNotFoundError:
            self.mapping_df = pd.read_csv(os.path.join(os.getcwd(), 'mydataset', 'mapping_df.csv'))
        self.mapping_df = self.mapping_df[['book_id','keyword_name']]
        self.cuda = 'cpu'

    def string_time_to_unix(self, s):
        return int(time.mktime(datetime.datetime.strptime(s, "%m/%d/%Y, %H:%M:%S").timetuple()))

    def for_like_reward(self, like):
        if like > 1200:
            like = 1200
        if like < 0:
            like = 0
        like = (1 / 120) * like - 5
        return like
    def prepare_dataset(self, df):
        # df['liked'] = df['liked'].apply(lambda a: self.for_like_reward(a))
        ##@TODO: 이미 등록한 키워드 제외 로직 필요
        # seen_id_list = list(df['keyword_name'])
        # df = df.sort_values(ascending=False, by=['keyword_name'])
        # le = LabelEncoder()
        # le.fit(df.keyword_name)
        # df['book_id'] = le.transform(df['keyword_name'])
        # df['book_id'] = df['book_id'].astype(int)
        # df.drop('keyword_name', axis=1, inplace=True)
        ### str일 경우(json에서 온 데이터 파싱하기)
        df['when'] = df['when'].apply(lambda x: dateutil.parser.parse(x))
        #df['when'] = df['when'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M"))
        ### datetime일경우 바로
        df['when'] = df['when'].apply(lambda x: datetime.datetime.strftime(x, "%m/%d/%Y, %H:%M:%S"))
        df['when'] = df['when'].apply(self.string_time_to_unix)
        # users = df[['reader_id', 'book_id']].groupby(['reader_id']).size()
        # users = users[users > self.frame_size].sort_values(ascending=False).index
        ratings = df.sort_values(by='when').set_index('reader_id').drop('when', axis=1).groupby('reader_id')

        # Groupby user
        user_dict = {}
        # userid = df.reader_id[0]
        userid = 1
        def app(x):
            # userid = x.index[0]
            userid = 1
            user_dict[int(userid)] = {}
            # user_dict[int(userid)]['items'] = x['book_id'].values
            # user_dict[int(userid)]['ratings'] = x['rating'].values
            user_dict[int(userid)]['states'] = x[
                ['페이지1 방문', '페이지2 방문', '페이지3 방문', '누적방문수', '누적클릭수', '누적체류시간', '누적뷰티키워드검색', '누적IT키워드검색']].values

        ratings.apply(app)

        # item_t = torch.tensor([user_dict[int(userid)]['items']])
        # ratings_t = torch.tensor([user_dict[int(userid)]['ratings']])
        # myembedding = torch.stack([value for key, value in myembedding.items()])
        # items_emb = myembedding[item_t.long()]
        # b_size = ratings_t.size(0)
        # items = items_emb[:, :, :].view(b_size, -1)
        # ratings = ratings_t[:, :]
        # state = torch.cat([items, ratings], 1)
        state = torch.tensor(user_dict[int(userid)]['states'])
        return state, userid #, seen_id_list
    def make_state(self,prediction_data_sample):
        """
        prediction에 필요한 input 생성 함수
        """
        # prediction_data_sample.columns = ['reader_id', 'liked', 'when', 'content_id']
        # myembedding_df = self.db.get_embedding(tuple(prediction_data_sample['content_id']))
        # myembedding_df = myembedding_df.sort_values(ascending=False, by=['content_id'])
        # myembedding = dict({a: torch.tensor(b) for a, b in zip(myembedding_df['content_id'], myembedding_df['embeddings'])})
        state, userid= self.prepare_dataset(prediction_data_sample)
        return state, userid #, seen_id_list

    def get_id_list(self, policy_net2, state, mapping_df):
        """
        추천리스트 prediction하는 함수
        """
        x = F.relu(policy_net2.linear1(state.float()))
        action_scores = policy_net2.linear2(x)
        rank = action_scores.argsort(descending=True)
        rank_id_df = pd.DataFrame(rank.numpy()[0], columns=['book_id'])
        rank_df = pd.merge(rank_id_df, mapping_df, on=['book_id'])
        ### 본건 제외
        # rank_df = rank_df.loc[~rank_df['content_id'].isin(seen_id_list), :]
        rank_df.drop('book_id', axis=1, inplace=True)
        ##Todo: rank 몇위까지 보여줄지 정해야
        return rank_df
    def run(self, prediction_data_sample):
        """

        :params prediction_data_sample: frame_size만큼 모인 사용자 로그 데이터
        :return rank_df: 사용자별 추천 콘텐츠 목록

        """
        state, userid= self.make_state(prediction_data_sample)
        cuda = self.cuda
        policy_net2 = recnn.nn.models.DiscreteActor(self.input_size, self.num_items, 2048).to(cuda)
        try:
            policy_net2.load_state_dict(torch.load(os.path.join(os.getcwd(), "model", "policy_net.pt")))
        except FileNotFoundError:
            policy_net2.load_state_dict(torch.load(os.path.join("..", "model", "policy_net.pt")))
        policy_net2.eval()
        rank_df = self.get_id_list(policy_net2, state, self.mapping_df)
        return rank_df
    def run_xai(self, prediction_data_sample):

        state, userid= self.make_state(prediction_data_sample)
        cuda = self.cuda
        policy_net2 = recnn.nn.models.DiscreteActor(self.input_size, self.num_items, 2048).to(cuda)
        try:
            policy_net2.load_state_dict(torch.load(os.path.join(os.getcwd(), "model", "policy_net.pt")))
        except FileNotFoundError:
            policy_net2.load_state_dict(torch.load(os.path.join("..", "model", "policy_net.pt")))
        policy_net2.eval()
        shap.initjs()
        explainer = shap.DeepExplainer(policy_net2, data= state.float())
        ### shap은 state를 여러개 넣어야 한다
        shap_value = explainer.shap_values(state.float())
        # shap_value[451] = np.array([[0.4,0.5,0.6,0.9,0.8,-0.1,-0.7,0.5]])
        shap.force_plot(explainer.expected_value[451], shap_value[451],features=prediction_data_sample[['페이지1 방문', '페이지2 방문', '페이지3 방문', '누적방문수', '누적클릭수', '누적체류시간', '누적뷰티키워드검색', '누적IT키워드검색']].values,feature_names=['vistpage1', 'vistpage2', 'vistpage3', 'sumvist', 'sumclick', 'sumstaytime', 'beautykeywordsearch', 'ITkeywordsearch'], show=False,matplotlib=True).savefig(os.getcwd() + '\\output\\force.png')
        ## 해당 코드는 스테이트 2개 이상일때 실행하자 스테이트가 2개 이상이여야 기여도가 나온다
        # shap.force_plot(explainer.expected_value[451], shap_value[451][0], features=prediction_data_sample[
        #     ['페이지1 방문', '페이지2 방문', '페이지3 방문', '누적방문수', '누적클릭수', '누적체류시간', '누적뷰티키워드검색', '누적IT키워드검색']].values,
        #                 feature_names=['vistpage1', 'vistpage2', 'vistpage3', 'sumvist', 'sumclick', 'sumstaytime',
        #                                'beautykeywordsearch', 'ITkeywordsearch'], show=False, matplotlib=True).savefig(
        #     os.getcwd() + '\\output\\force.png')
        return shap_value