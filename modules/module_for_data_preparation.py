import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd())))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from sklearn.preprocessing import LabelEncoder
import datetime
import dateutil.parser


class DataPreparation:
    def __init__(self, path):
        self.recommend_limit = 5000
        self.path = path

    def refine_behavioral_data(self, behavioral_data, recommend_limit):
        """
        행동데이터 정제 후 로컬 저장, mapping_df(label_encoding data를 실제 content_id로 바꾸는 데 필요) 생성
        """
        # mydf2 = behavioral_data[['user_id', 'keyword_name', 'stay_time', 'content_click_time']]
        mydf2 = behavioral_data
        num_recommend = [i for i in range(recommend_limit)]
        # mydf2.columns = ['userId', 'movieId', 'rating', 'timestamp']
        le = LabelEncoder()
        le.fit(mydf2.keyword_name)
        mydf2['book_id'] = le.transform(mydf2['keyword_name'])
        mydf2['book_id'] = mydf2['book_id'].astype(int)
        mydf2 = mydf2.loc[mydf2['book_id'].isin(num_recommend), :]  ## 5000개 이하만
        tmp_df = mydf2[['book_id', 'keyword_name']]
        tmp_df.drop_duplicates(subset=['book_id'], inplace=True)
        # mydf3 = mydf2[['userId', 'rating', 'timestamp', 'book_id']]
        # mydf3.columns = ['reader_id', 'liked', 'when', 'book_id']
        ### str일 경우
        # mydf3['when'] = mydf3['when'].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d %H:%M"))
        mydf2['when'] = mydf2['when'].apply(lambda x: dateutil.parser.parse(x))
        ### datetime일경우 바로
        mydf2['when'] = mydf2['when'].apply(lambda x: datetime.datetime.strftime(x, "%m/%d/%Y, %H:%M:%S"))
        ## 학습 데이터 로컬 저장

        mydf2.to_csv(os.path.join(self.path, 'mydataset', 'mydf.csv'), index=False)

        return tmp_df

    def run(self, behavioral_data):
        tmp_df = self.refine_behavioral_data(behavioral_data, self.recommend_limit)
        tmp_df.to_csv(os.path.join(self.path, 'mydataset', 'mapping_df.csv', mode='w', encoding='utf-8-sig'))
