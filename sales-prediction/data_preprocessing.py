import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from pandas.tseries.offsets import Week

# 1. 원본 데이터 로드
def load_data(data_path):
    data_path = data_path           
    df = pd.read_csv(data_path)
    return df

# 2. 비어있는 주차 처리 함수 >> 결측 주차에 대해서 '평균 or 1' 로 채움
def fill_missing_weeks_and_impute(df):
    filled_df = pd.DataFrame()
    # week_date -> datetime 형태로 변형
    df['week_date'] = pd.to_datetime(df['week_date'], errors='coerce')
    # 각 상품별로 데이터 프레임을 생성하여 결측 추가 채움
    for product_id, group in df.groupby('product_id'):
        # 주차 범위를 생성
        
        all_weeks = pd.date_range(start=group['week_date'].min(), end=group['week_date'].max(), freq=Week())
        
        group = group.set_index('week_date').reindex(all_weeks).rename_axis('week_date').reset_index()
        
        group['product_id'] = product_id 

        # 학습 성능에 영향을 주는 결측값 (변동성 ) -------------------------------------------------------------
        group['week_review_count'] = group['week_review_count'].fillna(1)
        group['week_purchase_cnt'] = group['week_purchase_cnt'].fillna(1)
        group['average_review_score'] = group['average_review_score'].fillna(group['average_review_score'].mean())
        # --------------------------------------------------------------------------------------------------
        # 결측값 채우기
        group['price'] = group['price'].fillna(group['price'].mean())
        
        group['review_cnt'] = group['review_cnt'].fillna(group['review_cnt'].mean())
        group['wish_cnt'] = group['wish_cnt'].fillna(group['wish_cnt'].mean())
        group['sixMothRatio(puchase_cnt/review_cnt)'] = group['sixMothRatio(puchase_cnt/review_cnt)'].fillna(group['sixMothRatio(puchase_cnt/review_cnt)'].mean())
        group['purchase_cnt'] = group['purchase_cnt'].fillna(group['purchase_cnt'].mean())
        group['category1_encoded'] = group['category1_encoded'].fillna(group['category1_encoded'].mean())
        group['category2_encoded'] = group['category2_encoded'].fillna(group['category2_encoded'].mean())
        group['category3_encoded'] = group['category3_encoded'].fillna(group['category3_encoded'].mean())
        
        # 결측 주가 채워진 데이터프레임 filled_df
        filled_df = pd.concat([filled_df, group])
        most_frequent_names = filled_df.groupby('product_id')['product_name'].apply(lambda x: x.mode()[0])
        filled_df['product_name'] = filled_df.apply(lambda row: most_frequent_names[row['product_id']] if pd.isna(row['product_name']) else row['product_name'], axis=1)


    return filled_df.reset_index(drop=True)

# 3. <피처 추가 > 6주 동안의 이동평균(rolling mean)과 표준편차(roling_std)를 계산 >> 이동평균을 통해 최근 트렌드를, 표준편차는 판매량의 변동성을 파악하는데 도움
def rolling_(df):
    df['rolling_mean_purchase'] = df.groupby('product_id')['week_purchase_cnt'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )
    df['rolling_std_purchase'] = df.groupby('product_id')['week_purchase_cnt'].transform(
        lambda x: x.rolling(window=6, min_periods=1).std()
    )
    # 첫 번째 주 이동표준편차 값 (이전 5주, 최소 1주 전 데이터가 없기 때문에) NaN에대한 처리 (0으로 대체)
    df['rolling_std_purchase'].fillna(0, inplace=True)
    
    return df

# 4. <피처 추가 > 시계열 특성 생성 함수
def create_time_features(df):
    df['week_num'] = pd.to_datetime(df['week_date']).dt.isocalendar().week
    df['month'] = pd.to_datetime(df['week_date']).dt.month
    
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['week_sin'] = np.sin(2 * np.pi * df['week_num']/52)
    df['week_cos'] = np.cos(2 * np.pi * df['week_num']/52)
    
    return df

# 5. 전체 데이터 전처리 
def data_preprocessing(df):
    # 5-1. 사용하지 않는 컬럼에 대한 드랍 처리
    df.drop(columns=['Unnamed: 0', 'url', 'registration_dt'], axis=1, inplace=True)
    # 5-2. category3Id의 null 값 : 0으로 대체
    df = df.fillna(0)
    # 5-3. category1,2,3 id값 인코딩 진행
    categories = ['category1Id', 'category2Id', 'category3Id']
    encoded_categories = ['category1_encoded', 'category2_encoded', 'category3_encoded']
    label_encoders = {}
    for c, new_col in zip(categories, encoded_categories):
        label_encoder = LabelEncoder()
        df[new_col] = label_encoder.fit_transform(df.pop(c))  # 원래 열을 인코딩 후 새 열로 저장
        label_encoders[c] = label_encoder       
    
    df_filled = fill_missing_weeks_and_impute(df)

    df_filled = rolling_(df_filled)
    
    data = create_time_features(df_filled)

    return data


