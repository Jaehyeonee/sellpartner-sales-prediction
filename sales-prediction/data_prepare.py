import pandas as pd

max_prediction_length = 1  # 다음 주 예측
max_encoder_length = 24    # 과거 12주 사용 (기존 24주에서 축소)
min_encoder_length = 8     # 최소 8주의 데이터 필요

def create_timeseries(filtered_data):
    # time_idx 재생성
    # week_date를 datetime 형식으로 변환
    filtered_data['week_date'] = pd.to_datetime(filtered_data['week_date'])
    # product_id를 문자열로 변환
    filtered_data['product_id'] = filtered_data['product_id'].astype(str)
    # 각 product_id마다 0부터 시작하는 time_idx 생성 -> 상품 별로 시계열 예측을 진행하도록 함.
    filtered_data = filtered_data.sort_values(["product_id", "week_date"]).reset_index(drop=True)
    filtered_data["time_idx"] = filtered_data.groupby("product_id").cumcount()

    return filtered_data


def prepare_data(data):
    # 데이터 준비 단계
    print("데이터 필터링 전:")
    print(f"전체 제품 수: {len(data['product_id'].unique())}")

    # 시퀀스 길이 계산
    sequence_lengths = data.groupby('product_id').size()
    print("\n시퀀스 길이 통계:")
    print(f"최소 시퀀스 길이: {sequence_lengths.min()}")
    print(f"최대 시퀀스 길이: {sequence_lengths.max()}")
    print(f"평균 시퀀스 길이: {sequence_lengths.mean():.2f}")

    # 최소 필요 길이 설정(8주 이상의 데이터를 가지고 있을 경우에만 활용)
    min_required_length = min_encoder_length + max_prediction_length
    valid_products = sequence_lengths[sequence_lengths >= min_required_length].index

    print(f"\n조정된 최소 필요 데이터 길이: {min_required_length}")
    print(f"충분한 데이터를 가진 제품 수: {len(valid_products)}")
    print(f"포함된 제품 비율: {(len(valid_products) / len(sequence_lengths) * 100):.2f}%")

    # 유효한 제품만 필터링
    filtered_data = data[data['product_id'].isin(valid_products)].copy()
    prepared_data = create_timeseries(filtered_data)
    
    return prepared_data
    