import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss
from lightning.pytorch.tuner import Tuner
import numpy as np
from pytorch_lightning import Trainer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE
import os
import torch
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from data_preprocessing import load_data, data_preprocessing
from data_prepare import prepare_data, min_encoder_length, max_encoder_length, max_prediction_length

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    # data_path = '../data/train_sales_projection_v0.5.csv'
    data_path = '/Users/jenzennii/Development/sellpartner/model/sellpartner-sales-prediction/data/train_sales_projection_v0.5.csv'
    features = [ 
                'price',
                'review_cnt',
                'wish_cnt',
                'sixMothRatio(puchase_cnt/review_cnt)',
                'week_review_count',
                'average_review_score',
                'category1_encoded',
                'category2_encoded',
                'category3_encoded',
                'rolling_mean_purchase',
                'rolling_std_purchase',
                'week_num',
                'month',
                'month_sin',
                'month_cos',
                'week_sin',
                'week_cos']

    save_dir = './saved_models'
    os.makedirs(save_dir, exist_ok=True)
    model_filename = os.path.join(save_dir, 'temporal_fusion_transformer.pt')

    # 1. 데이터 로드
    df = load_data(data_path)
    print('data 로드 성공')
    # 2. 데이터 전처리
    data = data_preprocessing(df)
    print('data 전처리 성공')
    # 3. 학습 데이터로 준비
    data = prepare_data(data)
    print('data 학습 데이터로 변환 성공')

    # 모델 학습
    def TFT_(prepared_data):
        # training cutoff 설정 (각 시퀀스의 80%를 훈련에 사용)
        prepared_data["training_cutoff"] = prepared_data.groupby("product_id")["time_idx"].transform(
            lambda x: int(len(x) * 0.8)
        )
        # Training dataset 생성
        training = TimeSeriesDataSet(
            prepared_data[lambda x: x.time_idx <= x.training_cutoff],
            time_idx="time_idx",
            target="week_purchase_cnt",
            group_ids=["product_id"],
            min_encoder_length=min_encoder_length,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["product_id"],
            time_varying_known_reals=["time_idx"] + features,
            time_varying_unknown_reals=["week_purchase_cnt"],
            target_normalizer=GroupNormalizer(groups=["product_id"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Validation dataset 생성
        validation = TimeSeriesDataSet.from_dataset(
            training,
            prepared_data,
            min_prediction_idx=training.index.time.max() + 1,
            stop_randomization=True
        )

        # 데이터 로더 생성
        batch_size = 128  # 배치 크기 증가
        train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=0)


        # Early stopping 설정 조정
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=1e-4,
            patience=10,
            verbose=True,
            mode="min"
        )

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        # 모델 파라미터 조정
        # 모델 초기화
        pl.seed_everything(42)

        tft = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=0.03,
            hidden_size=32,
            attention_head_size=2,
            dropout=0.2,  # 드롭아웃 증가
            hidden_continuous_size=16,
            loss=RMSE(),
        )

        # Trainer 초기화
        trainer = pl.Trainer(
            max_epochs=100,
            devices=1,
            accelerator='gpu',
            gradient_clip_val=0.1,
            callbacks=[early_stop_callback, lr_monitor],
            enable_progress_bar=True
        )


        # 모델 학습
        print("\n모델 학습 시작...")
        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        # 모델 저장
        print(f"\n모델을 {model_filename}에 저장 중...")
        torch.save({
            'model_state_dict': tft.state_dict(),
            'dataset_parameters': {
                'time_varying_known_reals': training.time_varying_known_reals,
                'time_varying_unknown_reals': training.time_varying_unknown_reals,
                'group_ids': training.group_ids,
                'min_encoder_length': training.min_encoder_length,
                'max_encoder_length': training.max_encoder_length,
                'max_prediction_length': training.max_prediction_length,
            }
        }, model_filename)

        print("모델 저장 완료!")

        # 검증 세트에 대한 성능 평가
        print("\n검증 세트 성능 평가:")
        validation_predictions = tft.predict(val_dataloader)
        validation_actual = torch.cat([y[0] for x, y in iter(val_dataloader)])

        # GPU 텐서를 CPU로 이동하여 계산
        val_predictions_np = validation_predictions.cpu().numpy()
        val_actual_np = validation_actual.cpu().numpy()

        rmse = np.sqrt(mean_squared_error(val_actual_np, val_predictions_np))
        r2 = r2_score(val_actual_np, val_predictions_np)

        print(f"Validation RMSE: {rmse:.2f}")
        print(f"Validation R²: {r2:.4f}")


    TFT_(data)