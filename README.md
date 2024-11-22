# sellpartner-sales-prediction

data 로드 성공
data 전처리 성공
데이터 필터링 전:
전체 제품 수: 453

시퀀스 길이 통계:
최소 시퀀스 길이: 2
최대 시퀀스 길이: 310
평균 시퀀스 길이: 51.25

조정된 최소 필요 데이터 길이: 9
충분한 데이터를 가진 제품 수: 431
포함된 제품 비율: 95.14%
data 학습 데이터로 변환 성공


모델 학습 시작...

   | Name                               | Type                            | Params | Mode 
------------------------------------------------------------------------------------------------
0  | loss                               | RMSE                            | 0      | train
1  | logging_metrics                    | ModuleList                      | 0      | train
2  | input_embeddings                   | MultiEmbedding                  | 13.8 K | train
3  | prescalers                         | ModuleDict                      | 736    | train
4  | static_variable_selection          | VariableSelectionNetwork        | 6.0 K  | train
5  | encoder_variable_selection         | VariableSelectionNetwork        | 44.9 K | train
6  | decoder_variable_selection         | VariableSelectionNetwork        | 42.3 K | train
7  | static_context_variable_selection  | GatedResidualNetwork            | 4.3 K  | train
8  | static_context_initial_hidden_lstm | GatedResidualNetwork            | 4.3 K  | train
9  | static_context_initial_cell_lstm   | GatedResidualNetwork            | 4.3 K  | train
10 | static_context_enrichment          | GatedResidualNetwork            | 4.3 K  | train
11 | lstm_encoder                       | LSTM                            | 8.4 K  | train
12 | lstm_decoder                       | LSTM                            | 8.4 K  | train
13 | post_lstm_gate_encoder             | GatedLinearUnit                 | 2.1 K  | train
14 | post_lstm_add_norm_encoder         | AddNorm                         | 64     | train
15 | static_enrichment                  | GatedResidualNetwork            | 5.3 K  | train
16 | multihead_attn                     | InterpretableMultiHeadAttention | 3.2 K  | train
17 | post_attn_gate_norm                | GateAddNorm                     | 2.2 K  | train
18 | pos_wise_ff                        | GatedResidualNetwork            | 4.3 K  | train
19 | pre_output_gate_norm               | GateAddNorm                     | 2.2 K  | train
20 | output_layer                       | Linear                          | 33     | train
------------------------------------------------------------------------------------------------
159 K     Trainable params
0         Non-trainable params
159 K     Total params
0.639     Total estimated model params size (MB)
772       Modules in train mode
0         Modules in eval mode
