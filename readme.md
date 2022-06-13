# keyword_recommender_system
- This is a variation of https://github.com/qordmlwls/reinforcement_recommender_system
- The difference between this repositery and reinforcement_recommender_system is that XAI method is applied to this project(reason for recommendation)
## Structure
- [mydataset]
  - behavioral_data.csv (behavioral data, keyword rating data for preparing training data)
  - mydf.csv (training dataset)
  - prediction_data_sample (prediction data)
  - mapping_df.csv (This data is for hasing keyword data)
  - fram_env.pkl, myembeddings.pickle (needed for training)
  
- [output]
  - force.png (XAI analysis result force plot)
  - keyword_recommendation_example.csv (prediction result; keyword recommendation list)
  
- [model] 
  - policy_net.pt (trained model)
  - model_config (trained model config)
  
- [modules]
  - module_for_data_preparation.py -> data preparation module
  - module_for_real_time_prediction.py -> model prediction module
  - module_for_train_recommendation_model.py -> model training module

- [test] : run module

- [RecNN] : training, predicting model library


## Shap output
![force](https://user-images.githubusercontent.com/43153661/169966279-76eac40d-e7cf-494e-938c-cbac3d4f9151.png)