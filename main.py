import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def main():
    #Read the individual data frames
    anp_df = pd.read_pickle(r'Data/anp.pickle')
    total_df.head()
    face_df = pd.read_pickle(r'Data/face.pickle')
    image_df = pd.read_pickle(r'Data/image_data.pickle')
    metrics_df = pd.read_pickle(r'Data/image_metrics.pickle')
    object_labels_df = pd.read_pickle(r'Data/object_labels.pickle')
    survey_df = pd.read_pickle(r'Data/survey.pickle')

    # Merge them based on the image_id so that we have a large data frame containing all the elements
    image_anp_frame = pd.merge(image_df, anp_df, how='inner', on='image_id')
    im_anp_obj_frame = pd.merge(image_anp_frame, object_labels_df, how='inner', on='image_id')
    im_anp_obj_face_frame = pd.merge(im_anp_obj_frame, face_df, how='inner', on='image_id')
    im_anp_obj_face_metrics_frame = pd.merge(im_anp_obj_face_frame, metrics_df, how='inner', on='image_id')
    survey_df['insta_user_id'] = pd.to_numeric(survey_df['insta_user_id'])
    im_anp_obj_face_metrics_frame['user_id'] =  pd.to_numeric(im_anp_obj_face_metrics_frame['user_id'])
    total_df = pd.merge(im_anp_obj_face_metrics_frame, survey_df, how='left', left_on='user_id', right_on='insta_user_id')


if __name__ == "__main__":
    main()
