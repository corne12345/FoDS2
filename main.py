import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Function that adds ratio values to new dataframe (f.a.: c = a / b for some
# arbitrary a, b and c).
def addRatioValues(newdf, olddf, a, b, c):
    newdf[c] = olddf[a] / olddf[b]
    return newdf

# Function that copies values from some arbitrary column "a" to new dataframe.
def copyColumnValues(newdf, olddf, a):
    newdf[a] = olddf[a]
    return newdf

# Function that add mean values of arbitrary columns "a" and "b" to new
# dataframe (f.a.: c = (a + b) / 2).
def addMeanValues(newdf, olddf, a, b, c):
    newdf[c] = (olddf[a] + olddf[b]) / 2
    return newdf

# Function that returns the complete dataframe.
def getCompleteDF():
    #Read the individual data frames
    anp_df = pd.read_pickle(r'Data/anp.pickle')
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

    return total_df

# Function that returns dataframe consisting of just the useful columns.
def getUsefulColumnsDF(total_df):
    newdf = pd.DataFrame()
    newdf = addRatioValues(newdf, total_df, 'image_height', 'image_width', 'image_ratio')
    print(newdf.head())
    return newdf

# Main function.
def main():
    total_df = getCompleteDF()
    usable_df = getUsefulColumnsDF(total_df)

if __name__ == "__main__":
    main()
