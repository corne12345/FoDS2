import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from random import shuffle

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

# Function that takes a string of income and transforms it to its averae
def income_from_string(string):
    new = ''.join([c for c in string if (c.isdigit() or c=='$')])
    values = [int(i) for i in new.split('$') if len(i) > 0]
    try:
        average = sum(values)/len(values)
    except:
        average = 999999
    return average

# apply function of income_from_string
def income_transform(newdf, total_df):
    newdf['income'] = total_df['income'].apply(income_from_string)
    return newdf

def one_hot_encode (newdf, totaldf, column, drop_first=False):
    newdf = pd.concat([newdf, pd.get_dummies(totaldf[column], drop_first=drop_first)], axis=1)
    return newdf

# Function to transfer the end and start time of the questionnaire into a duration. Output is addidition to newdf
def duration_questionnaire(newdf, totaldf):
    totaldf['end_q'] = pd.to_datetime(totaldf['end_q'])
    totaldf['start_q'] = pd.to_datetime(totaldf['start_q'])
    newdf['dur_quest'] = totaldf['end_q'] - totaldf['start_q']
    return newdf

# Function to convert birthyear into age
def born_to_age (newdf, total_df):
    newdf['age'] = - (total_df['born'] - 2019)
    return newdf

# Function to create a correlation matrix, which may be helpful to find interactions
def create_correlation_matrix(newdf):
    correlation_matrix = newdf.corr()
    fig = plt.figure()
    names = correlation_matrix.columns
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlation_matrix, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks=np.arange(0,newdf.shape[1])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names, rotation='vertical')
    ax.set_yticklabels(names)
    plt.show()

# Function that performs linear regression on a given dataframe and returns coefficients and R-squared
def linear_regression(total_df, y, method="LR"):
    total_df['y'] = y
    total_df = total_df.dropna(axis=0)
    y = total_df['y']
    X = total_df.drop(columns=['y'])
#     X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_test, y_train, y_test = split_according_to_user(X,y)
    print(X_train.shape)
    if method == "LR":
        lr = LinearRegression().fit(X_train, y_train)
    if method == "Lasso":
        lr = Lasso().fit(X_train, y_train)
    if method == "Ridge":
        lr = Ridge().fit(X_train, y_train)
    pred_y_train = lr.predict(X_train)
    pred_y_test = lr.predict(X_test)
    total_df['pred_y'] = lr.predict(X.drop(columns='y'))
    total_df['true_y'] = y
    print('Variance score on training set: %.2f' % r2_score(y_train, pred_y_train))
    print('Variance score on test set: %.2f' % r2_score(y_test, pred_y_test))
    print('Coefficients:\n', lr.coef_)
    return X

# Function to split Dataframe according to user and not totally random
def split_according_to_user(X,y, part = 0.8):
    users = list(X['user_id'].unique())
    shuffle(users)
    trainers = users[:round(part * len(users))]
    testers = users[round(part * len(users)):]
    print(len(trainers), len(testers))
    X['y'] = y
    X_train = X[X['user_id'].isin(trainers)]
    X_test = X[X['user_id'].isin(testers)]
    y_train = X_train['y']
    y_test = X_test['y']
    X_train = X_train.drop(columns='y')
    X_test = X_test.drop(columns='y')
    print(X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test

# Function that returns the complete dataframe.
def getCompleteDF():
    #Read the individual data frames
    anp_df = pd.read_pickle(r'Data/anp.pickle').sort_values('emotion_score', ascending=False).drop_duplicates(['image_id'])
    face_df = pd.read_pickle(r'Data/face.pickle').sort_values('emo_confidence', ascending=False).drop_duplicates(['image_id'])
    image_df = pd.read_pickle(r'Data/image_data.pickle')
    metrics_df = pd.read_pickle(r'Data/image_metrics.pickle').sort_values('like_count', ascending=False).drop_duplicates(['image_id'])
    object_labels_df = pd.read_pickle(r'Data/object_labels.pickle').sort_values('data_amz_label_confidence', ascending=False).drop_duplicates(['image_id'])
    survey_df = pd.read_pickle(r'Data/survey.pickle')

    # Merge them based on the image_id so that we have a large data frame containing all the elements
    image_anp_frame = pd.merge(image_df, anp_df, how='inner', on='image_id')
    im_anp_obj_frame = pd.merge(image_anp_frame, object_labels_df, how='inner', on='image_id')
    im_anp_obj_face_frame = pd.merge(im_anp_obj_frame, face_df, how='inner', on='image_id')
    im_anp_obj_face_metrics_frame = pd.merge(im_anp_obj_face_frame, metrics_df, how='inner', on='image_id')
    survey_df['insta_user_id'] = pd.to_numeric(survey_df['insta_user_id'])
    im_anp_obj_face_metrics_frame['user_id'] =  pd.to_numeric(im_anp_obj_face_metrics_frame['user_id'])
    total_df = pd.merge(im_anp_obj_face_metrics_frame, survey_df, how='inner', left_on='user_id', right_on='insta_user_id')

    return total_df

# Function that returns dataframe consisting of just the useful columns.
def getUsefulColumnsDF(total_df):
    newdf = pd.DataFrame()
    newdf = addRatioValues(newdf, total_df, 'image_height', 'image_width', 'image_ratio')
    newdf = addRatioValues(newdf, total_df, 'user_followed_by', 'user_follows', 'popularity')
    newdf = addMeanValues(newdf, total_df, 'face_age_range_high', 'face_age_range_low', 'face_age_mean')
    newdf = duration_questionnaire(newdf, total_df)
    newdf = one_hot_encode(newdf, total_df, column='image_filter')
    newdf = one_hot_encode(newdf, total_df, column='face_gender', drop_first=True)
    newdf = one_hot_encode(newdf, total_df, column='education')
    newdf = one_hot_encode(newdf, total_df, column='employed')
    newdf = one_hot_encode(newdf, total_df, column='gender', drop_first=True)
    newdf = one_hot_encode(newdf, total_df, column='participate', drop_first=True)
    newdf = income_transform(newdf, total_df)
    newdf = copyColumnValues(newdf, total_df, 'data_memorability')
    newdf = copyColumnValues(newdf, total_df, 'user_followed_by')
    newdf = copyColumnValues(newdf, total_df, 'user_follows')
    newdf = copyColumnValues(newdf, total_df, 'user_posted_photos')
    newdf = copyColumnValues(newdf, total_df, 'comment_count')
    newdf = copyColumnValues(newdf, total_df, 'like_count')
    newdf = copyColumnValues(newdf, total_df, 'HAP')
    newdf = copyColumnValues(newdf, total_df, 'imagecount')
    newdf = duration_questionnaire(newdf, total_df)
    newdf = income_transform(newdf, total_df)
    newdf = born_to_age(newdf, total_df)
    return newdf

# Main function.
def main():
    total_df = getCompleteDF()
    usable_df = getUsefulColumnsDF(total_df)
    linear_regression(usable_df, total_df['PERMA'])

if __name__ == "__main__":
    main()
