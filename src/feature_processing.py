import pandas as pd
from src import config
from sklearn.preprocessing import LabelEncoder
import joblib
def feature_processing(data, mode):
    df = pd.read_csv(data)
    df.head()
    df.info()

    # feature imputation/ removing null value from society, bhk, bath, balcony

    df.society.fillna("No_society",inplace=True)
    df.bhk.fillna(0,inplace=True)
    df.bath.fillna(1,inplace=True)
    df.balcony.fillna(0,inplace=True)
    df.info()

    # Label Encoding for categorical values

    print(df['area_type'].unique())
    print(df['availability'].unique())
    print(df['location'].unique())
    print(df['society'].unique())

    labelencoder={}
    for c in ["area_type", "availability", "location", "society" ]:
        labelencoder[c] = LabelEncoder()
        if mode == 'train':
            df[c] = labelencoder[c].fit_transform(df[c])
        else:
            labelencoder = joblib.load(config.MODELS_PATH + "feature_encoders.pkl")
            df[c] = labelencoder[c].transform(df[c])


    df.to_csv(config.TRAIN_PROCESSED_DATA, index=False)

    if mode == 'train':
        joblib.dump(labelencoder,config.MODELS_PATH+'feature_encoders.pkl')

feature_processing(config.TRAIN_DATA ,'train' )
