import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

class EncodingAndScalingClass:
    def read_file(self, file_path):
        df = pd.read_csv(file_path)
        df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)
        return df

    def split_df_to_X_y(self, df):
        X = df.drop(columns=['Price'])
        y = df['Price']
        return X, y

    def train_test_split(self, X, y):
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def encoding_and_scaling(self):
        trf1 = ColumnTransformer([
            ('OneHot', OneHotEncoder(drop='first', handle_unknown='ignore'), [0, 1, 2])
        ], remainder='passthrough')

        trf2 = ColumnTransformer([
            ('Ordinal', OrdinalEncoder(categories=[['non-stop', '1 stop', '2 stops', '3 stops', '4 stops']]), [16])
        ], remainder='passthrough')

        trf3 = ColumnTransformer([
            ('scale', StandardScaler(), slice(25))  # Scale first 25 columns
        ])
        
        return make_pipeline(trf1, trf2, trf3)
    
    def transform_X_train(self, pipe, X_train):
        return pd.DataFrame(pipe.fit_transform(X_train))
    
    def transform_X_test(self, pipe, X_test):
        return pd.DataFrame(pipe.transform(X_test))
    
    def save_dataframe(self, df, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)

if __name__ == "__main__":
    file_path = "Data/03_noOutlierData/noOutlierDataFile.csv"
    obj = EncodingAndScalingClass()
    df = obj.read_file(file_path)
    X, y = obj.split_df_to_X_y(df)
    X_train, X_test, y_train, y_test = obj.train_test_split(X, y)
    
    pipe = obj.encoding_and_scaling()
    X_train_transformed = obj.transform_X_train(pipe, X_train)
    X_test_transformed = obj.transform_X_test(pipe, X_test)
    
    obj.save_dataframe(X_train_transformed, "Data/04_encoded_Data/X_train.csv")
    obj.save_dataframe(X_test_transformed, "Data/04_encoded_Data/X_test.csv")
    obj.save_dataframe(y_train, "Data/04_encoded_Data/y_train.csv")
    obj.save_dataframe(y_test, "Data/04_encoded_Data/y_test.csv")
