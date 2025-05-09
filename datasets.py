# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 09:47:07 2022

@author: Mert
"""
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from utils import one_hot_encode
from uci_datasets import Dataset

def load_dataset(
        dataset='icu',
        path='./data',
        ):

    fs, ys, sigma = [None] * int(1e6), [None] * int(1e6), [None] * int(1e6)

    if dataset.lower() == 'iowa':
        predictive_distribution = 'Normal'
        y_scale = 1e5
        train = pd.read_csv(
            path + '/iowa-housing/train.csv'
            )
        test = pd.read_csv(
            path + '/iowa-housing/test.csv'
            )
        drop = [
            'Id',
            'YearRemodAdd',
            'MasVnrArea',
            'BsmtUnfSF',
            'LowQualFinSF',
            'BsmtFullBath',
            'BsmtHalfBath',
            'GarageYrBlt',
            '3SsnPorch',
            'MiscVal',
            'BsmtFinSF1',
            'BsmtFinSF2',
            'LotFrontage',
            'GarageArea',
            '1stFlrSF',
            '2ndFlrSF',
            'Fireplaces',
            'MSSubClass',
            'WoodDeckSF',
            'OpenPorchSF',
            # 'BedroomAbvGr',
            'HalfBath',
            'OverallCond',
            # 'GrLivArea',
            'EnclosedPorch',
            'ScreenPorch',
            # 'PoolArea',
            # 'YearBuilt',
            'MoSold',
            'BldgType',
            'FullBath',
            'KitchenAbvGr',
            'TotalBsmtSF',
            'Neighborhood'
            ]

        no_drop = ['Neighborhood']

        df = pd.concat([train, test])
        #drop all other categorical feature that are not included in no drop or drop
        for feature in train.dtypes.keys():
            if feature not in no_drop:
                if feature not in drop:
                    if df.dtypes[feature] == 'O':
                        drop.append(feature)

        df = df.drop(columns=drop)
        #fill missing values
        for col_name in df.columns:
            try:
                df[col_name].fillna(df[col_name].median(), inplace=True)
            except:
                continue
        #the numberical categorical feature - turn to int then O
        categorical = [
            # 'YrSold', 'TotRmsAbvGrd', 'GarageCars', 'OverallQual'
            ]

        for category in categorical:
            try:
                df[category] = df[category].astype(np.int32)
            except:
                continue
            df[category] = df[category].astype('O')
        #one hot encode categorical features
        for key in df.keys():
            if df.dtypes[key] == 'O':
                df = one_hot_encode(df, key)
        #target and covariates are ready
        target = df['SalePrice']
        df = df.drop(columns=['SalePrice'])
        dtypes = np.asarray(list(df.dtypes))

        features = list(df.keys())
        target = target.values

        X = df.values
        y = target/y_scale

    elif dataset.lower() == 'medical':
        predictive_distribution = 'Normal'

        y_scale = 10000.0
        df = pd.read_csv(
            path +'/medical-cost/insurance.csv'
            )

        for col_name in df.columns:
            try:
                df[col_name].fillna(df[col_name].median(), inplace=True)
            except:
                continue

        for key in df.keys():
            if df.dtypes[key] == 'O':
                df = one_hot_encode(df, key)

        y = df['charges'].values / y_scale
        X = df.drop(columns=['charges'])

        features = list(X.keys())
        dtypes = np.asarray(list(X.dtypes))

        X = X.values

    elif dataset.lower() == 'life':
        predictive_distribution = 'Normal'

        y_scale = 10.0
        df = pd.read_csv(
            path +'/life-expectancy-v2/LIFEEXPECTANCY.csv'
            )
        df = df.drop(columns=['Location', 'Adolescent birth rate'])
        for col_name in df.columns:
            try:
                df[col_name].fillna(df[col_name].median(), inplace=True)
            except:
                continue

        categorical = []

        for category in categorical:
            try:
                df[category] = df[category].astype(np.int32)
            except:
                continue
            df[category] = df[category].astype('O')

        for key in df.keys():
            if df.dtypes[key] == 'O':
                df = one_hot_encode(df, key)

        y = df['Life expectancy'].values/y_scale
        X = df.drop(columns=['Life expectancy'])

        features = list(X.keys())
        dtypes = np.asarray(list(X.dtypes))
        X = X.values

    elif 'synthetic' in dataset.lower():
        predictive_distribution = 'Normal'
        y_scale = 1

        data = np.load(path + '/{}/data.npy'.format(dataset))
        try:
            fs = [
                np.load(path + '/{}/f1.npy'.format(dataset)),
                np.load(path + '/{}/f2.npy'.format(dataset)),
                np.load(path + '/{}/f3.npy'.format(dataset)),
                ]
            try:
                ys = [
                    np.load(path + '/{}/y1.npy'.format(dataset)),
                    np.load(path + '/{}/y2.npy'.format(dataset)),
                    np.load(path + '/{}/y3.npy'.format(dataset)),
                    ]
            except:
                pass
        except:
            pass

        y = data[:,0]
        X = data[:,1:]
        features = ['x1', 'x2', 'x3']
        dtypes = np.asarray([X.dtype] * X.shape[1])

    elif dataset.lower() == 'icu':

        y_scale = 1
        predictive_distribution = 'Bernoulli'

        df = pd.read_csv(path + '/icu-survival/training_v2.csv')

        keep = [
            'glucose_apache',
            'fio2_apache',
            'map_apache',
            'paco2_apache',
            'sodium_apache',
            'urineoutput_apache',
            'hospital_death',
            'd1_lactate_min',
            'd1_lactate_max',
            'gcs_eyes_apache',
            'gcs_verbal_apache',
            'intubated_apache',
            'bun_apache',
            'd1_inr_max',
            'd1_inr_min',
            'wbc_apache',
            'age',
            'heart_rate_apache',
            'bilirubin_apache',
            'creatinine_apache',
            'd1_resprate_max',
            'd1_resprate_min',
            'apache_3j_bodysystem',
            'd1_spo2_min',
            'd1_spo2_max',
            'temp_apache',
            'ph_apache'
            ]

        #choose relevant features
        df = df[keep]
        #fill missing with nan
        for col_name_max in df.columns:
            if 'max' in col_name_max:
                col_name_min = col_name_max.replace('_max','_min')
                col_name_avg = col_name_max.replace('_max','_avg')
                average = (df[col_name_max] + df[col_name_min])/2
                df[col_name_avg] = average
                df = df.drop(columns=[col_name_min])
                df = df.drop(columns=[col_name_max])

        for col_name in df.columns:
            try:
                df[col_name].fillna(df[col_name].median(), inplace=True)
            except:
                continue

        categorical = []
        for category in categorical:
            try:
                df[category] = df[category].astype(np.int32)
            except:
                continue
            df[category] = df[category].astype('O')
        #one hot encode categorical
        for key in df.keys():
            if df.dtypes[key] == 'O':
                df = one_hot_encode(df, key)
        #first frop target, tkae dtype then features
        target = df['hospital_death']
        df = df.drop(columns=['hospital_death'])
        features = list(df.keys())
        features = [key.replace('apache_','') for key in features]
        features = [key.replace('_apache','') for key in features]
        features = [key.replace('d1_','') for key in features]
        features = [key.replace('h1_','') for key in features]
        features = [key.replace('_avg','') for key in features]
        features = [key.replace('3j_bodysystem','diagnosis1') for key in features]
        features = [key.replace('2_bodysystem','diagnosis2') for key in features]
        features = [key.replace('gcs_','gcs-') for key in features]
        features = [key.replace('heart_rate','heart rate') for key in features]

        dtypes = np.asarray(list(df.dtypes))

        df = df.values
        target = target.values

        bias = np.ones(df.shape[0]).reshape(-1,1)

        X = df
        y = target

        X1 = X[y==1]
        X0 = X[y==0]

        size = X1.shape[0]

        choice = np.random.choice(X0.shape[0], size=size, replace=False)

        X0 = X0[choice]

        #sub-sample
        X = np.concatenate([X1,X0])
        y = np.concatenate([np.ones(X1.shape[0]),np.zeros(X0.shape[0])])

    elif dataset.lower() == 'mimic':
        y_scale = 1
        predictive_distribution = 'Bernoulli'

        df = pd.read_csv('./data/mimic/mimic_cad_.csv')
        df = df.drop(
            columns=['Unnamed: 0', 'stay_id', 'subject_id',
                     'hadm_id', 'time_delta','censoring'
                     ]
            )
        y = df['time_to_event (D)']
        X = df.drop(columns=['time_to_event (D)'])

        scale = ((X - X.min(0) )/ (X.max(0) - X.min(0))).std(0)
        scale = scale[~scale.isna()]
        scale = scale[scale>=0.1]
        X = X[scale.index]
        dtypes = np.asarray(list(X.dtypes))
        features = list(X)
        X = X.values
        y = y.values

    elif dataset.lower() == 'fico':
        y_scale = 1
        predictive_distribution = 'Bernoulli'

        df = pd.read_csv(
            './data/fico-score/fico_score.data'
            )
        df['RiskPerformance'] = [
            1 if x == 'Good' else 0 for x in df['RiskPerformance']
            ]
        df[df<0] = np.nan

        for col_name in df.columns:
            df[col_name].fillna(df[col_name].median(), inplace=True)

        categorical = [
            'MaxDelq2PublicRecLast12M',
            'MaxDelqEver'
            ]

        mdlq2prl12m = {
            0:'derogatory comment',
            1:'120+ days delinquent',
            2:'90 days delinquent',
            3:'60 days delinquent',
            4:'30 days delinquent',
            5:'unknown delinquent',
            6:'unknown delinquent',
            7:'current and never delinquent',
            8:'all other',
            9:'all other'
            }

        maxdlqever = {
            1:'No such value',
            2:'derogatory comment',
            3:'120+ days delinquent',
            4:'90 days delinquent',
            5:'60 days delinquent',
            6:'30 days delinquent',
            7:'unknown delinquency',
            8:'current and never deliquent',
            9:'all other'
            }

        df['MaxDelq2PublicRecLast12M'] = [
            mdlq2prl12m[x] for x in df['MaxDelq2PublicRecLast12M']
            ]
        df['MaxDelqEver'] = [
            maxdlqever[x] for x in df['MaxDelqEver']
            ]

        for category in categorical:
            try:
                df[category] = df[category].astype(np.int32)
            except:
                continue
            df[category] = df[category].astype('O')

        for key in df.keys():
            if df.dtypes[key] == 'O':
                df = one_hot_encode(df, key)

        y = df['RiskPerformance'].values
        X = df.drop(columns=['RiskPerformance'])
        X = X.drop(columns=['ExternalRiskEstimate'])

        features = list(X.keys())
        dtypes = np.asarray(list(X.dtypes))
        X = X.values

    elif dataset.lower() == 'spambase':
        y_scale = 1
        predictive_distribution = 'Bernoulli'
        df = pd.read_csv(path + '/spambase/spambase.data', header=None)
        y = df[57].values
        X = df.drop(columns=[57])
        features = list(pd.read_csv(
            path + '/spambase/features.csv', header=None
            )[0].values)
        dtypes = np.asarray(list(X.dtypes))
        X = X.values

    elif dataset.lower() == 'adult':

        y_scale = 1
        predictive_distribution = 'Bernoulli'

        df = pd.read_csv(
            path + '/adult-income/adult.data', header=None
            )
        df.columns = [
            'Age',
            'WorkClass',
            'fnlwgt',
            'Education',
            'EducationNum',
            'MaritalStatus',
            'Occupation',
            'Relationship',
            'Race',
            'Gender',
            'CapitalGain',
            'CapitalLoss',
            'HoursPerWeek',
            'NativeCountry',
            'Income'
            ]

        train_cols = df.columns[0:-1]
        label = df.columns[-1]
        x_df = df[train_cols]
        y_df = df[label]

        for key in x_df.keys():
            if x_df.dtypes[key] == 'O':
                x_df = one_hot_encode(x_df, key)

        dtypes = np.asarray(list(x_df.dtypes))

        features = list(x_df.columns)

        X = x_df.values
        y = y_df.values

        y = (y == ' >50K').astype(np.float32)

    elif 'mnist_normal' in dataset.lower():

        detect = int(dataset.split('_')[-1])
        print(detect)
        y_scale = 1
        predictive_distribution = 'Bernoulli'

        train_images = np.load('./data/mnist/train_mnist_10_segments_normal_images.npy')
        train_inputs = np.load('./data/mnist/train_mnist_10_segments_normal_inputs.npy')
        train_labels = np.load('./data/mnist/train_mnist_10_segments_normal_labels.npy')

        train_labels = 1. * (train_labels == detect)

        test_images = np.load('./data/mnist/test_mnist_10_segments_normal_images.npy')
        test_inputs = np.load('./data/mnist/test_mnist_10_segments_normal_inputs.npy')
        test_labels = np.load('./data/mnist/test_mnist_10_segments_normal_labels.npy')

        test_labels = 1. * (test_labels == detect)

        X = np.concatenate([train_inputs, test_inputs])
        y = np.concatenate([train_labels, test_labels])
        fs = np.concatenate([train_images, test_images])

        choice = np.random.choice(X.shape[0], X.shape[0]//5, replace=False)

        X = X[choice, :]
        y = y[choice]
        fs = fs[choice, :]

        dtypes = np.asarray([X.dtype for i in range(X.shape[-1])])
        features = ['x{}'.format(i) for i in range(X.shape[-1])]

    elif 'mnist_cb' in dataset.lower():

        detect = int(dataset.split('_')[-1])
        print(detect)
        y_scale = 1
        predictive_distribution = 'Bernoulli'

        train_images = np.load('./data/mnist/train_mnist_10_segments_cb_images.npy')
        train_inputs = np.load('./data/mnist/train_mnist_10_segments_cb_inputs.npy')
        train_labels = np.load('./data/mnist/train_mnist_10_segments_cb_labels.npy')

        train_labels = 1. * (train_labels == detect)

        test_images = np.load('./data/mnist/test_mnist_10_segments_cb_images.npy')
        test_inputs = np.load('./data/mnist/test_mnist_10_segments_cb_inputs.npy')
        test_labels = np.load('./data/mnist/test_mnist_10_segments_cb_labels.npy')

        test_labels = 1. * (test_labels == detect)

        X = np.concatenate([train_inputs, test_inputs])
        y = np.concatenate([train_labels, test_labels])
        fs = np.concatenate([train_images, test_images])

        choice = np.random.choice(X.shape[0], X.shape[0]//5, replace=False)

        X = X[choice, :]
        y = y[choice]
        fs = fs[choice, :]

        dtypes = np.asarray([X.dtype for i in range(X.shape[-1])])
        features = ['x{}'.format(i) for i in range(X.shape[-1])]


    else:
        y_scale = 1
        predictive_distribution = 'Normal'

        data = Dataset(dataset)
        x_train, y_train, x_test, y_test = data.get_split(split=0)
        X = np.concatenate([x_train, x_test])
        y = np.concatenate([y_train, y_test]).reshape(-1)
        y = (y - y.mean()) / (y.std() + 1e-5)
        uniques = [len(np.unique(x, axis=0)) for x in X.T]
        uniques_ = [len(np.unique(x, axis=0)) for x in X.astype(np.int64).T]
        dtypes = [u == u_ for (u, u_) in zip(uniques, uniques_)]
        dtypes_ = [u_ <= 5 for u_ in uniques_]
        dtypes = np.asarray(dtypes) * np.asarray(dtypes_) * (
            np.sum(X.astype(np.int64).T - X.T, 1) == 0
            )
        dtypes = np.asarray(
            [
                np.dtype(
                    np.float64
                    ) if not d else np.dtype(
                        np.int64
                        ) for d in dtypes
                        ]
                        )
        features = ['x{}'.format(i) for i in range(X.shape[-1])]

    return (
        shuffle(X, y), (X, y, fs, ys, sigma),
        ), features, dtypes, y_scale, predictive_distribution
