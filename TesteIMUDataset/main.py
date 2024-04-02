import numpy as np
import pandas as pd
import os
from time import time
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from matplotlib import rcParams
le = preprocessing.LabelEncoder()
from numba import jit
import itertools
from seaborn import countplot,lineplot, barplot
from numba import jit
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from scipy.stats import randint as sp_randint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import matplotlib.style as style
style.use('ggplot')

import warnings
warnings.filterwarnings('ignore')
import gc
gc.enable()

print("packages loaded")

data = pd.read_csv('dataset/X_train.csv')
tr = pd.read_csv('dataset/X_train.csv')
sub = pd.read_csv('dataset/sample_submission.csv')
test = pd.read_csv('dataset/X_test.csv')
target = pd.read_csv('dataset/y_train.csv')
print ("Data is ready !!")

"""
Each series has 128 measurements.

1 serie = 128 measurements.

For example, serie with series_id=0 has a surface = fin_concrete and 128 measurements.
"""

totalt = data.isnull().sum().sort_values(ascending=False)
percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([totalt, percent], axis=1, keys=['Total', 'Percent'])
print ("Missing Data at Training")
missing_data.tail()

print ("Test has ", (test.shape[0]-data.shape[0])/128, "series more than Train (later I will prove it) = 768 registers")
dif = test.shape[0]-data.shape[0]
print ("Let's check this extra 6 series")
test.tail(768).describe()

sns.set(style='darkgrid')
sns.countplot(y = 'surface',
              data = target,
              order = target['surface'].value_counts().index)
plt.show()

serie1 = tr.head(128)
serie1.head()

plt.figure(figsize=(26, 16))
for i, col in enumerate(serie1.columns[3:]):
    plt.subplot(3, 4, i + 1)
    plt.plot(serie1[col])
    plt.title(col)

del serie1
gc.collect()

series_dict = {}
for series in (data['series_id'].unique()):
    series_dict[series] = data[data['series_id'] == series]

def plotSeries(series_id):
    style.use('ggplot')
    plt.figure(figsize=(28, 16))
    print(target[target['series_id'] == series_id]['surface'].values[0].title())
    for i, col in enumerate(series_dict[series_id].columns[3:]):
        if col.startswith("o"):
            color = 'red'
        elif col.startswith("a"):
            color = 'green'
        else:
            color = 'blue'
        if i >= 7:
            i+=1
        plt.subplot(3, 4, i + 1)
        plt.plot(series_dict[series_id][col], color=color, linewidth=3)
        plt.title(col)

id_series = 15
plotSeries(id_series)

del series_dict
gc.collect()

train_x = pd.read_csv('dataset/X_train.csv')
train_y = pd.read_csv('dataset/y_train.csv')

import math

def prepare_data(t):
    def f(d):
        d=d.sort_values(by=['measurement_number'])
        return pd.DataFrame({
         'lx':[ d['linear_acceleration_X'].values ],
         'ly':[ d['linear_acceleration_Y'].values ],
         'lz':[ d['linear_acceleration_Z'].values ],
         'ax':[ d['angular_velocity_X'].values ],
         'ay':[ d['angular_velocity_Y'].values ],
         'az':[ d['angular_velocity_Z'].values ],
        })

    t= t.groupby('series_id').apply(f)

    def mfft(x):
        return [ x/math.sqrt(128.0) for x in np.absolute(np.fft.fft(x)) ][1:65]

    t['lx_f']=[ mfft(x) for x in t['lx'].values ]
    t['ly_f']=[ mfft(x) for x in t['ly'].values ]
    t['lz_f']=[ mfft(x) for x in t['lz'].values ]
    t['ax_f']=[ mfft(x) for x in t['ax'].values ]
    t['ay_f']=[ mfft(x) for x in t['ay'].values ]
    t['az_f']=[ mfft(x) for x in t['az'].values ]
    return t

t=prepare_data(train_x)
t=pd.merge(t,train_y[['series_id','surface','group_id']],on='series_id')
t=t.rename(columns={"surface": "y"})

def aggf(d, feature):
    va= np.array(d[feature].tolist())
    mean= sum(va)/va.shape[0]
    var= sum([ (va[i,:]-mean)**2 for i in range(va.shape[0]) ])/va.shape[0]
    dev= [ math.sqrt(x) for x in var ]
    return pd.DataFrame({
        'mean': [ mean ],
        'dev' : [ dev ],
    })

display={
'hard_tiles_large_space':'r-.',
'concrete':'g-.',
'tiled':'b-.',

'fine_concrete':'r-',
'wood':'g-',
'carpet':'b-',
'soft_pvc':'y-',

'hard_tiles':'r--',
'soft_tiles':'g--',
}

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8 * 7))
# plt.margins(x=0.0, y=0.0)
# plt.tight_layout()
# plt.figure()

features = ['lx_f', 'ly_f', 'lz_f', 'ax_f', 'ay_f', 'az_f']
count = 0

for feature in features:
    stat = t.groupby('y').apply(aggf, feature)
    stat.index = stat.index.droplevel(-1)
    b = [*range(len(stat.at['carpet', 'mean']))]

    count += 1
    plt.subplot(len(features) + 1, 1, count)
    for i, (k, v) in enumerate(display.items()):
        plt.plot(b, stat.at[k, 'mean'], v, label=k)
        # plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)

    leg = plt.legend(loc='best', ncol=3, mode="expand", shadow=True, fancybox=True)
    plt.title("sensor: " + feature)
    plt.xlabel("frequency component")
    plt.ylabel("amplitude")

count += 1
plt.subplot(len(features) + 1, 1, count)
k = 'concrete'
v = display[k]
feature = 'lz_f'
stat = t.groupby('y').apply(aggf, feature)
stat.index = stat.index.droplevel(-1)
b = [*range(len(stat.at['carpet', 'mean']))]

plt.errorbar(b, stat.at[k, 'mean'], yerr=stat.at[k, 'dev'], fmt=v)
plt.title("sample for error bars (lz_f, surface concrete)")
plt.xlabel("frequency component")
plt.ylabel("amplitude")

plt.show()

del train_x, train_y
gc.collect()

# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1
def quaternion_to_euler(x, y, z, w):
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return X, Y, Z


def fe_step0(actual):
    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html
    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html
    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

    # Spoiler: you don't need this ;)

    actual['norm_quat'] = (
                actual['orientation_X'] ** 2 + actual['orientation_Y'] ** 2 + actual['orientation_Z'] ** 2 + actual[
            'orientation_W'] ** 2)
    actual['mod_quat'] = (actual['norm_quat']) ** 0.5
    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']
    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']
    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']
    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    return actual

data = fe_step0(data)
test = fe_step0(test)
print(data.shape)
data.head()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(18, 5))

ax1.set_title('quaternion X')
sns.kdeplot(data['norm_X'], ax=ax1, label="train")
sns.kdeplot(test['norm_X'], ax=ax1, label="test")

ax2.set_title('quaternion Y')
sns.kdeplot(data['norm_Y'], ax=ax2, label="train")
sns.kdeplot(test['norm_Y'], ax=ax2, label="test")

ax3.set_title('quaternion Z')
sns.kdeplot(data['norm_Z'], ax=ax3, label="train")
sns.kdeplot(test['norm_Z'], ax=ax3, label="test")

ax4.set_title('quaternion W')
sns.kdeplot(data['norm_W'], ax=ax4, label="train")
sns.kdeplot(test['norm_W'], ax=ax4, label="test")

plt.show()

# quarterions to euler angles

def fe_step1(actual):
    """Quaternions to Euler Angles"""

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual[
        'norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)

    actual['euler_x'] = nx
    actual['euler_y'] = ny
    actual['euler_z'] = nz
    return actual

data = fe_step1(data)
test = fe_step1(test)
print (data.shape)
data.head()

fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(15, 5))

ax1.set_title('Roll')
sns.kdeplot(data['euler_x'], ax=ax1, label="train")
sns.kdeplot(test['euler_x'], ax=ax1, label="test")

ax2.set_title('Pitch')
sns.kdeplot(data['euler_y'], ax=ax2, label="train")
sns.kdeplot(test['euler_y'], ax=ax2, label="test")

ax3.set_title('Yaw')
sns.kdeplot(data['euler_z'], ax=ax3, label="train")
sns.kdeplot(test['euler_z'], ax=ax3, label="test")

plt.show()


def feat_eng(data):
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X'] ** 2 + data['angular_velocity_Y'] ** 2 + data[
        'angular_velocity_Z'] ** 2) ** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X'] ** 2 + data['linear_acceleration_Y'] ** 2 + data[
        'linear_acceleration_Z'] ** 2) ** 0.5
    data['totl_xyz'] = (data['orientation_X'] ** 2 + data['orientation_Y'] ** 2 + data['orientation_Z'] ** 2) ** 0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))

    for col in data.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max']) / 2
    return df

data = feat_eng(data)
test = feat_eng(test)
print ("New features: ",data.shape)

from scipy.stats import kurtosis
from scipy.stats import skew


def _kurtosis(x):
    return kurtosis(x)


def CPT5(x):
    den = len(x) * np.exp(np.std(x))
    return sum(np.exp(x)) / den


def skewness(x):
    return skew(x)


def SSC(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x, x[1])
    xn = x[1:len(x) - 1]
    xn_i2 = x[2:len(x)]  # xn+1
    xn_i1 = x[0:len(x) - 2]  # xn-1
    ans = np.heaviside((xn - xn_i1) * (xn - xn_i2), 0)
    return sum(ans[1:])


def wave_length(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x, x[1])
    xn = x[1:len(x) - 1]
    xn_i2 = x[2:len(x)]  # xn+1
    return sum(abs(xn_i2 - xn))


def norm_entropy(x):
    tresh = 3
    return sum(np.power(abs(x), tresh))


def SRAV(x):
    SRA = sum(np.sqrt(abs(x)))
    return np.power(SRA / len(x), 2)


def mean_abs(x):
    return sum(abs(x)) / len(x)


def zero_crossing(x):
    x = np.array(x)
    x = np.append(x[-1], x)
    x = np.append(x, x[1])
    xn = x[1:len(x) - 1]
    xn_i2 = x[2:len(x)]  # xn+1
    return sum(np.heaviside(-xn * xn_i2, 0))


def fe_advanced_stats(data):
    df = pd.DataFrame()

    for col in data.columns:
        if col in ['row_id', 'series_id', 'measurement_number']:
            continue
        if 'orientation' in col:
            continue

        print("FE on column ", col, "...")

        df[col + '_skew'] = data.groupby(['series_id'])[col].skew()
        df[col + '_mad'] = data.groupby(['series_id'])[col].mad()
        df[col + '_q25'] = data.groupby(['series_id'])[col].quantile(0.25)
        df[col + '_q75'] = data.groupby(['series_id'])[col].quantile(0.75)
        df[col + '_q95'] = data.groupby(['series_id'])[col].quantile(0.95)
        df[col + '_iqr'] = df[col + '_q75'] - df[col + '_q25']
        df[col + '_CPT5'] = data.groupby(['series_id'])[col].apply(CPT5)
        df[col + '_SSC'] = data.groupby(['series_id'])[col].apply(SSC)
        df[col + '_skewness'] = data.groupby(['series_id'])[col].apply(skewness)
        df[col + '_wave_lenght'] = data.groupby(['series_id'])[col].apply(wave_length)
        df[col + '_norm_entropy'] = data.groupby(['series_id'])[col].apply(norm_entropy)
        df[col + '_SRAV'] = data.groupby(['series_id'])[col].apply(SRAV)
        df[col + '_kurtosis'] = data.groupby(['series_id'])[col].apply(_kurtosis)
        df[col + '_zero_crossing'] = data.groupby(['series_id'])[col].apply(zero_crossing)

    return df

basic_fe = ['linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z',
           'angular_velocity_X','angular_velocity_Y','angular_velocity_Z']


def fe_plus(data):
    aux = pd.DataFrame()
    X_train = pd.read_csv('dataset/X_train.csv')

    for serie in data.index:
        # if serie%500 == 0: print ("> Serie = ",serie)

        aux = X_train[X_train['series_id'] == serie]

        for col in basic_fe:
            data.loc[serie, col + '_unq'] = aux[col].round(3).nunique()
            data.loc[serie, col + 'ratio_unq'] = aux[col].round(3).nunique() / 18
            try:
                data.loc[serie, col + '_freq'] = aux[col].value_counts().idxmax()
            except:
                data.loc[serie, col + '_freq'] = 0

            data.loc[serie, col + '_max_freq'] = aux[aux[col] == aux[col].max()].shape[0]
            data.loc[serie, col + '_min_freq'] = aux[aux[col] == aux[col].min()].shape[0]
            data.loc[serie, col + '_pos_freq'] = aux[aux[col] >= 0].shape[0]
            data.loc[serie, col + '_neg_freq'] = aux[aux[col] < 0].shape[0]
            data.loc[serie, col + '_nzeros'] = (aux[col] == 0).sum(axis=0)

#https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
corr_matrix = data.corr().abs()
raw_corr = data.corr()

sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                 .stack()
                 .sort_values(ascending=False))
top_corr = pd.DataFrame(sol).reset_index()
top_corr.columns = ["var1", "var2", "abs corr"]
# with .abs() we lost the sign, and it's very important.
for x in range(len(top_corr)):
    var1 = top_corr.iloc[x]["var1"]
    var2 = top_corr.iloc[x]["var2"]
    corr = raw_corr[var1][var2]
    top_corr.at[x, "raw corr"] = corr

target['surface'] = le.fit_transform(target['surface'])

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=59)

predicted = np.zeros((test.shape[0],9))
measured= np.zeros((data.shape[0]))
score = 0

for times, (trn_idx, val_idx) in enumerate(folds.split(data.values, target['surface'].values)):
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    # model = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_split=5, n_jobs=-1)
    model.fit(data.iloc[trn_idx], target['surface'][trn_idx])
    measured[val_idx] = model.predict(data.iloc[val_idx])
    predicted += model.predict_proba(test) / folds.n_splits
    score += model.score(data.iloc[val_idx], target['surface'][val_idx])
    print("Fold: {} score: {}".format(times, model.score(data.iloc[val_idx], target['surface'][val_idx])))

    importances = model.feature_importances_
    indices = np.argsort(importances)
    features = data.columns

    """
    if model.score(data.iloc[val_idx], target['surface'][val_idx]) > 0.92000:
        hm = 30
        plt.figure(figsize=(7, 10))
        plt.title('Feature Importances')
        plt.barh(range(len(indices[:hm])), importances[indices][:hm], color='b', align='center')
        plt.yticks(range(len(indices[:hm])), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()
    """

    gc.collect()

print('Avg Accuracy RF', score / folds.n_splits)