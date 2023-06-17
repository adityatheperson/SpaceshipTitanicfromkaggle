import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from tpot import TPOTClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
passengerid = test["PassengerId"]
enconders = {}


def clean_test(data):
    data = data.drop(["Name", "PassengerId"], axis=1)
    # Dropping useless columns

    data["Cabin_Deck"] = data["Cabin"].str.slice(0, 1)
    data["Cabin_SP"] = data["Cabin"].str.split(pat="/")
    data["Cabin_SP"] = data["Cabin_SP"].str[2]
    data = data.drop("Cabin", axis=1)
    # Splitting the Cabin column to make it more usable

    data.Age.fillna(data["Age"].mean(), inplace=True)
    data.RoomService.fillna(0, inplace=True)
    data.FoodCourt.fillna(0, inplace=True)
    data.Spa.fillna(0, inplace=True)
    data.ShoppingMall.fillna(0, inplace=True)
    data.VRDeck.fillna(0, inplace=True)
    data.HomePlanet.fillna("U", inplace=True)
    data.Destination.fillna("U", inplace=True)
    data.Cabin_Deck.fillna("U", inplace=True)
    data.Cabin_SP.fillna("U", inplace=True)
    # filling the empty spaces in the columns with a mean or 0 or random value
    data["VIP"] = data["VIP"].replace(True, 2)
    data["VIP"] = data["VIP"].replace(False, 1)
    data.VIP.fillna(3, inplace=True)

    data["CryoSleep"] = data["CryoSleep"].replace(True, 2)
    data["CryoSleep"] = data["CryoSleep"].replace(False, 1)
    data.CryoSleep.fillna(3, inplace=True)

    print(data.shape)
    data.dropna()
    print(data.shape)

    cols = ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_SP"]

    le = preprocessing.LabelEncoder()

    for col in cols:
        data[col] = enconders[col].transform(data[col])
    # Using a Label Encoder to change all the strings into numerical values

    return data


def clean(data):
    data = data.drop(["Name", "PassengerId"], axis=1)
    # Dropping useless columns

    data["Cabin_Deck"] = data["Cabin"].str.slice(0, 1)
    data["Cabin_SP"] = data["Cabin"].str.split(pat="/")
    data["Cabin_SP"] = data["Cabin_SP"].str[2]
    data = data.drop("Cabin", axis=1)
    # Splitting the Cabin column to make it more usable

    data.HomePlanet.fillna("U", inplace=True)
    data.Destination.fillna("U", inplace=True)
    data.Cabin_Deck.fillna("U", inplace=True)
    data.Cabin_SP.fillna("U", inplace=True)
    data.Age.fillna(data["Age"].mean(), inplace=True)
    data.RoomService.fillna(0, inplace=True)
    data.FoodCourt.fillna(0, inplace=True)
    data.Spa.fillna(0, inplace=True)
    data.ShoppingMall.fillna(0, inplace=True)
    data.VRDeck.fillna(0, inplace=True)

    data["VIP"] = data["VIP"].replace(True, 2)
    data["VIP"] = data["VIP"].replace(False, 1)
    data.VIP.fillna(3, inplace=True)

    data["CryoSleep"] = data["CryoSleep"].replace(True, 2)
    data["CryoSleep"] = data["CryoSleep"].replace(False, 1)
    data.CryoSleep.fillna(3, inplace=True)

    # filling the empty spaces in the columns with a mean or 0

    print(data.shape)
    data = data.dropna()
    print(data.shape)
    # checking how much data was lost

    cols = ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_SP"]

    le = preprocessing.LabelEncoder()

    for col in cols:
        le = preprocessing.LabelEncoder()
        enconders[col] = le.fit(data[col])
        data[col] = enconders[col].transform(data[col])
    # Using a Label Encoder to change all the strings into numerical values

    return data


cleaned_data = clean(data)
cleaned_test = clean_test(test)

# cleaning the data

y = cleaned_data["Transported"]
X = cleaned_data.drop(["Transported"], axis=1)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
cleaned_test = scaler.transform(cleaned_test)
# scaling the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# making the train and validation split


def testmodel(model):
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)


# Making a function to test accuracy easily


logisticregressionmodel = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
# a simple logistic Regression model, accuracy score -> 0.7719638242894057

randomforestmodel = RandomForestClassifier(max_depth=15, random_state=0)
randomforestmodel.fit(X_train, y_train)
# a random forest classifier model, accuracy score -> 0.8029715762273901


# param_grid = {
#     "max_depth": [3, 4, 5, 7],
#     "learning_rate": [0.1, 0.01, 0.05],
#     "gamma": [0, 0.25, 1],
#     "reg_lambda": [0, 1, 10],
#     "scale_pos_weight": [1, 3, 5],
#     "subsample": [0.8],
#     "colsample_bytree": [0.5],
# }
# xgb_cl = xgb.XGBClassifier(objective="binary:logistic")
# grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")
# _ = grid_cv.fit(X_train, y_train)
# print(grid_cv.best_score_)
# print(grid_cv.best_params_)
# trying to find the best XGBoost configurations


xgbmodel = xgb.XGBClassifier(colsample_bytree=0.5, gamma=1, learning_rate=0.1, max_depth=5, reg_lambda=1,
                             scale_pos_weight=1, subsample=0.8)
xgbmodel.fit(X_train, y_train)
# an XGBClassifier with optimized settings, accuracy score -> 0.8087855297157622

from sklearn.neural_network import MLPClassifier

mlpmodel = MLPClassifier(hidden_layer_sizes=(8, 8, 8), activation='relu', solver='adam', max_iter=500)
mlpmodel.fit(X_train, y_train)

# model = TPOTClassifier(generations=5, population_size=20, cv=5,
#                        random_state=42, verbosity=2)
# model.fit(X_train, y_train)
# print(model.score(X_val, y_val))
# model.export('tpot_exported_pipeline.py')
# trying to find the best model + settings, using tpot.


extratreesmodel = ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001,
                                       min_samples_leaf=7,
                                       min_samples_split=6, n_estimators=100)
extratreesmodel.fit(X_train, y_train)
# 'best' model found using tpot, accuracy score -> 0.810077519379845


print("Logistic Regression --> " + str(testmodel(logisticregressionmodel)))
print("Random Forest --> " + str(testmodel(randomforestmodel)))
print("XGBoost --> " + str(testmodel(xgbmodel)))
print("Extra Trees --> " + str(testmodel(extratreesmodel)))
print("MLP --> "+ str(testmodel(mlpmodel)))

# submission_pred = extratreesmodel.predict(cleaned_test)
# df = pd.DataFrame({"PassengerId": passengerid.values,
#                    "Transported": submission_pred})
#
# df.to_csv("submission.csv", index=False)
