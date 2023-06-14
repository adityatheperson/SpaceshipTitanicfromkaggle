import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from tpot import TPOTClassifier

data = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")


def clean(data):
    data = data.drop(["Name", "PassengerId"], axis=1)

    data.Age.fillna(data["Age"].mean(), inplace=True)
    data.RoomService.fillna(0, inplace=True)
    data.FoodCourt.fillna(0, inplace=True)
    data.Spa.fillna(0, inplace=True)
    data.ShoppingMall.fillna(0, inplace=True)
    data.VRDeck.fillna(0, inplace=True)

    data["Cabin_Deck"] = data["Cabin"].str.slice(0, 1)

    data["Cabin_SP"] = data["Cabin"].str.split(pat="/")
    data["Cabin_SP"] = data["Cabin_SP"].str[2]

    data = data.drop("Cabin", axis=1)
    print(data.shape)

    data = data.dropna()
    print(data.shape)

    le = preprocessing.LabelEncoder()

    cols = ["HomePlanet", "Destination", "Cabin_Deck", "Cabin_SP"]

    for col in cols:
        data[col] = le.fit_transform(data[col])
        print(le.classes_)

    cols = ["VIP", "CryoSleep"]
    for col in cols:
        data[col] = data[col].astype(int)


    # cols = ["RoomService", "Age", "Spa", "ShoppingMall", "VRDeck", "FoodCourt"]
    #
    # for col in cols:
    #     data[col] = scaler.fit_transform(data[col])
    #
    return data


cleaned_data = clean(data)

y = cleaned_data["Transported"]
X = cleaned_data.drop(["Transported"], axis=1)

X = preprocessing.StandardScaler().fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)


# model = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)


def testmodel(model):
    predictions = model.predict(X_val)
    return accuracy_score(y_val, predictions)


# clf = RandomForestClassifier(max_depth=10, random_state=0)
# clf.fit(X_train, y_train)
#
# model = xgb.XGBClassifier(objective="binary:logistic", colsample_bytree=0.5,
#                           gamma=1,
#                           learning_rate=0.1,
#                           max_depth=4,
#                           reg_lambda=1,
#                           scale_pos_weight=1,
#                           subsample=0.8)
# model.fit(X_train, y_train)

#
# model = TPOTClassifier(generations=5, population_size=20, cv=5,
#                        random_state=42, verbosity=2)
#
# model.fit(X_train, y_train)
# print(model.score(X_val, y_val))
# model.export('tpot_exported_pipeline.py')


model = ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.7500000000000001, min_samples_leaf=7, min_samples_split=6, n_estimators=100)
model.fit(X_train, y_train)
print(testmodel(model))

# Test