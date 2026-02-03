"""
Prediction de la survie d'un individu sur le Titanic
"""
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix

from dotenv import load_dotenv

load_dotenv()
JETON_API = os.environ.get("JETON_API", "")

if JETON_API.startswith("$"):
    print("API token has been configured properly")
else:
    print("API token has not been configured")

print(20 * "-")

os.chdir("/home/onyxia/work/application/")

TrainingData = pd.read_csv("data.csv")

parser = argparse.ArgumentParser(description="number of trees")
parser.add_argument(
    "--n_trees", type=int, default=20, help="Number of trees"
)
args = parser.parse_args()

TrainingData.head()

TrainingData["Ticket"].str.split("/").str.len()

TrainingData["Name"].str.split(",").str.len()

TrainingData.isnull().sum()


## Un peu d'exploration et de feature engineering

### Statut socioéconomique

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6)
)  # layout matplotlib 1 ligne 2 colonnes taile 16*8
fig1_pclass = sns.countplot(data=TrainingData, x="Pclass", ax=axes[0]).set_title(
    "fréquence des Pclass"
)
fig2_pclass = sns.barplot(
    data=TrainingData, x="Pclass", y="Survived", ax=axes[1]
).set_title("survie des Pclass")


### Age

sns.histplot(data=TrainingData, x="Age", bins=15, kde=False).set_title(
    "Distribution de l'âge"
)
plt.show()

## Encoder les données imputées ou transformées.
#N_TREES = 20
N_TREES = args.n_trees
print("n_trees : ", N_TREES)
print(20 * "-")

MAX_DEPTH = None
MAX_FEATURES = "sqrt"
JETON_API = "$trotskitueleski1917"

# SPLIT TRAIN/TEST --------------------------------

# On _split_ notre _dataset_ d'apprentisage
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.

y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# PIPELINE ----------------------------

# Définition des variables
numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

# Variables numériques
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

# Variables catégorielles
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

# Pipeline
pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=20)),
    ]
)


# splitting samples
y = TrainingData["Survived"]
X = TrainingData.drop("Survived", axis="columns")

# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
pd.concat([x_train, y_train], axis=1).to_csv("train.csv")
pd.concat([x_test, y_test], axis=1).to_csv("test.csv")


# Random Forest


# Ici demandons d'avoir 20 arbres
pipe.fit(x_train, y_train)

# score
rdmf_score = pipe.score(x_test, y_test)
print(
    f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation"
)

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(x_test)))
