import warnings

import matplotlib.pyplot as plt
import mne
import seaborn as sns
from mne.decoding import CSP
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)


import moabb
from moabb.datasets import BNCI2014_001, Zhou2016
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import LeftRightImagery


mne.set_log_level("CRITICAL")
moabb.set_log_level("info")
warnings.filterwarnings("ignore")


##############################################################################
# Initializing Datasets
# ---------------------
datasets = [Zhou2016(), BNCI2014_001()]
subj = [1, 2, 3]
for d in datasets:
    d.subject_list = subj


##############################################################################
# Pipeline
# ---------------------
pipelines = {}

# Ensemble model with voting classifier
estimators_voting = [
    ("svc", SVC(C=0.1, kernel="linear", probability=True)),
    ("gbc", GradientBoostingClassifier()),
]

# Create voting layer
voting_ensemble = VotingClassifier(estimators=estimators_voting, voting="soft")

pipelines["model ensemble voting classifier"] = make_pipeline(CSP(n_components=8), voting_ensemble)


# Ensemble model with Stacking classifier
estimators_stack = [
    ("svc", SVC(C=0.1, kernel="linear")),
    ("gbc", GradientBoostingClassifier()),
]
final_estimator = RandomForestClassifier(n_estimators=400, random_state=42)
stacking_ensemble = StackingClassifier(estimators=estimators_stack, final_estimator=final_estimator)

pipelines["model ensemble stacking classifier"] = make_pipeline(CSP(n_components=8), stacking_ensemble)


# SVC by its own
pipelines["csp+svc"] = make_pipeline(CSP(n_components=8), SVC(C=0.1, kernel="linear"))


# GradientBoostingClassifier by its own
pipelines["csp+gbc"] = make_pipeline(CSP(n_components=8), GradientBoostingClassifier())


##############################################################################
# Choose paradigm and evaluation
# ---------------------

paradigm = LeftRightImagery()

evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False)

results = evaluation.process(pipelines)


##############################################################################
# Plotting Results
# ----------------
results["subj"] = [str(resi).zfill(2) for resi in results["subject"]]

g = sns.catplot(
    kind="bar",
    x="score",
    y="subj",
    hue="pipeline",
    col="dataset",
    height=12,
    aspect=0.5,
    data=results,
    orient="h",
    palette="viridis",
)
plt.show()
