"""Reproduce the Titanic feature engineering experiments and figures."""
import argparse
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder,
                                   StandardScaler, MinMaxScaler, RobustScaler)

SEED = 42
BASE = ["Pclass", "SibSp", "Parch", "Fare"]
NUM = ["Age", "SibSp", "Parch", "Fare"]
CAT = ["Pclass", "Sex", "Embarked"]
DERIVED = ["FamilySize", "FarePerPerson"]
# (numeric imputation, categorical encoding, numeric scaling, top-k, derived)
SPECS = {
    "Base": None,
    "Exp-1": ("mean", "one-hot", "standard", "all", True),
    "Exp-2": ("median", "ordinal", "minmax", 6, True),
    "Exp-3": ("most_frequent", "one-hot", "robust", 6, True),
    "Mean-to-median": ("median", "one-hot", "standard", "all", True),
    "Mean-to-mode": ("most_frequent", "one-hot", "standard", "all", True),
    "One-hot-to-ordinal": ("mean", "ordinal", "standard", "all", True),
    "Standard-to-MinMax": ("mean", "one-hot", "minmax", "all", True),
    "Standard-to-Robust": ("mean", "one-hot", "robust", "all", True),
    "No-derived": ("mean", "one-hot", "standard", "all", False),
    "Exp-2-all": ("median", "ordinal", "minmax", "all", True),
    "Exp-3-all": ("most_frequent", "one-hot", "robust", "all", True),
}


def pipeline(spec, model):
    if spec is None:
        return Pipeline([
            ("prep", ColumnTransformer([("raw", "passthrough", BASE)])),
            ("model", model),
        ])
    imputation, encoding, scaling, top_k, derived = spec
    encoder = (OneHotEncoder(handle_unknown="ignore") if encoding == "one-hot"
               else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    scaler = {"standard": StandardScaler, "minmax": MinMaxScaler,
              "robust": RobustScaler}[scaling]()
    numeric = Pipeline([("fill", SimpleImputer(strategy=imputation)), ("scale", scaler)])
    categorical = Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                            ("encode", encoder)])
    prep = ColumnTransformer([
        ("num", numeric, NUM + (DERIVED if derived else [])),
        ("cat", categorical, CAT),
    ])
    selection = SelectKBest(f_classif, k=top_k) if top_k != "all" else "passthrough"
    return Pipeline([("prep", prep), ("select", selection), ("model", model)])


def eda(df, out):
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), layout="constrained")
    sns.histplot(data=df, x="Age", bins=24, color="#166c72", ax=axes[0, 0])
    axes[0, 0].set_title("Age distribution")
    sns.boxplot(data=df, x="Fare", color="#56a3a1", ax=axes[0, 1])
    axes[0, 1].set_title("Fare and outliers")
    sns.countplot(data=df, x="Survived", hue="Survived", palette=["#166c72", "#e29050"],
                  legend=False, ax=axes[1, 0])
    for container in axes[1, 0].containers:
        axes[1, 0].bar_label(container, padding=2)
    axes[1, 0].set_title("Survival counts")
    sns.heatmap(df[["Survived", "Pclass", *NUM]].corr(), annot=True, fmt=".2f",
                cmap="vlag", vmin=-1, vmax=1, ax=axes[1, 1], annot_kws={"size": 9})
    axes[1, 1].set_title("Pearson correlations")
    fig.savefig(out / "eda.png", dpi=180)
    plt.close(fig)
    missing = df.isna().sum().rename("n").to_frame()
    missing["pct"] = missing["n"] * 100 / len(df)
    missing.to_csv(out / "missing.csv")
    quartiles = df[["Age", "Fare"]].quantile([.25, .75])
    iqr = quartiles.loc[.75] - quartiles.loc[.25]
    low, high = quartiles.loc[.25] - 1.5 * iqr, quartiles.loc[.75] + 1.5 * iqr
    pd.DataFrame({"low": low, "high": high,
                  "n": (df[low.index].lt(low) | df[high.index].gt(high)).sum()}
                 ).to_csv(out / "outliers.csv")
    df.groupby("Sex")["Survived"].agg(["size", "mean"]).to_csv(out / "by_sex.csv")


def scores(estimator, x, y):
    predicted = estimator.predict(x)
    proba = estimator.predict_proba(x)[:, 1]
    return {
        "accuracy": accuracy_score(y, predicted),
        "precision": precision_score(y, predicted, zero_division=0),
        "recall": recall_score(y, predicted, zero_division=0),
        "f1": f1_score(y, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
    }


def run(data, out):
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data)
    print("Shape:", df.shape)
    print(df.dtypes.to_string())
    assert len(df) >= 500 and df["Survived"].notna().all()
    assert df[BASE].notna().all().all()
    eda(df, out)
    x = df.drop(columns="Survived").copy()
    x["FamilySize"] = x["SibSp"] + x["Parch"] + 1
    x["FarePerPerson"] = x["Fare"] / x["FamilySize"]
    train, test, y_train, y_test = train_test_split(
        x, df["Survived"], test_size=.2, random_state=SEED, stratify=df["Survived"])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "LR": LogisticRegression(max_iter=2000, random_state=SEED),
        "RF": RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1),
    }
    results, selections = [], []
    for (label, spec), (name, model) in product(SPECS.items(), models.items()):
        estimator = pipeline(spec, model)
        validation = cross_validate(estimator, train, y_train, cv=cv,
                                    scoring="roc_auc", return_train_score=True,
                                    error_score="raise")
        estimator.fit(train, y_train)
        results.append({"experiment": label, "model": name,
                        "cv_auc": validation["test_score"].mean(),
                        "cv_sd": validation["test_score"].std(ddof=1),
                        "train_cv_auc": validation["train_score"].mean(),
                        **scores(estimator, test, y_test)})
        if label in {"Exp-2", "Exp-3"} and name == "LR":
            features = estimator.named_steps["prep"].get_feature_names_out()
            selector = estimator.named_steps["select"]
            selections.extend({"experiment": label, "feature": feature, "f": score,
                               "selected": kept} for feature, score, kept in zip(
                features, selector.scores_, selector.get_support()))
    result = pd.DataFrame(results)
    result.to_csv(out / "comparison.csv", index=False)
    print(result.round(4).to_string(index=False))
    pd.DataFrame(selections).sort_values(["experiment", "f"], ascending=[True, False]
                                        ).to_csv(out / "selection.csv", index=False)
    primary = result[result["experiment"].isin(["Exp-1", "Exp-2", "Exp-3"])]
    best = primary.sort_values("cv_auc", ascending=False).iloc[0]
    grids = {
        "LR": {"model__C": [.01, .1, 1, 10, 100]},
        "RF": {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 6]},
    }
    search = GridSearchCV(pipeline(SPECS[best.experiment], models[best.model]),
                          grids[best.model], cv=cv, scoring="roc_auc", error_score="raise")
    search.fit(train, y_train)
    summary = {"experiment": best.experiment, "model": best.model,
               "params": str(search.best_params_), "cv_auc": search.best_score_,
               **scores(search, test, y_test)}
    pd.DataFrame([summary]).to_csv(out / "grid.csv", index=False)
    print("Grid search:", summary)
    used_columns = NUM + CAT
    # Permute each original variable before rebuilding any dependent feature.
    def score_raw(_estimator, raw, labels):
        rebuilt = raw.assign(FamilySize=raw["SibSp"] + raw["Parch"] + 1)
        rebuilt["FarePerPerson"] = rebuilt["Fare"] / rebuilt["FamilySize"]
        return roc_auc_score(labels, _estimator.predict_proba(rebuilt)[:, 1])

    imp = permutation_importance(search.best_estimator_, test[used_columns].copy(),
                                 y_test, scoring=score_raw, n_repeats=25, random_state=SEED)
    order = np.argsort(imp.importances_mean)
    fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
    ax.barh(np.array(used_columns)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#166c72", alpha=.85)
    ax.axvline(0, linewidth=.8, color="#333333")
    ax.set_xlabel("ROC-AUC decrease (mean ± SD, 25 permutations)")
    ax.set_title("Permutation importance on the held-out set")
    fig.savefig(out / "importance.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("titanic.csv"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    arguments = parser.parse_args()
    run(arguments.data, arguments.out)
