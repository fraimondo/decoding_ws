# %%
from pathlib import Path
import seaborn as sns
from seaborn import load_dataset
from julearn import PipelineCreator, run_cross_validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

# %%
import shap

# shap.initjs()
# %%
df_iris = load_dataset("iris")

df_iris = df_iris[df_iris["species"].isin(["versicolor", "virginica"])]

# %%
df_train, df_test = train_test_split(df_iris, test_size=0.2, random_state=42)

model_name = "svm"

importances = {}
coefs = {}

y = "species"
X_types = {"continuous": [".*"]}
creator = PipelineCreator(problem_type="classification")
creator.add("svm", kernel="linear", probability=True)

# %% Run original model
print("Testing original model")
X = ["sepal_length", "petal_length"]
scores, model = run_cross_validation(
    X=X,
    y=y,
    data=df_train,
    model=creator,
    return_estimator="final",
    X_types=X_types,
)

print(f"\tScores: {scores['test_score'].mean()}")

r = permutation_importance(
    model, df_test[X], df_test[y], n_repeats=30, random_state=0
)
print(f"\tImportances: {r.importances_mean}")
importances["original"] = r.importances_mean

coefs["original"] = model.named_steps["svm"].coef_
print(f"\tcoefs/importances: {coefs['original']}")

# %% Plot decision function
ax = sns.scatterplot(
    x="sepal_length", y="petal_length", hue=y, data=df_train, s=20
)

ax.set_title("Iris data (lenghts only)")

plt.show(ax)
ax = sns.scatterplot(
    x="sepal_length", y="petal_length", hue=y, data=df_train, s=20
)

xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)
a = ax.contour(
    XX, YY, Z, colors="k", levels=[0], alpha=0.5, linestyles=["-"]
)
ax.set_title("Data with SVM decision function boundaries")
# %%
fig, ax = plt.subplots(1, 1, figsize=(3, 5))
ax = sns.barplot(x=X, y=coefs["original"][0], ax=ax)
ax.set_title("SVM Coefficients")


# %%
from julearn.inspect import preprocess
df_train_preprocessed = preprocess(model, data=df_train, X=X)
df_test_preprocessed = preprocess(model, data=df_test, X=X)
# %%
explainer = shap.Explainer(model.steps[-1][1], df_train_preprocessed[X])
# %%
shap_values = explainer(df_test_preprocessed)
# %%
shap.plots.waterfall(shap_values[1])
shap.plots.waterfall(shap_values[4])
# %%
shap.plots.bar(shap_values[1])
shap.plots.bar(shap_values[4])

# %%
shap.plots.force(shap_values[1])
# %%
shap.plots.force(shap_values[4])
# %%
shap.plots.bar(shap_values)
# %%
shap.plots.beeswarm(shap_values)
# %%
