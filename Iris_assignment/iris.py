# pandas display setting
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression  # LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore")

iris_dataset = load_iris()


class AnalyzeIris:
    def __init__(self):
        self.df = pd.DataFrame(iris_dataset.data, columns=iris_dataset.feature_names)

    def get(self):
        df_add_label = self.df.copy()
        df_add_label["Label"] = iris_dataset.target
        return df_add_label

    def get_correlation(self):
        data_corr = self.df.corr()
        return data_corr

    def pair_plot(self, diag="hist"):
        df = self.get()
        label_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        df["LabelName"] = df["Label"].map(label_map)
        sns.pairplot(df.drop(columns=["Label"]), hue="LabelName", diag_kind=diag)
        plt.show()

    def calc_supervised_scores(self, n_neighbors=4):
        models = [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("LinearSVC", LinearSVC(max_iter=10000)),
            ("SVC", SVC()),
            ("DecisionTreeClassifier", DecisionTreeClassifier()),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ("RandomForestClassifier", RandomForestClassifier()),
            ("GradientBoostingClassifier", GradientBoostingClassifier()),
            ("MLPClassifier", MLPClassifier(max_iter=2000)),
        ]

        X = self.df
        y = iris_dataset.target

        results_dict = {}

        for name, model in models:
            results = cross_validate(model, X, y, cv=5, return_train_score=True)
            results_dict[name] = results

        return results_dict

    def all_supervised(self, n_neighbors=4):
        results_dict = self.calc_supervised_scores(n_neighbors)

        for name, results in results_dict.items():
            print("== {} ==".format(name))
            for test_score, train_score in zip(
                results["test_score"],
                results["train_score"],
            ):
                print(
                    "test score: {:.3f}, train score: {:.3f}".format(
                        test_score, train_score
                    )
                )
            print()

    def get_supervised(self):
        results_dict = self.calc_supervised_scores()

        score_dict = {}

        for name, results in results_dict.items():
            score_dict[name] = results["test_score"]

        return pd.DataFrame(score_dict)

    def best_supervised(self):

        df_score = self.get_supervised().describe()
        best_method = df_score.loc["mean"].idxmax()
        best_score = df_score.loc["mean"].max()

        return (best_method, best_score)

    def plot_feature_importances_all(self):
        X = self.df
        y = iris_dataset.target

        models = [
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=0)),
            ("RandomForestClassifier", RandomForestClassifier(random_state=0)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=0)),
        ]

        for name, model in models:
            model.fit(X, y)

            plt.figure()
            n_features = X.shape[1]
            plt.barh(range(n_features), model.feature_importances_, align="center")
            plt.yticks(np.arange(n_features), X.columns)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.title(name)
            plt.show()

    def visualize_decision_tree(self):
        X = self.df
        y = iris_dataset.target

        model = DecisionTreeClassifier(random_state=0)
        model.fit(X, y)

        plt.figure(figsize=(16, 10))
        graph = plot_tree(
            model,
            feature_names=X.columns,
            class_names=iris_dataset.target_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.show()

        return graph
