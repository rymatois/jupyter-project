# -*- coding: utf-8 -*-
"""Irisデータセットの探索的分析と分類モデル評価モジュール。

このモジュールはsklearnのIrisデータセットを用いて、特徴量の可視化・
相関分析・複数の教師あり学習モデルの交差検証・特徴量重要度の可視化を
行うクラスを提供する。

Example:
    基本的な使い方::

        analyzer = AnalyzeIris()
        analyzer.pair_plot()
        analyzer.all_supervised()
        best_model, best_score = analyzer.best_supervised()

Attributes:
    iris_dataset (sklearn.utils.Bunch): sklearnから読み込んだIrisデータセット。
    LABEL_NAME_MAP (dict): ラベル番号と種名のマッピング。
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore")

iris_dataset = load_iris()

LABEL_NAME_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}
"""dict: ラベル番号と種名のマッピング。"""


class AnalyzeIris:
    """Irisデータセットの分析・可視化・モデル評価を行うクラス。

    Attributes:
        feature_df (pd.DataFrame): 特徴量のみのDataFrame（ラベルなし）。
    """

    def __init__(self):
        """Irisデータセットを読み込み、特徴量DataFrameを初期化する。"""
        self.feature_df = pd.DataFrame(
            iris_dataset.data, columns=iris_dataset.feature_names
        )

    def get(self):
        """ラベル列を付加したDataFrameを返す。

        Returns:
            pd.DataFrame: ラベル列（Label）を含むDataFrame。
        """
        labeled_df = self.feature_df.copy()
        labeled_df["Label"] = iris_dataset.target
        return labeled_df

    def get_correlation(self):
        """特徴量間の相関係数行列を返す。

        Returns:
            pd.DataFrame: 特徴量間の相関係数を格納したDataFrame。
        """
        correlation_matrix = self.feature_df.corr()
        return correlation_matrix

    def pair_plot(self, diag_kind="hist"):
        """ペアプロットを表示する。

        Args:
            diag_kind (str): 対角成分のグラフ種別。``"hist"`` または
                ``"kde"`` を指定する。デフォルトは ``"hist"``。
        """
        labeled_df = self.get()
        labeled_df["LabelName"] = labeled_df["Label"].map(LABEL_NAME_MAP)
        sns.pairplot(
            labeled_df.drop(columns=["Label"]),
            hue="LabelName",
            diag_kind=diag_kind,
        )
        plt.show()

    def _build_classifier_list(self, n_neighbors):
        """評価対象の分類モデル一覧を生成する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。

        Returns:
            list: (モデル名 (str), モデルインスタンス) のタプルのリスト。
        """
        return [
            ("LogisticRegression", LogisticRegression(max_iter=1000)),
            ("LinearSVC", LinearSVC(max_iter=10000)),
            ("SVC", SVC()),
            ("DecisionTreeClassifier", DecisionTreeClassifier()),
            ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=n_neighbors)),
            ("RandomForestClassifier", RandomForestClassifier()),
            ("GradientBoostingClassifier", GradientBoostingClassifier()),
            ("MLPClassifier", MLPClassifier(max_iter=2000)),
        ]

    def calc_cv_scores(self, n_neighbors=4):
        """全分類モデルに対して交差検証スコアを計算する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。
                デフォルトは4。

        Returns:
            dict: モデル名 (str) をキー、cross_validateの結果 (dict) を
                値とするdict。
        """
        classifiers = self._build_classifier_list(n_neighbors)
        X = self.feature_df
        y = iris_dataset.target

        cv_results = {}
        for model_name, classifier in classifiers:
            result = cross_validate(classifier, X, y, cv=5, return_train_score=True)
            cv_results[model_name] = result

        return cv_results

    def all_supervised(self, n_neighbors=4):
        """全分類モデルの交差検証スコアをコンソールに出力する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。
                デフォルトは4。
        """
        cv_results = self.calc_cv_scores(n_neighbors)

        for model_name, result in cv_results.items():
            print("== {} ==".format(model_name))
            for test_score, train_score in zip(
                result["test_score"],
                result["train_score"],
            ):
                print(
                    "test score: {:.3f}, train score: {:.3f}".format(
                        test_score, train_score
                    )
                )
            print()

    def get_supervised(self):
        """全分類モデルのテストスコアをDataFrameで返す。

        Returns:
            pd.DataFrame: モデル名を列名、各foldのテストスコアを行とする
                DataFrame。
        """
        cv_results = self.calc_cv_scores()

        test_score_dict = {
            model_name: result["test_score"]
            for model_name, result in cv_results.items()
        }
        return pd.DataFrame(test_score_dict)

    def best_supervised(self):
        """平均テストスコアが最も高い分類モデルを返す。

        Returns:
            tuple: 以下の要素を持つタプル。

                - best_model_name (str): 最良モデルの名前。
                - best_mean_score (float): 最良モデルの平均テストスコア。
        """
        score_stats_df = self.get_supervised().describe()
        best_model_name = score_stats_df.loc["mean"].idxmax()
        best_mean_score = score_stats_df.loc["mean"].max()

        return (best_model_name, best_mean_score)

    def plot_feature_importances_all(self):
        """木ベースの分類モデルの特徴量重要度を棒グラフで表示する。

        DecisionTreeClassifier、RandomForestClassifier、
        GradientBoostingClassifierの特徴量重要度をそれぞれ描画する。
        """
        X = self.feature_df
        y = iris_dataset.target

        tree_classifiers = [
            ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=0)),
            ("RandomForestClassifier", RandomForestClassifier(random_state=0)),
            ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=0)),
        ]

        for model_name, classifier in tree_classifiers:
            classifier.fit(X, y)

            n_features = X.shape[1]
            plt.figure()
            plt.barh(
                range(n_features),
                classifier.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(n_features), X.columns)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.title(model_name)
            plt.show()

    def visualize_decision_tree(self):
        """決定木の構造を可視化して表示する。

        DecisionTreeClassifierを全データで学習し、木構造を図示する。

        Returns:
            list: plot_treeが返すArtistオブジェクトのリスト。
        """
        X = self.feature_df
        y = iris_dataset.target

        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(X, y)

        plt.figure(figsize=(16, 10))
        tree_artists = plot_tree(
            classifier,
            feature_names=X.columns,
            class_names=iris_dataset.target_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.show()

        return tree_artists
