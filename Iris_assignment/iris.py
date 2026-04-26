# -*- coding: utf-8 -*-
"""Irisデータセットの探索的分析と分類モデル評価モジュール。

このモジュールはsklearnのIrisデータセットを用いて、特徴量の可視化・
相関分析・複数の教師あり学習モデルの交差検証・特徴量重要度の可視化を
行うクラスを提供する。

Example:
    基本的な使い方::

        iris = AnalyzeIris()
        iris.get()
        iris.all_supervised()
        best_method, best_score = iris.best_supervised()
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

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
from sklearn.utils import Bunch

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore")

RANDOM_STATE: int = 0
"""再現性を保つために全モデルで共通利用する乱数シード。"""

DEFAULT_N_NEIGHBORS: int = 4
"""KNeighborsClassifierで使用するデフォルトの近傍数。"""

CV_SPLITS: int = 5
"""交差検証の分割数。"""

IRIS_DATASET: Bunch = load_iris()

DEFAULT_CLASSIFIERS: dict[str, Any] = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "LinearSVC": LinearSVC(max_iter=10000, random_state=RANDOM_STATE),
    "SVC": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "KNeighborsClassifier": KNeighborsClassifier(
        n_neighbors=DEFAULT_N_NEIGHBORS,
    ),
    "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),
    "MLPClassifier": MLPClassifier(max_iter=2000, random_state=RANDOM_STATE),
}

DEFAULT_TREE_CLASSIFIERS: dict[str, Any] = {
    "DecisionTreeClassifier": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForestClassifier": RandomForestClassifier(random_state=RANDOM_STATE),
    "GradientBoostingClassifier": GradientBoostingClassifier(
        random_state=RANDOM_STATE,
    ),
}


class AnalyzeIris:
    """Irisデータセットの分析・可視化・モデル評価を行うクラス。

    Attributes:
        dataset: 読み込んだIrisデータセット。
        df_feature (pd.DataFrame): 特徴量のみのDataFrame（ラベルなし）。
    """

    def __init__(self) -> None:
        """Irisデータセットを読み込み、特徴量DataFrameを初期化する。"""
        self.dataset: Bunch = IRIS_DATASET
        self.df_feature: pd.DataFrame = pd.DataFrame(
            self.dataset.data,
            columns=self.dataset.feature_names,
        )

    def _validate_diag_kind(
        self,
        diag_kind: Literal["auto", "hist", "kde"] | None,
    ) -> None:
        """pairplotの対角要素の描画方法を確認する。

        Args:
            diag_kind (Literal["auto", "hist", "kde"] | None): 対角要素の描画方法。

        Raises:
            ValueError: ``diag_kind`` が使用可能な候補以外の場合。
        """
        if diag_kind not in {"auto", "hist", "kde", None}:
            raise ValueError(
                "diag は 'auto', 'hist', 'kde', None のいずれかです。"
                f" 受け取った値: {diag_kind!r}"
            )

    def _validate_n_neighbors(self, n_neighbors: int) -> None:
        """KNeighborsClassifierの近傍数を確認する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。

        Raises:
            ValueError: ``n_neighbors`` が使用可能な範囲外の場合。
        """
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int):
            raise ValueError("n_neighbors はint型です。")
        if n_neighbors < 1:
            raise ValueError("n_neighbors は1以上のint型です。")

        int_train_sample_count = len(self.df_feature) * (CV_SPLITS - 1) // CV_SPLITS
        if n_neighbors > int_train_sample_count:
            raise ValueError(
                "n_neighbors が大きすぎます。"
                f" {CV_SPLITS}分割交差検証では {int_train_sample_count} 以下にしてください。"
            )

    def get(self, head: int | None = None) -> pd.DataFrame:
        """ラベル列を付加したDataFrameを返す。

        Args:
            head (int | None): 先頭から返す行数。``None`` の場合は全行を返す。

        Returns:
            pd.DataFrame: ラベル列（Label）を含むDataFrame。
        """
        df_labeled: pd.DataFrame = self.df_feature.copy()
        df_labeled["Label"] = self.dataset.target

        if head is not None:
            return df_labeled.head(head)
        return df_labeled

    def get_correlation(self) -> pd.DataFrame:
        """特徴量間の相関係数行列を返す。

        Returns:
            pd.DataFrame: 特徴量間の相関係数を格納したDataFrame。
        """
        return self.df_feature.corr()

    def pair_plot(
        self,
        diag_kind: Literal["auto", "hist", "kde"] | None = "hist",
    ) -> None:
        """全特徴量の組み合わせを散布図で表示する。

        Args:
            diag_kind (Literal["auto", "hist", "kde"] | None): 対角成分の
                グラフ種別。``"auto"``, ``"hist"``, ``"kde"``, ``None`` を
                指定する。デフォルトは ``"hist"``。

        Raises:
            ValueError: ``diag_kind`` が使用可能な候補以外の場合。
        """
        self._validate_diag_kind(diag_kind)

        df_labeled: pd.DataFrame = self.get()
        dict_label_name: dict[int, str] = {
            int(index): str(label_name)
            for index, label_name in enumerate(self.dataset.target_names)
        }
        df_labeled["LabelName"] = df_labeled["Label"].map(dict_label_name)

        sns.pairplot(
            df_labeled.drop(columns=["Label"]),
            hue="LabelName",
            diag_kind=diag_kind,
        )
        plt.show()

    def calc_supervised_scores(
        self,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> dict[str, dict[str, np.ndarray]]:
        """全分類モデルに対して交差検証スコアを計算する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            dict[str, dict[str, np.ndarray]]: モデル名をキー、cross_validateの
                結果を値とするdict。

        Raises:
            ValueError: ``n_neighbors`` がint型でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        self._validate_n_neighbors(n_neighbors)

        dict_classifier: dict[str, Any] = DEFAULT_CLASSIFIERS.copy()
        dict_classifier["KNeighborsClassifier"] = KNeighborsClassifier(
            n_neighbors=n_neighbors,
        )

        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target
        dict_results: dict[str, dict[str, np.ndarray]] = {}

        for str_classifier_name, classifier in dict_classifier.items():
            dict_results[str_classifier_name] = cross_validate(
                classifier,
                df_feature,
                ndarray_target,
                cv=CV_SPLITS,
                return_train_score=True,
            )

        return dict_results

    def all_supervised(self, n_neighbors: int = DEFAULT_N_NEIGHBORS) -> None:
        """全分類モデルの交差検証スコアをコンソールに出力する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Raises:
            ValueError: ``n_neighbors`` がint型でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        dict_results = self.calc_supervised_scores(n_neighbors)

        for str_classifier_name, dict_score in dict_results.items():
            print("== {} ==".format(str_classifier_name))
            for test_score, train_score in zip(
                dict_score["test_score"],
                dict_score["train_score"],
            ):
                print(
                    "test score: {:.3f}, train score: {:.3f}".format(
                        test_score,
                        train_score,
                    )
                )
            print()

    def get_supervised(
        self,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> pd.DataFrame:
        """全分類モデルのテストスコアをDataFrameで返す。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            pd.DataFrame: モデル名を列名、各foldのテストスコアを行とする
                DataFrame。

        Raises:
            ValueError: ``n_neighbors`` がint型でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        dict_results = self.calc_supervised_scores(n_neighbors)
        dict_test_score: dict[str, np.ndarray] = {}

        for str_classifier_name, dict_score in dict_results.items():
            dict_test_score[str_classifier_name] = dict_score["test_score"]

        return pd.DataFrame(dict_test_score)

    def best_supervised(
        self,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> tuple[str, float]:
        """平均テストスコアが最も高い分類モデルを返す。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            tuple[str, float]: 以下の要素を持つタプル。

                - str_best_method (str): 最良モデルの名前。
                - float_best_score (float): 最良モデルの平均テストスコア。

        Raises:
            ValueError: ``n_neighbors`` がint型でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        df_score = self.get_supervised(n_neighbors).describe()
        str_best_method: str = str(df_score.loc["mean"].idxmax())
        float_best_score: float = float(df_score.loc["mean"].max())
        return (str_best_method, float_best_score)

    def plot_feature_importances_all(self) -> None:
        """木ベースの分類モデルの特徴量重要度を棒グラフで表示する。"""
        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target

        for str_classifier_name, classifier in DEFAULT_TREE_CLASSIFIERS.items():
            classifier.fit(df_feature, ndarray_target)

            int_feature_count: int = df_feature.shape[1]
            plt.figure()
            plt.barh(
                range(int_feature_count),
                classifier.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(int_feature_count), df_feature.columns)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.title(str_classifier_name)
            plt.show()

    def visualize_decision_tree(self) -> list[Any]:
        """決定木の構造を可視化して表示する。

        DecisionTreeClassifierを全データで学習し、木構造を図示する。

        Returns:
            list[Any]: plot_treeが返すArtistオブジェクトのリスト。
        """
        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target

        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
        classifier.fit(df_feature, ndarray_target)

        plt.figure(figsize=(16, 10))
        list_graph = plot_tree(
            classifier,
            feature_names=df_feature.columns,
            class_names=self.dataset.target_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.show()

        return list_graph
