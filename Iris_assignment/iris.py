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
from itertools import combinations
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler
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

DEFAULT_SCALERS: dict[str, Any | None] = {
    "Original": None,
    "MinMaxScaler": MinMaxScaler(),
    "StandardScaler": StandardScaler(),
    "RobustScaler": RobustScaler(),
    "Normalizer": Normalizer(),
}
"""plot_scaled_dataで比較するスケーラー一覧。"""

SCALED_PLOT_TICK_SETTINGS = {
    ("Original", "sepal length (cm)", "sepal width (cm)"): {
        "xlim": (4.8, 8.1),
        "xticks": [6.0, 8.0],
        "ylim": (1.9, 4.5),
        "yticks": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
    },
    ("MinMaxScaler", "sepal length (cm)", "sepal width (cm)"): {
        "xlim": (-0.05, 1.05),
        "xticks": [0.0, 0.5, 1.0],
        "ylim": (-0.1, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    ("StandardScaler", "sepal length (cm)", "sepal width (cm)"): {
        "xlim": (-2.2, 2.2),
        "xticks": [-2.0, 0.0, 2.0],
        "ylim": (-2.5, 3.3),
        "yticks": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    },
    ("RobustScaler", "sepal length (cm)", "sepal width (cm)"): {
        "xlim": (-1.2, 1.6),
        "xticks": [-1.0, 0.0, 1.0],
        "ylim": (-2.2, 3.0),
        "yticks": [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    },
    ("Normalizer", "sepal length (cm)", "sepal width (cm)"): {
        "xlim": (0.66, 0.86),
        "xticks": [0.7, 0.8],
        "ylim": (0.22, 0.62),
        "yticks": [0.3, 0.4, 0.5, 0.6],
    },
    ("Original", "sepal length (cm)", "petal length (cm)"): {
        "xlim": (4.8, 8.1),
        "xticks": [6.0, 8.0],
        "ylim": (0.8, 7.2),
        "yticks": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    },
    ("MinMaxScaler", "sepal length (cm)", "petal length (cm)"): {
        "xlim": (-0.05, 1.05),
        "xticks": [0.0, 0.5, 1.0],
        "ylim": (-0.05, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    ("StandardScaler", "sepal length (cm)", "petal length (cm)"): {
        "xlim": (-2.2, 2.2),
        "xticks": [-2.0, 0.0, 2.0],
        "ylim": (-1.7, 1.7),
        "yticks": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    },
    ("RobustScaler", "sepal length (cm)", "petal length (cm)"): {
        "xlim": (-1.2, 1.6),
        "xticks": [-1.0, 0.0, 1.0],
        "ylim": (-1.05, 0.8),
        "yticks": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
    },
    ("Normalizer", "sepal length (cm)", "petal length (cm)"): {
        "xlim": (0.66, 0.86),
        "xticks": [0.7, 0.8],
        "ylim": (0.15, 0.64),
        "yticks": [0.2, 0.3, 0.4, 0.5, 0.6],
    },
    ("Original", "sepal length (cm)", "petal width (cm)"): {
        "xlim": (4.8, 8.1),
        "xticks": [6.0, 8.0],
        "ylim": (0.0, 2.55),
        "yticks": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    },
    ("MinMaxScaler", "sepal length (cm)", "petal width (cm)"): {
        "xlim": (-0.05, 1.05),
        "xticks": [0.0, 0.5, 1.0],
        "ylim": (-0.02, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    ("StandardScaler", "sepal length (cm)", "petal width (cm)"): {
        "xlim": (-2.2, 2.2),
        "xticks": [-2.0, 0.0, 2.0],
        "ylim": (-1.6, 1.8),
        "yticks": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    },
    ("RobustScaler", "sepal length (cm)", "petal width (cm)"): {
        "xlim": (-1.2, 1.6),
        "xticks": [-1.0, 0.0, 1.0],
        "ylim": (-0.85, 0.85),
        "yticks": [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
    },
    ("Normalizer", "sepal length (cm)", "petal width (cm)"): {
        "xlim": (0.66, 0.86),
        "xticks": [0.7, 0.8],
        "ylim": (0.0, 0.29),
        "yticks": [0.05, 0.1, 0.15, 0.2, 0.25],
    },
    ("Original", "sepal width (cm)", "petal length (cm)"): {
        "xlim": (1.9, 4.4),
        "xticks": [2.0, 3.0, 4.0],
        "ylim": (0.8, 7.1),
        "yticks": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    },
    ("MinMaxScaler", "sepal width (cm)", "petal length (cm)"): {
        "xlim": (-0.05, 1.05),
        "xticks": [0.0, 0.5, 1.0],
        "ylim": (-0.05, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    ("StandardScaler", "sepal width (cm)", "petal length (cm)"): {
        "xlim": (-2.7, 2.7),
        "xticks": [-2.5, 0.0, 2.5],
        "ylim": (-1.7, 1.7),
        "yticks": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    },
    ("RobustScaler", "sepal width (cm)", "petal length (cm)"): {
        "xlim": (-2.2, 2.4),
        "xticks": [-2.0, 0.0, 2.0],
        "ylim": (-1.05, 0.8),
        "yticks": [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
    },
    ("Normalizer", "sepal width (cm)", "petal length (cm)"): {
        "xlim": (0.28, 0.62),
        "xticks": [0.4, 0.6],
        "ylim": (0.15, 0.64),
        "yticks": [0.2, 0.3, 0.4, 0.5, 0.6],
    },
    ("Original", "sepal width (cm)", "petal width (cm)"): {
        "xlim": (1.9, 4.4),
        "xticks": [2.0, 3.0, 4.0],
        "ylim": (0.0, 2.55),
        "yticks": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    },
    ("MinMaxScaler", "sepal width (cm)", "petal width (cm)"): {
        "xlim": (-0.05, 1.05),
        "xticks": [0.0, 0.5, 1.0],
        "ylim": (-0.02, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    ("StandardScaler", "sepal width (cm)", "petal width (cm)"): {
        "xlim": (-2.7, 2.7),
        "xticks": [-2.5, 0.0, 2.5],
        "ylim": (-1.6, 1.8),
        "yticks": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    },
    ("RobustScaler", "sepal width (cm)", "petal width (cm)"): {
        "xlim": (-2.2, 2.4),
        "xticks": [-2.0, 0.0, 2.0],
        "ylim": (-0.85, 0.85),
        "yticks": [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
    },
    ("Normalizer", "sepal width (cm)", "petal width (cm)"): {
        "xlim": (0.28, 0.62),
        "xticks": [0.4, 0.6],
        "ylim": (0.0, 0.29),
        "yticks": [0.05, 0.1, 0.15, 0.2, 0.25],
    },
    ("Original", "petal length (cm)", "petal width (cm)"): {
        "xlim": (1.0, 7.0),
        "xticks": [2.5, 5.0],
        "ylim": (0.0, 2.55),
        "yticks": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
    },
    ("MinMaxScaler", "petal length (cm)", "petal width (cm)"): {
        "xlim": (-0.05, 1.05),
        "xticks": [0.0, 0.5, 1.0],
        "ylim": (-0.02, 1.05),
        "yticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    },
    ("StandardScaler", "petal length (cm)", "petal width (cm)"): {
        "xlim": (-1.4, 1.7),
        "xticks": [-1.0, 0.0, 1.0],
        "ylim": (-1.6, 1.8),
        "yticks": [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
    },
    ("RobustScaler", "petal length (cm)", "petal width (cm)"): {
        "xlim": (-1.2, 0.8),
        "xticks": [-1.0, 0.0],
        "ylim": (-0.85, 0.85),
        "yticks": [-0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75],
    },
    ("Normalizer", "petal length (cm)", "petal width (cm)"): {
        "xlim": (0.16, 0.64),
        "xticks": [0.2, 0.4, 0.6],
        "ylim": (0.0, 0.29),
        "yticks": [0.05, 0.1, 0.15, 0.2, 0.25],
    },
}
"""模範解答に合わせたplot_scaled_data用の固定目盛り。"""


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

    def _validate_positive_int(
        self,
        value: int,
        variable_name: str,
        upper_bound: int | None = None,
    ) -> None:
        """値が1以上の整数かどうかを確認する。

        Args:
            value (int): 検証対象の値。
            variable_name (str): エラーメッセージに表示する変数名。
            upper_bound (int | None): 上限値。``None`` の場合は上限チェックを
                行わない。

        Raises:
            TypeError: ``value`` が整数以外の場合。
            ValueError: ``value`` が1未満、または上限値を超える場合。
        """
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{variable_name} はint型です。"
                f" 受け取った値: {value!r} (type={type(value).__name__})"
            )
        if value < 1:
            raise ValueError(f"{variable_name} は1以上のint型です。")
        if upper_bound is not None and value > upper_bound:
            raise ValueError(
                f"{variable_name} は {upper_bound} 以下にしてください。"
                f" 受け取った値: {value}"
            )

    def _validate_positive_number(self, value: float, variable_name: str) -> None:
        """値が正の数値かどうかを確認する。

        Args:
            value (float): 検証対象の値。
            variable_name (str): エラーメッセージに表示する変数名。

        Raises:
            TypeError: ``value`` が数値以外の場合。
            ValueError: ``value`` が0以下の場合。
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"{variable_name} は数値型(int または float)です。"
                f" 受け取った値: {value!r} (type={type(value).__name__})"
            )
        if value <= 0:
            raise ValueError(f"{variable_name} は正の数値です。受け取った値: {value}")

    def _validate_bool(self, value: bool, variable_name: str) -> None:
        """値がbool型かどうかを確認する。

        Args:
            value (bool): 検証対象の値。
            variable_name (str): エラーメッセージに表示する変数名。

        Raises:
            TypeError: ``value`` がbool以外の場合。
        """
        if not isinstance(value, bool):
            raise TypeError(
                f"{variable_name} はbool型です。"
                f" 受け取った値: {value!r} (type={type(value).__name__})"
            )

    def _get_feature_array(self, scaling: bool = True) -> np.ndarray:
        """特徴量をクラスタリング用のndarrayとして返す。

        Args:
            scaling (bool): ``True`` の場合はStandardScalerで標準化する。

        Returns:
            np.ndarray: 特徴量配列。
        """
        self._validate_bool(scaling, "scaling")

        if scaling:
            scaler = StandardScaler()
            return scaler.fit_transform(self.df_feature)
        return self.df_feature.to_numpy()

    def _get_pca_projection(self, ndarray_feature: np.ndarray) -> np.ndarray:
        """特徴量をPCAで2次元に投影する。

        Args:
            ndarray_feature (np.ndarray): 特徴量配列。

        Returns:
            np.ndarray: PCAの第1・第2主成分得点。
        """
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        return pca.fit_transform(ndarray_feature)

    def _calc_cluster_summary(
        self,
        str_method_name: str,
        ndarray_feature: np.ndarray,
        ndarray_cluster: np.ndarray,
    ) -> dict[str, Any]:
        """クラスタリング結果の要約指標を計算する。

        Args:
            str_method_name (str): 手法名。
            ndarray_feature (np.ndarray): クラスタリングに使った特徴量配列。
            ndarray_cluster (np.ndarray): 推定クラスタラベル。

        Returns:
            dict[str, Any]: クラスタ数、ノイズ数、ARI、シルエット係数。
        """
        set_labels = set(ndarray_cluster)
        int_noise_count = int(np.sum(ndarray_cluster == -1))
        int_cluster_count = len(set_labels) - (1 if -1 in set_labels else 0)

        float_silhouette: float | None = None
        if len(set_labels) >= 2 and len(set_labels) < len(ndarray_cluster):
            float_silhouette = float(silhouette_score(ndarray_feature, ndarray_cluster))

        return {
            "method": str_method_name,
            "n_clusters": int_cluster_count,
            "n_noise": int_noise_count,
            "adjusted_rand_score": float(
                adjusted_rand_score(self.dataset.target, ndarray_cluster)
            ),
            "silhouette_score": float_silhouette,
        }

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

    def plot_scaled_data(self) -> pd.DataFrame:
        """各スケーリング手法でのLinearSVCのスコアと散布図を表示する。

        5分割交差検証の各foldについて、Original / MinMaxScaler /
        StandardScaler / RobustScaler / Normalizer の各スケーリングを適用し、
        LinearSVCのtest/trainスコアと散布図行列を表示する。

        Returns:
            pd.DataFrame: 各スケーリング手法のtest/trainスコアの平均と
                標準偏差をまとめたDataFrame。
        """
        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target

        kfold = StratifiedKFold(n_splits=CV_SPLITS, shuffle=False)
        list_records: list[dict[str, Any]] = []

        for int_fold, (ndarray_train_idx, ndarray_test_idx) in enumerate(
            kfold.split(df_feature, ndarray_target)
        ):
            if int_fold > 0:
                print(
                    "========================================================================="
                )

            list_records.extend(
                self._process_scaled_data_fold(
                    df_feature=df_feature,
                    ndarray_target=ndarray_target,
                    ndarray_train_idx=ndarray_train_idx,
                    ndarray_test_idx=ndarray_test_idx,
                    int_fold=int_fold,
                )
            )

        print(
            "========================================================================="
        )
        df_scaled_scores = pd.DataFrame(list_records)
        self.df_scaled_scores_detail_ = df_scaled_scores
        return self._summarize_scaled_scores(df_scaled_scores)

    def _summarize_scaled_scores(self, df_scaled_scores: pd.DataFrame) -> pd.DataFrame:
        """各スケーリング手法のtest/trainスコア要約表を作成する。"""
        df_summary = (
            df_scaled_scores.groupby("scaler")[["test_score", "train_score"]]
            .agg(["mean", "std"])
            .round(3)
        )
        df_summary.columns = [
            "test_score_mean",
            "test_score_std",
            "train_score_mean",
            "train_score_std",
        ]
        return df_summary.sort_values("test_score_mean", ascending=False)

    def _process_scaled_data_fold(
        self,
        df_feature: pd.DataFrame,
        ndarray_target: np.ndarray,
        ndarray_train_idx: np.ndarray,
        ndarray_test_idx: np.ndarray,
        int_fold: int,
    ) -> list[dict[str, Any]]:
        """plot_scaled_dataの1 fold分を評価・描画する。"""
        ndarray_x_train = df_feature.iloc[ndarray_train_idx].to_numpy()
        ndarray_x_test = df_feature.iloc[ndarray_test_idx].to_numpy()
        ndarray_y_train = ndarray_target[ndarray_train_idx]
        ndarray_y_test = ndarray_target[ndarray_test_idx]

        list_feature_pairs = list(combinations(range(df_feature.shape[1]), 2))
        fig, ndarray_axes = plt.subplots(
            len(list_feature_pairs),
            len(DEFAULT_SCALERS),
            figsize=(15, 26),
        )
        list_scaled_results = self._build_scaled_fold_results(
            ndarray_x_train=ndarray_x_train,
            ndarray_x_test=ndarray_x_test,
            ndarray_y_train=ndarray_y_train,
            ndarray_y_test=ndarray_y_test,
        )

        self._plot_scaled_fold(
            ndarray_axes=ndarray_axes,
            list_feature_pairs=list_feature_pairs,
            list_scaled_results=list_scaled_results,
            list_feature_names=df_feature.columns.tolist(),
        )

        fig.tight_layout()
        plt.show()

        return [
            {
                "fold": int_fold,
                "scaler": dict_scaled_result["scaler"],
                "test_score": dict_scaled_result["test_score"],
                "train_score": dict_scaled_result["train_score"],
            }
            for dict_scaled_result in list_scaled_results
        ]

    def _build_scaled_fold_results(
        self,
        ndarray_x_train: np.ndarray,
        ndarray_x_test: np.ndarray,
        ndarray_y_train: np.ndarray,
        ndarray_y_test: np.ndarray,
    ) -> list[dict[str, Any]]:
        """各スケーラーでの評価結果と描画用データを作成する。"""
        list_scaled_results: list[dict[str, Any]] = []

        for str_scaler_name, scaler in DEFAULT_SCALERS.items():
            if scaler is None:
                ndarray_x_train_scaled = ndarray_x_train
                ndarray_x_test_scaled = ndarray_x_test
            else:
                ndarray_x_train_scaled = scaler.fit_transform(ndarray_x_train)
                ndarray_x_test_scaled = scaler.transform(ndarray_x_test)

            classifier = LinearSVC(max_iter=10000, random_state=RANDOM_STATE)
            classifier.fit(ndarray_x_train_scaled, ndarray_y_train)
            float_test_score = classifier.score(
                ndarray_x_test_scaled,
                ndarray_y_test,
            )
            float_train_score = classifier.score(
                ndarray_x_train_scaled,
                ndarray_y_train,
            )

            print(
                "{:<14} :  test score: {:<11.3f}train score: {:<10.3f}".format(
                    str_scaler_name,
                    float_test_score,
                    float_train_score,
                )
            )

            list_scaled_results.append(
                {
                    "scaler": str_scaler_name,
                    "ndarray_x_train_scaled": ndarray_x_train_scaled,
                    "ndarray_x_test_scaled": ndarray_x_test_scaled,
                    "test_score": float_test_score,
                    "train_score": float_train_score,
                }
            )

        return list_scaled_results

    def _plot_scaled_fold(
        self,
        ndarray_axes: np.ndarray,
        list_feature_pairs: list[tuple[int, int]],
        list_scaled_results: list[dict[str, Any]],
        list_feature_names: list[str],
    ) -> None:
        """1 fold分のスケーリング結果を散布図行列として描画する。"""
        for int_idx, dict_scaled_result in enumerate(list_scaled_results):
            for int_pair_idx, (int_x_idx, int_y_idx) in enumerate(list_feature_pairs):
                ax = ndarray_axes[int_pair_idx, int_idx]
                ax.scatter(
                    dict_scaled_result["ndarray_x_train_scaled"][:, int_x_idx],
                    dict_scaled_result["ndarray_x_train_scaled"][:, int_y_idx],
                    c="blue",
                    marker="o",
                    s=98,
                    linewidths=0,
                )
                ax.scatter(
                    dict_scaled_result["ndarray_x_test_scaled"][:, int_x_idx],
                    dict_scaled_result["ndarray_x_test_scaled"][:, int_y_idx],
                    c="red",
                    marker="^",
                    s=118,
                    linewidths=0,
                )
                str_x_label = list_feature_names[int_x_idx]
                str_y_label = list_feature_names[int_y_idx]
                ax.set_title(dict_scaled_result["scaler"], fontsize=15, pad=6)
                ax.set_xlabel(str_x_label, fontsize=12)
                ax.set_ylabel(str_y_label, fontsize=12)
                ax.tick_params(labelsize=12)
                self._set_scaled_plot_ticks(
                    ax=ax,
                    str_scaler_name=dict_scaled_result["scaler"],
                    str_x_label=str_x_label,
                    str_y_label=str_y_label,
                    ndarray_x_train_scaled=dict_scaled_result["ndarray_x_train_scaled"],
                    ndarray_x_test_scaled=dict_scaled_result["ndarray_x_test_scaled"],
                    int_x_idx=int_x_idx,
                    int_y_idx=int_y_idx,
                )
                ax.set_box_aspect(1.65)

    def _set_scaled_plot_ticks(
        self,
        ax: Any,
        str_scaler_name: str,
        str_x_label: str,
        str_y_label: str,
        ndarray_x_train_scaled: np.ndarray,
        ndarray_x_test_scaled: np.ndarray,
        int_x_idx: int,
        int_y_idx: int,
    ) -> None:
        """散布図の表示範囲と目盛りを見やすい刻みに固定する。"""
        dict_tick_setting = SCALED_PLOT_TICK_SETTINGS.get(
            (str_scaler_name, str_x_label, str_y_label)
        )
        if dict_tick_setting is not None:
            ax.set_xticks(dict_tick_setting["xticks"])
            ax.set_yticks(dict_tick_setting["yticks"])
            float_x_lower, float_x_upper = tuple(dict_tick_setting["xlim"])
            float_y_lower, float_y_upper = tuple(dict_tick_setting["ylim"])
            ndarray_x = np.concatenate(
                [
                    ndarray_x_train_scaled[:, int_x_idx],
                    ndarray_x_test_scaled[:, int_x_idx],
                ]
            )
            ndarray_y = np.concatenate(
                [
                    ndarray_x_train_scaled[:, int_y_idx],
                    ndarray_x_test_scaled[:, int_y_idx],
                ]
            )
            float_x_pad = (float_x_upper - float_x_lower) * 0.04
            float_y_pad = (float_y_upper - float_y_lower) * 0.04
            float_x_lower = min(float_x_lower, float(ndarray_x.min()) - float_x_pad)
            float_x_upper = max(float_x_upper, float(ndarray_x.max()) + float_x_pad)
            float_y_lower = min(float_y_lower, float(ndarray_y.min()) - float_y_pad)
            float_y_upper = max(float_y_upper, float(ndarray_y.max()) + float_y_pad)
            ax.set_xlim(float_x_lower, float_x_upper)
            ax.set_ylim(float_y_lower, float_y_upper)
            return

    def plot_pca(self, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """StandardScaler後にPCAを適用し、結果を可視化する。

        Args:
            n_components (int): 主成分の数。デフォルトは2。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, PCA]: 以下の要素を持つタプル。

                - df_x_scaled (pd.DataFrame): 標準化後の特徴量DataFrame。
                - df_pca (pd.DataFrame): 主成分得点のDataFrame。
                - pca (PCA): 学習済みのPCAインスタンス。
        """
        self._validate_positive_int(
            n_components,
            "n_components",
            upper_bound=self.df_feature.shape[1],
        )

        scaler = StandardScaler()
        ndarray_x_scaled = scaler.fit_transform(self.df_feature)
        df_x_scaled = pd.DataFrame(
            ndarray_x_scaled,
            columns=self.df_feature.columns,
        )

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        ndarray_pca = pca.fit_transform(ndarray_x_scaled)
        list_pc_columns = [
            "PC{}".format(int_idx + 1) for int_idx in range(n_components)
        ]
        df_pca = pd.DataFrame(ndarray_pca, columns=list_pc_columns)

        plt.figure(figsize=(8, 6))
        list_markers = ["o", "^", "v"]
        for int_label, (str_target_name, str_marker) in enumerate(
            zip(self.dataset.target_names, list_markers)
        ):
            ndarray_label_mask = self.dataset.target == int_label
            plt.scatter(
                ndarray_pca[ndarray_label_mask, 0],
                (
                    ndarray_pca[ndarray_label_mask, 1]
                    if n_components >= 2
                    else np.zeros(np.sum(ndarray_label_mask))
                ),
                marker=str_marker,
                s=60,
                label=str_target_name,
            )
        plt.xlabel("First component")
        plt.ylabel("Second component" if n_components >= 2 else "")
        plt.legend(loc="best")
        plt.show()

        list_y_labels = [
            "Component {}".format(int_idx + 1) for int_idx in range(n_components)
        ]

        plt.matshow(pca.components_, cmap="viridis")
        plt.yticks(range(n_components), list_y_labels)
        plt.colorbar()
        plt.xticks(
            range(len(self.df_feature.columns)),
            self.df_feature.columns,
            rotation=60,
            ha="left",
        )
        plt.xlabel("Feature")
        plt.ylabel("PCA components")
        plt.show()

        return (df_x_scaled, df_pca, pca)

    def plot_nmf(self, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, NMF]:
        """MinMaxScaler後にNMFを適用し、結果を可視化する。

        NMFは非負値を要求するため、MinMaxScalerで0以上にスケーリングしてから
        分解を行う。

        Args:
            n_components (int): 成分数。デフォルトは2。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, NMF]: 以下の要素を持つタプル。

                - df_x_scaled (pd.DataFrame): スケーリング後の特徴量DataFrame。
                - df_nmf (pd.DataFrame): NMFで変換した結果のDataFrame。
                - nmf (NMF): 学習済みのNMFインスタンス。
        """
        self._validate_positive_int(
            n_components,
            "n_components",
            upper_bound=self.df_feature.shape[1],
        )

        scaler = MinMaxScaler()
        ndarray_x_scaled = scaler.fit_transform(self.df_feature)
        df_x_scaled = pd.DataFrame(
            ndarray_x_scaled,
            columns=self.df_feature.columns,
        )

        nmf = NMF(
            n_components=n_components,
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
        ndarray_nmf = nmf.fit_transform(ndarray_x_scaled)
        list_nmf_columns = [
            "NMF{}".format(int_idx + 1) for int_idx in range(n_components)
        ]
        df_nmf = pd.DataFrame(ndarray_nmf, columns=list_nmf_columns)

        plt.figure(figsize=(8, 6))
        list_markers = ["o", "^", "v"]
        for int_label, (str_target_name, str_marker) in enumerate(
            zip(self.dataset.target_names, list_markers)
        ):
            ndarray_label_mask = self.dataset.target == int_label
            plt.scatter(
                ndarray_nmf[ndarray_label_mask, 0],
                (
                    ndarray_nmf[ndarray_label_mask, 1]
                    if n_components >= 2
                    else np.zeros(np.sum(ndarray_label_mask))
                ),
                marker=str_marker,
                s=60,
                label=str_target_name,
            )
        plt.xlabel("First component")
        plt.ylabel("Second component" if n_components >= 2 else "")
        plt.legend(loc="best")
        plt.show()

        list_y_labels = [
            "Component {}".format(int_idx + 1) for int_idx in range(n_components)
        ]

        plt.matshow(nmf.components_, cmap="viridis")
        plt.yticks(range(n_components), list_y_labels)
        plt.colorbar()
        plt.xticks(
            range(len(self.df_feature.columns)),
            self.df_feature.columns,
            rotation=60,
            ha="left",
        )
        plt.xlabel("Feature")
        plt.ylabel("NMF components")
        plt.show()

        return (df_x_scaled, df_nmf, nmf)

    def plot_tsne(self) -> None:
        """スケーリングなしのデータにt-SNEを適用し、2次元で可視化する。

        Notes:
            t-SNEは確率的手法のため、結果は ``RANDOM_STATE`` に依存する。
        """
        ndarray_target: np.ndarray = self.dataset.target

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        ndarray_tsne = tsne.fit_transform(self.df_feature)

        plt.figure(figsize=(8, 6))
        plt.xlim(ndarray_tsne[:, 0].min(), ndarray_tsne[:, 0].max())
        plt.ylim(ndarray_tsne[:, 1].min(), ndarray_tsne[:, 1].max())

        for int_idx, str_label in enumerate(ndarray_target.astype(str)):
            plt.text(
                ndarray_tsne[int_idx, 0],
                ndarray_tsne[int_idx, 1],
                str_label,
                fontdict={"weight": "bold", "size": 9},
            )

        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.show()

    def plot_k_means(
        self,
        n_clusters: int | None = None,
        scaling: bool = True,
    ) -> None:
        """KMeans法でクラスタリングし、結果を可視化する。

        Args:
            n_clusters (int | None): クラスタ数。``None`` の場合はIrisの
                クラス数(=3)を使用する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                KMeansを適用する。

        Raises:
            TypeError: ``n_clusters`` が整数以外、または ``scaling`` がbool以外の場合。
            ValueError: ``n_clusters`` が1未満、またはサンプル数を超える場合。
        """
        if n_clusters is None:
            n_clusters = len(self.dataset.target_names)
        self._validate_positive_int(
            n_clusters,
            "n_clusters",
            upper_bound=len(self.df_feature),
        )
        self._validate_bool(scaling, "scaling")

        ndarray_feature = self._get_feature_array(scaling)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        ndarray_cluster_labels = kmeans.fit_predict(ndarray_feature)

        print("KMeans法で予測したラベル:")
        print(ndarray_cluster_labels)

        ndarray_pca_points = self._get_pca_projection(ndarray_feature)

        pca_for_centers = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_for_centers.fit(ndarray_feature)
        ndarray_pca_centers = pca_for_centers.transform(kmeans.cluster_centers_)

        list_colors = ["blue", "red", "green"]
        list_markers = ["o", "^", "v"]

        plt.figure(figsize=(8, 6))
        for int_cluster_id in range(n_clusters):
            ndarray_cluster_mask = ndarray_cluster_labels == int_cluster_id
            plt.scatter(
                ndarray_pca_points[ndarray_cluster_mask, 0],
                ndarray_pca_points[ndarray_cluster_mask, 1],
                c=list_colors[int_cluster_id % len(list_colors)],
                marker=list_markers[int_cluster_id % len(list_markers)],
                s=60,
            )
        plt.scatter(
            ndarray_pca_centers[:, 0],
            ndarray_pca_centers[:, 1],
            c="black",
            marker="*",
            s=400,
        )
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.show()

        print("実際のラベル:")
        print(self.dataset.target)

        plt.figure(figsize=(8, 6))
        for int_label in range(len(self.dataset.target_names)):
            ndarray_label_mask = self.dataset.target == int_label
            plt.scatter(
                ndarray_pca_points[ndarray_label_mask, 0],
                ndarray_pca_points[ndarray_label_mask, 1],
                c=list_colors[int_label % len(list_colors)],
                marker=list_markers[int_label % len(list_markers)],
                s=60,
            )
        plt.scatter(
            ndarray_pca_centers[:, 0],
            ndarray_pca_centers[:, 1],
            c="black",
            marker="*",
            s=400,
        )
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.show()

    def plot_dendrogram(
        self,
        truncate: bool = False,
        scaling: bool = True,
    ) -> None:
        """凝集型階層クラスタリングのデンドログラムを表示する。

        Args:
            truncate (bool): ``True`` の場合は上位の枝のみを表示する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                linkageを計算する。

        Raises:
            TypeError: ``truncate`` または ``scaling`` がbool以外の場合。
        """
        self._validate_bool(truncate, "truncate")
        self._validate_bool(scaling, "scaling")

        ndarray_feature = self._get_feature_array(scaling)
        ndarray_linkage = linkage(ndarray_feature, method="ward")

        plt.figure(figsize=(10, 6))
        if truncate:
            dendrogram(
                ndarray_linkage,
                truncate_mode="lastp",
                p=10,
                show_leaf_counts=True,
                leaf_rotation=90,
            )
        else:
            dendrogram(ndarray_linkage)
        plt.show()

    def plot_dbscan(
        self,
        scaling: bool = False,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        """DBSCANでクラスタリングし、結果を可視化する。

        Args:
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                DBSCANを適用する。
            eps (float): 近傍とみなす距離の上限。
            min_samples (int): コア点とみなすための近傍サンプル数。

        Raises:
            TypeError: ``scaling`` がbool以外、``eps`` が数値以外、
                ``min_samples`` が整数以外の場合。
            ValueError: ``eps`` が0以下、または ``min_samples`` が1未満の場合。
        """
        self._validate_bool(scaling, "scaling")
        self._validate_positive_number(eps, "eps")
        self._validate_positive_int(min_samples, "min_samples")

        ndarray_feature = self._get_feature_array(scaling)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        ndarray_cluster = dbscan.fit_predict(ndarray_feature)

        list_colors = {
            -1: "blue",
            0: "red",
            1: "lime",
            2: "orange",
            3: "purple",
        }
        list_point_colors = [
            list_colors.get(int_cluster, "gray") for int_cluster in ndarray_cluster
        ]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            ndarray_feature[:, 2],
            ndarray_feature[:, 3],
            c=list_point_colors,
            s=60,
        )
        plt.xlabel("Feature 2")
        plt.ylabel("Feature 3")
        plt.show()

        print("Cluster Memberships:", ndarray_cluster)
