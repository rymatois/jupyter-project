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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_validate
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
            pd.DataFrame: 各fold・各スケーリング手法のtest/trainスコアを
                格納したDataFrame。
        """
        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target

        kfold = StratifiedKFold(n_splits=CV_SPLITS, shuffle=False)
        list_records: list[dict[str, Any]] = []

        for int_fold, (ndarray_train_idx, ndarray_test_idx) in enumerate(
            kfold.split(df_feature, ndarray_target)
        ):
            ndarray_x_train = df_feature.iloc[ndarray_train_idx].to_numpy()
            ndarray_x_test = df_feature.iloc[ndarray_test_idx].to_numpy()
            ndarray_y_train = ndarray_target[ndarray_train_idx]
            ndarray_y_test = ndarray_target[ndarray_test_idx]

            if int_fold > 0:
                print(
                    "========================================================================="
                )

            fig, list_axes = plt.subplots(2, 3, figsize=(15, 8))
            list_axes_flat = list_axes.flatten()

            for int_idx, (str_scaler_name, scaler) in enumerate(
                DEFAULT_SCALERS.items()
            ):
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

                list_records.append(
                    {
                        "fold": int_fold,
                        "scaler": str_scaler_name,
                        "test_score": float_test_score,
                        "train_score": float_train_score,
                    }
                )

                ax = list_axes_flat[int_idx]
                scatter = ax.scatter(
                    ndarray_x_train_scaled[:, 0],
                    ndarray_x_train_scaled[:, 1],
                    c=ndarray_y_train,
                    cmap="viridis",
                    edgecolor="k",
                    s=40,
                )
                ax.set_title("fold={} / {}".format(int_fold, str_scaler_name))
                ax.set_xlabel(df_feature.columns[0])
                ax.set_ylabel(df_feature.columns[1])

            list_axes_flat[5].axis("off")
            fig.colorbar(scatter, ax=list_axes_flat[:5], shrink=0.8)
            fig.tight_layout()
            plt.show()

        print(
            "========================================================================="
        )
        return pd.DataFrame(list_records)

    def plot_k_means(
        self,
        n_clusters: int | None = None,
        scaling: bool = True,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """KMeans法でクラスタリングし、結果を可視化する。

        Args:
            n_clusters (int | None): クラスタ数。``None`` の場合はIrisの
                クラス数(=3)を使用する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                KMeansを適用する。

        Returns:
            tuple[np.ndarray, pd.DataFrame]: 推定クラスタラベルと要約指標。

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
        ndarray_cluster = kmeans.fit_predict(ndarray_feature)

        print("KMeans法で予測したラベル:")
        print(ndarray_cluster)
        print()
        print("実際のラベル:")
        print(self.dataset.target)

        ndarray_pca = self._get_pca_projection(ndarray_feature)
        fig, list_axes = plt.subplots(1, 2, figsize=(12, 5))
        list_axes[0].scatter(
            ndarray_pca[:, 0],
            ndarray_pca[:, 1],
            c=ndarray_cluster,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        list_axes[0].set_title("KMeans labels (PCA projection)")
        list_axes[0].set_xlabel("PC1")
        list_axes[0].set_ylabel("PC2")

        list_axes[1].scatter(
            ndarray_pca[:, 0],
            ndarray_pca[:, 1],
            c=self.dataset.target,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        list_axes[1].set_title("True labels (PCA projection)")
        list_axes[1].set_xlabel("PC1")
        list_axes[1].set_ylabel("PC2")

        fig.tight_layout()
        plt.show()

        df_summary = pd.DataFrame(
            [self._calc_cluster_summary("KMeans", ndarray_feature, ndarray_cluster)]
        )
        return (ndarray_cluster, df_summary)

    def plot_dendrogram(
        self,
        truncate: bool = False,
        scaling: bool = True,
    ) -> np.ndarray:
        """凝集型階層クラスタリングのデンドログラムを表示する。

        Args:
            truncate (bool): ``True`` の場合は上位の枝のみを表示する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                linkageを計算する。

        Returns:
            np.ndarray: scipyのlinkage行列。

        Raises:
            TypeError: ``truncate`` または ``scaling`` がbool以外の場合。
        """
        self._validate_bool(truncate, "truncate")
        self._validate_bool(scaling, "scaling")

        ndarray_feature = self._get_feature_array(scaling)
        ndarray_linkage = linkage(ndarray_feature, method="ward")

        plt.figure(figsize=(15, 6))
        if truncate:
            dendrogram(
                ndarray_linkage,
                truncate_mode="lastp",
                p=12,
                show_leaf_counts=True,
            )
            plt.title("Dendrogram (truncated)")
        else:
            dendrogram(ndarray_linkage)
            plt.title("Dendrogram")
        plt.xlabel("Sample index")
        plt.ylabel("Cluster distance")
        plt.show()

        return ndarray_linkage

    def plot_dbscan(
        self,
        scaling: bool = True,
        eps: float = 0.8,
        min_samples: int = 5,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """DBSCANでクラスタリングし、結果を可視化する。

        Args:
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                DBSCANを適用する。
            eps (float): 近傍とみなす距離の上限。
            min_samples (int): コア点とみなすための近傍サンプル数。

        Returns:
            tuple[np.ndarray, pd.DataFrame]: 推定クラスタラベルと要約指標。

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

        print("DBSCANで予測したクラスタ:")
        print(ndarray_cluster)
        print("※ -1 はノイズとして判定されたデータです。")

        ndarray_pca = self._get_pca_projection(ndarray_feature)
        plt.figure(figsize=(8, 6))
        plt.scatter(
            ndarray_pca[:, 0],
            ndarray_pca[:, 1],
            c=ndarray_cluster,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        plt.title(
            "DBSCAN (scaling={}, eps={}, min_samples={})".format(
                scaling,
                eps,
                min_samples,
            )
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

        df_summary = pd.DataFrame(
            [self._calc_cluster_summary("DBSCAN", ndarray_feature, ndarray_cluster)]
        )
        return (ndarray_cluster, df_summary)

    def compare_unsupervised(
        self,
        n_clusters: int | None = None,
        dbscan_eps: float = 0.8,
        dbscan_min_samples: int = 5,
        scaling: bool = True,
    ) -> pd.DataFrame:
        """KMeans、階層クラスタリング、DBSCANの結果を指標で比較する。

        Args:
            n_clusters (int | None): KMeansと階層クラスタリングのクラスタ数。
                ``None`` の場合はIrisのクラス数(=3)を使用する。
            dbscan_eps (float): DBSCANのeps。
            dbscan_min_samples (int): DBSCANのmin_samples。
            scaling (bool): ``True`` の場合はStandardScalerで標準化する。

        Returns:
            pd.DataFrame: 手法ごとのクラスタ数、ノイズ数、ARI、
                シルエット係数をまとめたDataFrame。
        """
        if n_clusters is None:
            n_clusters = len(self.dataset.target_names)
        self._validate_positive_int(
            n_clusters,
            "n_clusters",
            upper_bound=len(self.df_feature),
        )
        self._validate_positive_number(dbscan_eps, "dbscan_eps")
        self._validate_positive_int(dbscan_min_samples, "dbscan_min_samples")
        self._validate_bool(scaling, "scaling")

        ndarray_feature = self._get_feature_array(scaling)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        ndarray_kmeans = kmeans.fit_predict(ndarray_feature)

        ndarray_linkage = linkage(ndarray_feature, method="ward")
        ndarray_hierarchical = (
            fcluster(ndarray_linkage, n_clusters, criterion="maxclust") - 1
        )

        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        ndarray_dbscan = dbscan.fit_predict(ndarray_feature)

        list_summary = [
            self._calc_cluster_summary("KMeans", ndarray_feature, ndarray_kmeans),
            self._calc_cluster_summary(
                "Dendrogram/Ward",
                ndarray_feature,
                ndarray_hierarchical,
            ),
            self._calc_cluster_summary("DBSCAN", ndarray_feature, ndarray_dbscan),
        ]

        return pd.DataFrame(list_summary)

    def unsupervised_consideration(self) -> pd.DataFrame:
        """3章最終課題用に、3手法の使い分けを表で返す。

        Returns:
            pd.DataFrame: KMeans、Dendrogram、DBSCANの特徴・向いている用途・
                注意点をまとめたDataFrame。
        """
        return pd.DataFrame(
            [
                {
                    "method": "KMeans",
                    "summary": "クラスタ数を先に決め、重心に近いデータ同士をまとめる。",
                    "good_for": "球状に近いクラスタを、指定した数に分けたい場合。",
                    "caution": "クラスタ数の指定が必要で、外れ値や非球状クラスタには弱い。",
                },
                {
                    "method": "Dendrogram/Ward",
                    "summary": "近いデータやクラスタを順に結合し、階層構造として見る。",
                    "good_for": "クラスタ数を決める前に、データのまとまり方を観察したい場合。",
                    "caution": "サンプル数が多いと図が読みづらく、切る高さの判断が必要。",
                },
                {
                    "method": "DBSCAN",
                    "summary": "密度が高い領域をクラスタとし、疎な点をノイズにする。",
                    "good_for": "外れ値を検出したい場合や、クラスタ数を事前に決めたくない場合。",
                    "caution": "epsとmin_samplesの影響が大きく、Irisでは2クラスタにまとまりやすい。",
                },
            ]
        )
