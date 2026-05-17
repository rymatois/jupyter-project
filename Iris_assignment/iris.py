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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
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

    def _get_pca_projection(self, feature_array: np.ndarray) -> np.ndarray:
        """特徴量をPCAで2次元に投影する。

        Args:
            feature_array (np.ndarray): 特徴量配列。

        Returns:
            np.ndarray: PCAの第1・第2主成分得点。
        """
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        return pca.fit_transform(feature_array)

    def _calc_cluster_summary(
        self,
        method_name: str,
        feature_array: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> dict[str, Any]:
        """クラスタリング結果の要約指標を計算する。

        Args:
            method_name (str): 手法名。
            feature_array (np.ndarray): クラスタリングに使った特徴量配列。
            cluster_labels (np.ndarray): 推定クラスタラベル。

        Returns:
            dict[str, Any]: クラスタ数、ノイズ数、ARI、シルエット係数。
        """
        unique_labels = set(cluster_labels)
        noise_count = int(np.sum(cluster_labels == -1))
        cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)

        silhouette_score_value: float | None = None
        if len(unique_labels) >= 2 and len(unique_labels) < len(cluster_labels):
            silhouette_score_value = float(
                silhouette_score(feature_array, cluster_labels)
            )

        return {
            "method": method_name,
            "n_clusters": cluster_count,
            "n_noise": noise_count,
            "adjusted_rand_score": float(
                adjusted_rand_score(self.dataset.target, cluster_labels)
            ),
            "silhouette_score": silhouette_score_value,
        }

    def get(self, head: int | None = None) -> pd.DataFrame:
        """ラベル列を付加したDataFrameを返す。

        Args:
            head (int | None): 先頭から返す行数。``None`` の場合は全行を返す。

        Returns:
            pd.DataFrame: ラベル列（Label）を含むDataFrame。
        """
        labeled_df: pd.DataFrame = self.df_feature.copy()
        labeled_df["Label"] = self.dataset.target

        if head is not None:
            return labeled_df.head(head)
        return labeled_df

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

        labeled_df: pd.DataFrame = self.get()
        label_name_by_id: dict[int, str] = {
            int(index): str(label_name)
            for index, label_name in enumerate(self.dataset.target_names)
        }
        labeled_df["LabelName"] = labeled_df["Label"].map(label_name_by_id)

        sns.pairplot(
            labeled_df.drop(columns=["Label"]),
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

        classifiers: dict[str, Any] = DEFAULT_CLASSIFIERS.copy()
        classifiers["KNeighborsClassifier"] = KNeighborsClassifier(
            n_neighbors=n_neighbors,
        )

        feature_df = self.df_feature
        target_array: np.ndarray = self.dataset.target
        results_by_classifier: dict[str, dict[str, np.ndarray]] = {}

        for classifier_name, classifier in classifiers.items():
            results_by_classifier[classifier_name] = cross_validate(
                classifier,
                feature_df,
                target_array,
                cv=CV_SPLITS,
                return_train_score=True,
            )

        return results_by_classifier

    def all_supervised(self, n_neighbors: int = DEFAULT_N_NEIGHBORS) -> None:
        """全分類モデルの交差検証スコアをコンソールに出力する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Raises:
            ValueError: ``n_neighbors`` がint型でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        results_by_classifier = self.calc_supervised_scores(n_neighbors)

        for classifier_name, score_dict in results_by_classifier.items():
            print("== {} ==".format(classifier_name))
            for test_score, train_score in zip(
                score_dict["test_score"],
                score_dict["train_score"],
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
        results_by_classifier = self.calc_supervised_scores(n_neighbors)
        test_scores_by_classifier: dict[str, np.ndarray] = {}

        for classifier_name, score_dict in results_by_classifier.items():
            test_scores_by_classifier[classifier_name] = score_dict["test_score"]

        return pd.DataFrame(test_scores_by_classifier)

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
        score_summary = self.get_supervised(n_neighbors).describe()
        best_method_name: str = str(score_summary.loc["mean"].idxmax())
        best_score: float = float(score_summary.loc["mean"].max())
        return (best_method_name, best_score)

    def plot_feature_importances_all(self) -> None:
        """木ベースの分類モデルの特徴量重要度を棒グラフで表示する。"""
        feature_df = self.df_feature
        target_array: np.ndarray = self.dataset.target

        for classifier_name, classifier in DEFAULT_TREE_CLASSIFIERS.items():
            classifier.fit(feature_df, target_array)

            feature_count: int = feature_df.shape[1]
            plt.figure()
            plt.barh(
                range(feature_count),
                classifier.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(feature_count), feature_df.columns)
            plt.xlabel("Feature importance")
            plt.ylabel("Feature")
            plt.title(classifier_name)
            plt.show()

    def visualize_decision_tree(self) -> list[Any]:
        """決定木の構造を可視化して表示する。

        DecisionTreeClassifierを全データで学習し、木構造を図示する。

        Returns:
            list[Any]: plot_treeが返すArtistオブジェクトのリスト。
        """
        feature_df = self.df_feature
        target_array: np.ndarray = self.dataset.target

        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
        classifier.fit(feature_df, target_array)

        plt.figure(figsize=(16, 10))
        tree_artists = plot_tree(
            classifier,
            feature_names=feature_df.columns,
            class_names=self.dataset.target_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.show()

        return tree_artists

    def plot_scaled_data(self) -> pd.DataFrame:
        """各スケーリング手法でのLinearSVCのスコアと散布図を表示する。

        5分割交差検証の各foldについて、Original / MinMaxScaler /
        StandardScaler / RobustScaler / Normalizer の各スケーリングを適用し、
        LinearSVCのtest/trainスコアと散布図行列を表示する。

        Returns:
            pd.DataFrame: 各fold・各スケーリング手法のtest/trainスコアを
                格納したDataFrame。
        """
        feature_df = self.df_feature
        target_array: np.ndarray = self.dataset.target

        kfold = StratifiedKFold(n_splits=CV_SPLITS, shuffle=False)
        score_records: list[dict[str, Any]] = []

        for fold_index, (train_indices, test_indices) in enumerate(
            kfold.split(feature_df, target_array)
        ):
            x_train = feature_df.iloc[train_indices].to_numpy()
            x_test = feature_df.iloc[test_indices].to_numpy()
            y_train = target_array[train_indices]
            y_test = target_array[test_indices]

            if fold_index > 0:
                print(
                    "========================================================================="
                )

            feature_index_pairs = list(combinations(range(feature_df.shape[1]), 2))
            fig, axes_grid = plt.subplots(
                len(feature_index_pairs),
                len(DEFAULT_SCALERS),
                figsize=(20, 24),
            )

            for scaler_index, (scaler_name, scaler) in enumerate(
                DEFAULT_SCALERS.items()
            ):
                if scaler is None:
                    x_train_scaled = x_train
                    x_test_scaled = x_test
                else:
                    x_train_scaled = scaler.fit_transform(x_train)
                    x_test_scaled = scaler.transform(x_test)

                classifier = LinearSVC(max_iter=10000, random_state=RANDOM_STATE)
                classifier.fit(x_train_scaled, y_train)
                test_score = classifier.score(
                    x_test_scaled,
                    y_test,
                )
                train_score = classifier.score(
                    x_train_scaled,
                    y_train,
                )

                print(
                    "{:<14} :  test score: {:<11.3f}train score: {:<10.3f}".format(
                        scaler_name,
                        test_score,
                        train_score,
                    )
                )

                score_records.append(
                    {
                        "fold": fold_index,
                        "scaler": scaler_name,
                        "test_score": test_score,
                        "train_score": train_score,
                    }
                )

                for pair_index, (x_index, y_index) in enumerate(
                    feature_index_pairs
                ):
                    ax = axes_grid[pair_index, scaler_index]
                    ax.scatter(
                        x_train_scaled[:, x_index],
                        x_train_scaled[:, y_index],
                        c="blue",
                        marker="o",
                        s=35,
                    )
                    ax.scatter(
                        x_test_scaled[:, x_index],
                        x_test_scaled[:, y_index],
                        c="red",
                        marker="^",
                        s=45,
                    )
                    ax.set_title(scaler_name)
                    ax.set_xlabel(feature_df.columns[x_index])
                    ax.set_ylabel(feature_df.columns[y_index])

            fig.tight_layout()
            plt.show()

        print(
            "========================================================================="
        )
        return pd.DataFrame(score_records)

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
        scaled_array = scaler.fit_transform(self.df_feature)
        scaled_df = pd.DataFrame(
            scaled_array,
            columns=self.df_feature.columns,
        )

        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        pca_points = pca.fit_transform(scaled_array)
        pc_column_names = [
            "PC{}".format(component_index + 1)
            for component_index in range(n_components)
        ]
        pca_df = pd.DataFrame(pca_points, columns=pc_column_names)

        plt.figure(figsize=(8, 6))
        label_markers = ["o", "^", "v"]
        for label_index, (target_name, marker) in enumerate(
            zip(self.dataset.target_names, label_markers)
        ):
            label_mask = self.dataset.target == label_index
            plt.scatter(
                pca_points[label_mask, 0],
                (
                    pca_points[label_mask, 1]
                    if n_components >= 2
                    else np.zeros(np.sum(label_mask))
                ),
                marker=marker,
                s=60,
                label=target_name,
            )
        plt.xlabel("First component")
        plt.ylabel("Second component" if n_components >= 2 else "")
        plt.legend(loc="best")
        plt.show()

        component_names = [
            "First component",
            "Second component",
            "Third component",
            "Fourth component",
        ]
        y_axis_labels = [
            component_names[component_index]
            if component_index < len(component_names)
            else "Component {}".format(component_index + 1)
            for component_index in range(n_components)
        ]

        plt.matshow(pca.components_, cmap="viridis")
        plt.yticks(range(n_components), y_axis_labels)
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

        return (scaled_df, pca_df, pca)

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
        scaled_array = scaler.fit_transform(self.df_feature)
        scaled_df = pd.DataFrame(
            scaled_array,
            columns=self.df_feature.columns,
        )

        nmf = NMF(
            n_components=n_components,
            random_state=RANDOM_STATE,
            max_iter=1000,
        )
        nmf_points = nmf.fit_transform(scaled_array)
        nmf_column_names = [
            "NMF{}".format(component_index + 1)
            for component_index in range(n_components)
        ]
        nmf_df = pd.DataFrame(nmf_points, columns=nmf_column_names)

        plt.figure(figsize=(8, 6))
        label_markers = ["o", "^", "v"]
        for label_index, (target_name, marker) in enumerate(
            zip(self.dataset.target_names, label_markers)
        ):
            label_mask = self.dataset.target == label_index
            plt.scatter(
                nmf_points[label_mask, 0],
                (
                    nmf_points[label_mask, 1]
                    if n_components >= 2
                    else np.zeros(np.sum(label_mask))
                ),
                marker=marker,
                s=60,
                label=target_name,
            )
        plt.xlabel("First component")
        plt.ylabel("Second component" if n_components >= 2 else "")
        plt.legend(loc="best")
        plt.show()

        component_names = [
            "First component",
            "Second component",
            "Third component",
            "Fourth component",
        ]
        y_axis_labels = [
            component_names[component_index]
            if component_index < len(component_names)
            else "Component {}".format(component_index + 1)
            for component_index in range(n_components)
        ]

        plt.matshow(nmf.components_, cmap="viridis")
        plt.yticks(range(n_components), y_axis_labels)
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

        return (scaled_df, nmf_df, nmf)

    def plot_tsne(self) -> None:
        """スケーリングなしのデータにt-SNEを適用し、2次元で可視化する。

        Notes:
            t-SNEは確率的手法のため、結果は ``RANDOM_STATE`` に依存する。
        """
        target_array: np.ndarray = self.dataset.target

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        tsne_points = tsne.fit_transform(self.df_feature)

        plt.figure(figsize=(8, 6))
        plt.xlim(tsne_points[:, 0].min(), tsne_points[:, 0].max())
        plt.ylim(tsne_points[:, 1].min(), tsne_points[:, 1].max())

        for point_index, label in enumerate(target_array.astype(str)):
            plt.text(
                tsne_points[point_index, 0],
                tsne_points[point_index, 1],
                label,
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

        feature_array = self._get_feature_array(scaling)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(feature_array)

        print("KMeans法で予測したラベル:")
        print(cluster_labels)

        pca_points = self._get_pca_projection(feature_array)

        pca_for_centers = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_for_centers.fit(feature_array)
        pca_centers = pca_for_centers.transform(kmeans.cluster_centers_)

        cluster_colors = ["blue", "red", "green"]
        cluster_markers = ["o", "^", "v"]

        plt.figure(figsize=(8, 6))
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            plt.scatter(
                pca_points[cluster_mask, 0],
                pca_points[cluster_mask, 1],
                c=cluster_colors[cluster_id % len(cluster_colors)],
                marker=cluster_markers[cluster_id % len(cluster_markers)],
                s=60,
            )
        plt.scatter(
            pca_centers[:, 0],
            pca_centers[:, 1],
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
        for label_index in range(len(self.dataset.target_names)):
            label_mask = self.dataset.target == label_index
            plt.scatter(
                pca_points[label_mask, 0],
                pca_points[label_mask, 1],
                c=cluster_colors[label_index % len(cluster_colors)],
                marker=cluster_markers[label_index % len(cluster_markers)],
                s=60,
            )
        plt.scatter(
            pca_centers[:, 0],
            pca_centers[:, 1],
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

        feature_array = self._get_feature_array(scaling)
        linkage_matrix = linkage(feature_array, method="ward")

        plt.figure(figsize=(10, 6))
        if truncate:
            dendrogram(
                linkage_matrix,
                truncate_mode="lastp",
                p=10,
                show_leaf_counts=True,
                leaf_rotation=90,
            )
        else:
            dendrogram(linkage_matrix)
        plt.show()

    def plot_dbscan(
        self,
        scaling: bool = True,
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

        feature_array = self._get_feature_array(scaling)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(feature_array)

        color_by_label = {
            -1: "blue",
            0: "red",
            1: "lime",
            2: "orange",
            3: "purple",
        }
        point_colors = [
            color_by_label.get(cluster_id, "gray") for cluster_id in cluster_labels
        ]

        plt.figure(figsize=(8, 6))
        plt.scatter(
            feature_array[:, 2],
            feature_array[:, 3],
            c=point_colors,
            s=60,
        )
        plt.xlabel("Feature 2")
        plt.ylabel("Feature 3")
        plt.show()

        print("Cluster Memberships:", cluster_labels)

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

        feature_array = self._get_feature_array(scaling)

        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        kmeans_labels = kmeans.fit_predict(feature_array)

        linkage_matrix = linkage(feature_array, method="ward")
        hierarchical_labels = (
            fcluster(linkage_matrix, n_clusters, criterion="maxclust") - 1
        )

        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        dbscan_labels = dbscan.fit_predict(feature_array)

        summary_rows = [
            self._calc_cluster_summary("KMeans", feature_array, kmeans_labels),
            self._calc_cluster_summary(
                "Dendrogram/Ward",
                feature_array,
                hierarchical_labels,
            ),
            self._calc_cluster_summary("DBSCAN", feature_array, dbscan_labels),
        ]

        return pd.DataFrame(summary_rows)

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

    def _plot_2d_with_labels(
        self,
        points_2d: np.ndarray,
        labels: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        """2次元データをラベル別に色分けして散布図表示する内部ヘルパー。

        Args:
            points_2d (np.ndarray): shape=(n_samples, 2) の座標配列。
            labels (np.ndarray): クラスラベル配列。
            title (str): グラフタイトル。
            xlabel (str): X軸ラベル。
            ylabel (str): Y軸ラベル。
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(
            points_2d[:, 0],
            points_2d[:, 1] if points_2d.shape[1] >= 2 else np.zeros(len(points_2d)),
            c=labels,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
