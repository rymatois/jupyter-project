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
from collections.abc import Sequence
from itertools import combinations
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
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

DBSCAN_COLORS: dict[int, str] = {
    -1: "blue",
    0: "red",
    1: "lime",
    2: "orange",
    3: "purple",
}
"""DBSCANのクラスタ可視化で使う色。``-1`` はノイズを表す。"""


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

    def _get_features_by_scaler(
        self,
        scaler_name: str | None = "StandardScaler",
    ) -> np.ndarray:
        """指定したスケーラーで特徴量をクラスタリング用の配列に変換する。

        Args:
            scaler_name (str | None): 使用するスケーラー名。``DEFAULT_SCALERS``
                のキーを指定する。``"Original"`` または ``None`` の場合は
                スケーリングしない。

        Returns:
            np.ndarray: 特徴量配列。
        """
        if scaler_name is None:
            return self.df_feature.to_numpy()

        scaler = DEFAULT_SCALERS[scaler_name]
        if scaler is None:
            return self.df_feature.to_numpy()
        return clone(scaler).fit_transform(self.df_feature)

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
        """
        df_labeled: pd.DataFrame = self.get()
        label_names: dict[int, str] = {
            int(index): str(label_name)
            for index, label_name in enumerate(self.dataset.target_names)
        }
        df_labeled["LabelName"] = df_labeled["Label"].map(label_names)

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
        """
        classifiers: dict[str, Any] = DEFAULT_CLASSIFIERS.copy()
        classifiers["KNeighborsClassifier"] = KNeighborsClassifier(
            n_neighbors=n_neighbors,
        )

        df_feature = self.df_feature
        target: np.ndarray = self.dataset.target
        results: dict[str, dict[str, np.ndarray]] = {}

        for classifier_name, classifier in classifiers.items():
            results[classifier_name] = cross_validate(
                classifier,
                df_feature,
                target,
                cv=CV_SPLITS,
                return_train_score=True,
                error_score="raise",
            )

        return results

    def all_supervised(self, n_neighbors: int = DEFAULT_N_NEIGHBORS) -> None:
        """全分類モデルの交差検証スコアをコンソールに出力する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。
        """
        results = self.calc_supervised_scores(n_neighbors)

        for classifier_name, score in results.items():
            print("== {} ==".format(classifier_name))
            for test_score, train_score in zip(
                score["test_score"],
                score["train_score"],
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
        """
        results = self.calc_supervised_scores(n_neighbors)
        test_scores: dict[str, np.ndarray] = {}

        for classifier_name, score in results.items():
            test_scores[classifier_name] = score["test_score"]

        return pd.DataFrame(test_scores)

    def best_supervised(
        self,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> tuple[str, float]:
        """平均テストスコアが最も高い分類モデルを返す。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            tuple[str, float]: 以下の要素を持つタプル。

                - best_method (str): 最良モデルの名前。
                - best_score (float): 最良モデルの平均テストスコア。
        """
        df_score = self.get_supervised(n_neighbors).describe()
        best_method: str = str(df_score.loc["mean"].idxmax())
        best_score: float = float(df_score.loc["mean"].max())
        return (best_method, best_score)

    def plot_feature_importances_all(self) -> None:
        """木ベースの分類モデルの特徴量重要度を棒グラフで表示する。"""
        df_feature = self.df_feature
        target: np.ndarray = self.dataset.target

        for classifier_name, classifier in DEFAULT_TREE_CLASSIFIERS.items():
            classifier.fit(df_feature, target)

            feature_count: int = df_feature.shape[1]
            plt.figure()
            plt.barh(
                range(feature_count),
                classifier.feature_importances_,
                align="center",
            )
            plt.yticks(np.arange(feature_count), df_feature.columns)
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
        df_feature = self.df_feature
        target: np.ndarray = self.dataset.target

        classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
        classifier.fit(df_feature, target)

        plt.figure(figsize=(16, 10))
        tree_artists = plot_tree(
            classifier,
            feature_names=df_feature.columns,
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
            pd.DataFrame: 各スケーリング手法のtest/trainスコアの平均と
                標準偏差をまとめたDataFrame。
        """
        df_feature = self.df_feature
        target: np.ndarray = self.dataset.target

        kfold = StratifiedKFold(n_splits=CV_SPLITS, shuffle=False)

        records: list[dict[str, Any]] = []
        # recordsの各要素:
        # {"fold": fold番号, "scaler": スケーラー名,
        #  "test_score": テストスコア, "train_score": 学習スコア}

        for fold, (train_idx, test_idx) in enumerate(kfold.split(df_feature, target)):
            if fold > 0:
                print(
                    "========================================================================="
                )

            records.extend(
                self._process_scaled_data_fold(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    fold=fold,
                )
            )

        print(
            "========================================================================="
        )
        df_scaled_scores = pd.DataFrame(records)
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
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        fold: int,
    ) -> list[dict[str, Any]]:
        """plot_scaled_dataの1 fold分を評価・描画する。

        Args:
            train_idx (np.ndarray): 学習データの行番号配列。
            test_idx (np.ndarray): テストデータの行番号配列。
            fold (int): 現在のfold番号。

        Returns:
            list[dict[str, Any]]: 各スケーラーの評価結果を表す辞書のリスト。
        """
        df_feature = self.df_feature
        target = self.dataset.target
        x_train = df_feature.iloc[train_idx].to_numpy()
        x_test = df_feature.iloc[test_idx].to_numpy()
        y_train = target[train_idx]
        y_test = target[test_idx]

        feature_pairs = list(combinations(range(df_feature.shape[1]), 2))
        fig, axes = plt.subplots(
            len(feature_pairs),
            len(DEFAULT_SCALERS),
            figsize=(15, 26),
        )
        scaled_results = self._build_scaled_fold_results(
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test,
        )

        self._plot_scaled_fold(
            axes=axes,
            feature_pairs=feature_pairs,
            scaled_results=scaled_results,
            feature_names=df_feature.columns.tolist(),
        )

        fig.tight_layout()
        plt.show()

        return [
            {
                "fold": fold,
                "scaler": scaled_result["scaler"],
                "test_score": scaled_result["test_score"],
                "train_score": scaled_result["train_score"],
            }
            for scaled_result in scaled_results
        ]

    def _build_scaled_fold_results(
        self,
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> list[dict[str, Any]]:
        """各スケーラーでの評価結果と描画用データを作成する。

        Args:
            x_train (np.ndarray): 学習用特徴量配列。
            x_test (np.ndarray): テスト用特徴量配列。
            y_train (np.ndarray): 学習用ラベル配列。
            y_test (np.ndarray): テスト用ラベル配列。

        Returns:
            list[dict[str, Any]]: スケーラー名、変換後データ、評価スコアを含む
                辞書のリスト。
        """
        scaled_results: list[dict[str, Any]] = []

        for scaler_name, scaler in DEFAULT_SCALERS.items():
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

            scaled_results.append(
                {
                    "scaler": scaler_name,
                    "x_train_scaled": x_train_scaled,
                    "x_test_scaled": x_test_scaled,
                    "test_score": test_score,
                    "train_score": train_score,
                }
            )

        return scaled_results

    def _plot_scaled_fold(
        self,
        axes: np.ndarray,
        feature_pairs: list[tuple[int, int]],
        scaled_results: list[dict[str, Any]],
        feature_names: list[str],
    ) -> None:
        """1 fold分のスケーリング結果を散布図行列として描画する。

        Args:
            axes (np.ndarray): 描画先のAxes配列。
            feature_pairs (list[tuple[int, int]]): 描画する特徴量ペアの
                インデックス一覧。
            scaled_results (list[dict[str, Any]]): 各スケーラーの
                変換結果と評価結果。
            feature_names (list[str]): 特徴量名の一覧。
        """
        for idx, scaled_result in enumerate(scaled_results):
            for pair_idx, (x_idx, y_idx) in enumerate(feature_pairs):
                ax = axes[pair_idx, idx]
                ax.scatter(
                    scaled_result["x_train_scaled"][:, x_idx],
                    scaled_result["x_train_scaled"][:, y_idx],
                    c="blue",
                    marker="o",
                    s=98,
                )
                ax.scatter(
                    scaled_result["x_test_scaled"][:, x_idx],
                    scaled_result["x_test_scaled"][:, y_idx],
                    c="red",
                    marker="^",
                    s=118,
                )
                x_label = feature_names[x_idx]
                y_label = feature_names[y_idx]
                ax.set_title(scaled_result["scaler"], fontsize=15, pad=6)
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel(y_label, fontsize=12)
                ax.tick_params(labelsize=12)
                ax.set_box_aspect(1.65)

    def _plot_transformed_scatter(
        self,
        transformed: np.ndarray,
        n_components: int,
    ) -> None:
        """変換後の特徴量を2次元散布図で描画する。

        Args:
            transformed (np.ndarray): 変換後の特徴量配列。
            n_components (int): 変換後の成分数。
        """
        plt.figure(figsize=(8, 6))
        markers = ["o", "^", "v"]
        for label_id, (target_name, marker) in enumerate(
            zip(self.dataset.target_names, markers)
        ):
            label_mask = self.dataset.target == label_id
            plt.scatter(
                transformed[label_mask, 0],
                (
                    transformed[label_mask, 1]
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

    def _plot_component_matrix(
        self,
        transformer: Any,
        n_components: int,
        ylabel: str,
    ) -> None:
        """学習済み変換器の成分行列を可視化する。

        Args:
            transformer (Any): 学習済み変換器。
            n_components (int): 表示する成分数。
            ylabel (str): y軸ラベル。
        """
        y_labels = ["Component {}".format(idx + 1) for idx in range(n_components)]

        plt.matshow(transformer.components_, cmap="viridis")
        plt.yticks(range(n_components), y_labels)
        plt.colorbar()
        plt.xticks(
            range(len(self.df_feature.columns)),
            self.df_feature.columns,
            rotation=60,
            ha="left",
        )
        plt.xlabel("Feature")
        plt.ylabel(ylabel)
        plt.show()

    def _transform_and_plot_components(
        self,
        scaler_name: str,
        transformer: Any,
        n_components: int,
        column_prefix: str,
        component_ylabel: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, Any]:
        """スケーリング後の変換・可視化・返り値生成を共通化する。

        Args:
            scaler_name (str): 前処理に使うスケーラー名。
            transformer (Any): 特徴量変換に使う学習器。
            n_components (int): 生成する成分数。
            column_prefix (str): 変換後列名の接頭辞。
            component_ylabel (str): 成分行列可視化時のy軸ラベル。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, Any]: スケーリング後の特徴量
                DataFrame、変換後DataFrame、学習済み変換器のタプル。
        """
        x_scaled = self._get_features_by_scaler(scaler_name)
        df_x_scaled = pd.DataFrame(
            x_scaled,
            columns=self.df_feature.columns,
        )
        transformed = transformer.fit_transform(x_scaled)
        columns = ["{}{}".format(column_prefix, idx + 1) for idx in range(n_components)]
        df_transformed = pd.DataFrame(transformed, columns=columns)
        self._plot_transformed_scatter(transformed, n_components)
        self._plot_component_matrix(
            transformer=transformer,
            n_components=n_components,
            ylabel=component_ylabel,
        )
        return (df_x_scaled, df_transformed, transformer)

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
        df_x_scaled, df_pca, fitted_transformer = self._transform_and_plot_components(
            scaler_name="StandardScaler",
            transformer=PCA(
                n_components=n_components,
                random_state=RANDOM_STATE,
            ),
            n_components=n_components,
            column_prefix="PC",
            component_ylabel="PCA components",
        )
        return (df_x_scaled, df_pca, fitted_transformer)

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
        df_x_scaled, df_nmf, fitted_transformer = self._transform_and_plot_components(
            scaler_name="MinMaxScaler",
            transformer=NMF(
                n_components=n_components,
                random_state=RANDOM_STATE,
                max_iter=1000,
            ),
            n_components=n_components,
            column_prefix="NMF",
            component_ylabel="NMF components",
        )
        return (df_x_scaled, df_nmf, fitted_transformer)

    def plot_tsne(self) -> None:
        """スケーリングなしのデータにt-SNEを適用し、2次元で可視化する。

        Notes:
            t-SNEは確率的手法のため、結果は ``RANDOM_STATE`` に依存する。
        """
        target: np.ndarray = self.dataset.target

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        tsne_embedding = tsne.fit_transform(self.df_feature)

        plt.figure(figsize=(8, 6))
        plt.xlim(tsne_embedding[:, 0].min(), tsne_embedding[:, 0].max())
        plt.ylim(tsne_embedding[:, 1].min(), tsne_embedding[:, 1].max())

        for idx, label in enumerate(target.astype(str)):
            plt.text(
                tsne_embedding[idx, 0],
                tsne_embedding[idx, 1],
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
        scaler_name: str | None = None,
    ) -> None:
        """KMeans法でクラスタリングし、結果を可視化する。

        Args:
            n_clusters (int | None): クラスタ数。``None`` の場合はIrisの
                クラス数(=3)を使用する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                KMeansを適用する。
            scaler_name (str | None): ``DEFAULT_SCALERS`` のキーを指定すると、
                ``scaling`` より優先してそのスケーラーを使用する。
        """
        if n_clusters is None:
            n_clusters = len(self.dataset.target_names)

        if scaler_name is None:
            scaler_name = "StandardScaler" if scaling else "Original"
        feature_array = self._get_features_by_scaler(scaler_name)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        cluster_labels = kmeans.fit_predict(feature_array)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        pca_points = pca.fit_transform(feature_array)
        pca_centers = pca.transform(kmeans.cluster_centers_)

        print("KMeans法で予測したラベル:")
        print(cluster_labels)
        print()
        print("実際のラベル:")
        print(self.dataset.target)

        colors = ["blue", "red", "green"]
        markers = ["o", "^", "v"]
        plot_settings = [
            ("KMeans labels", cluster_labels, kmeans.n_clusters),
            ("True labels", self.dataset.target, len(self.dataset.target_names)),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (title, labels, group_count) in zip(
            axes,
            plot_settings,
        ):
            for label_id in range(group_count):
                label_mask = labels == label_id
                ax.scatter(
                    pca_points[label_mask, 0],
                    pca_points[label_mask, 1],
                    c=colors[label_id],
                    marker=markers[label_id],
                    s=60,
                )
            ax.scatter(
                pca_centers[:, 0],
                pca_centers[:, 1],
                c="black",
                marker="*",
                s=400,
            )
            ax.set_title(title)
            ax.set_xlabel("First principal component")
            ax.set_ylabel("Second principal component")

        fig.tight_layout()
        plt.show()

    def plot_dendrogram(
        self,
        truncate: bool = False,
        scaling: bool = True,
        scaler_name: str | None = None,
    ) -> None:
        """凝集型階層クラスタリングのデンドログラムを表示する。

        Args:
            truncate (bool): ``True`` の場合は上位の枝のみを表示する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                linkageを計算する。
            scaler_name (str | None): ``DEFAULT_SCALERS`` のキーを指定すると、
                ``scaling`` より優先してそのスケーラーを使用する。
        """
        if scaler_name is None:
            scaler_name = "StandardScaler" if scaling else "Original"
        feature_array = self._get_features_by_scaler(scaler_name)
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
        scaling: bool = False,
        eps: float = 0.5,
        min_samples: int = 5,
        scaler_name: str | None = None,
    ) -> None:
        """DBSCANでクラスタリングし、結果を可視化する。

        Args:
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                DBSCANを適用する。
            eps (float): 近傍とみなす距離の上限。
            min_samples (int): コア点とみなすための近傍サンプル数。
            scaler_name (str | None): ``DEFAULT_SCALERS`` のキーを指定すると、
                ``scaling`` より優先してそのスケーラーを使用する。
        """
        if scaler_name is None:
            scaler_name = "StandardScaler" if scaling else "Original"
        feature_array = self._get_features_by_scaler(scaler_name)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(feature_array)

        point_colors = [
            DBSCAN_COLORS.get(cluster, "gray") for cluster in cluster_labels
        ]

        x_feature_index = 2
        y_feature_index = 3
        plt.figure(figsize=(8, 6))
        plt.scatter(
            feature_array[:, x_feature_index],
            feature_array[:, y_feature_index],
            c=point_colors,
            s=60,
        )
        plt.xlabel(self.df_feature.columns[x_feature_index])
        plt.ylabel(self.df_feature.columns[y_feature_index])
        plt.show()

        print("Cluster Memberships:", cluster_labels)

    def compare_dbscan(
        self,
        scaling: bool = False,
        eps: float = 0.5,
        min_samples: int = 5,
        scaler_name: str | None = None,
    ) -> None:
        """DBSCANのクラスタと正解ラベルを同じ形式の図で比較する。

        Args:
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                DBSCANを適用する。
            eps (float): 近傍とみなす距離の上限。
            min_samples (int): コア点とみなすための近傍サンプル数。
            scaler_name (str | None): ``DEFAULT_SCALERS`` のキーを指定すると、
                ``scaling`` より優先してそのスケーラーを使用する。
        """
        if scaler_name is None:
            scaler_name = "StandardScaler" if scaling else "Original"
        feature_array = self._get_features_by_scaler(scaler_name)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(feature_array)

        cluster_colors = [
            DBSCAN_COLORS.get(cluster, "gray") for cluster in cluster_labels
        ]
        target_colors = [
            DBSCAN_COLORS.get(label, "gray") for label in self.dataset.target
        ]

        x_feature_index = 2
        y_feature_index = 3
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        plot_settings = [
            ("DBSCAN clusters", cluster_colors),
            ("True labels", target_colors),
        ]
        for ax, (title, colors) in zip(axes, plot_settings):
            ax.scatter(
                feature_array[:, x_feature_index],
                feature_array[:, y_feature_index],
                c=colors,
                s=60,
            )
            ax.set_title(title)
            ax.set_xlabel(self.df_feature.columns[x_feature_index])
            ax.set_ylabel(self.df_feature.columns[y_feature_index])

        fig.tight_layout()
        plt.show()

    def plot_dbscan_parameter_grid(
        self,
        eps_values: Sequence[float] | None = None,
        min_samples_values: Sequence[int] | None = None,
        scaling: bool = True,
        scaler_name: str | None = None,
    ) -> None:
        """DBSCANのepsとmin_samplesを変えた結果をグリッド表示する。

        Args:
            eps_values (Sequence[float] | None): 比較するepsの値。``None`` の
                場合は4種類のデフォルト値を使用する。
            min_samples_values (Sequence[int] | None): 比較するmin_samplesの値。
                ``None`` の場合は4種類のデフォルト値を使用する。
            scaling (bool): ``True`` の場合はStandardScalerで標準化してから
                DBSCANを適用する。
            scaler_name (str | None): ``DEFAULT_SCALERS`` のキーを指定すると、
                ``scaling`` より優先してそのスケーラーを使用する。
        """
        if eps_values is None:
            eps_values = (0.3, 0.4, 0.5)
        if min_samples_values is None:
            min_samples_values = (3, 5, 6)
        if scaler_name is None:
            scaler_name = "StandardScaler" if scaling else "Original"

        feature_array = self._get_features_by_scaler(scaler_name)
        fig, axes = plt.subplots(
            len(min_samples_values),
            len(eps_values),
            figsize=(16, 16),
            sharex=True,
            sharey=True,
        )

        x_feature_index = 2
        y_feature_index = 3
        for row_idx, min_samples in enumerate(min_samples_values):
            for col_idx, eps in enumerate(eps_values):
                ax = axes[row_idx, col_idx]
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = dbscan.fit_predict(feature_array)
                point_colors = [
                    DBSCAN_COLORS.get(cluster, "gray") for cluster in cluster_labels
                ]

                unique_labels = set(cluster_labels)
                cluster_count = len(unique_labels) - (1 if -1 in unique_labels else 0)
                noise_count = int(np.sum(cluster_labels == -1))

                ax.scatter(
                    feature_array[:, x_feature_index],
                    feature_array[:, y_feature_index],
                    c=point_colors,
                    s=35,
                )
                ax.set_title(
                    "eps={}, min_samples={}\nclusters={}, noise={}".format(
                        eps,
                        min_samples,
                        cluster_count,
                        noise_count,
                    ),
                    fontsize=10,
                )
                if row_idx == len(min_samples_values) - 1:
                    ax.set_xlabel(self.df_feature.columns[x_feature_index])
                if col_idx == 0:
                    ax.set_ylabel(self.df_feature.columns[y_feature_index])

        fig.suptitle(
            "DBSCAN parameter comparison (scaler={})".format(scaler_name),
            fontsize=14,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show()
