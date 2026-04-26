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
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import NMF, PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    MinMaxScaler,
    Normalizer,
    RobustScaler,
    StandardScaler,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import Bunch

pd.set_option("display.max_rows", None)
warnings.filterwarnings("ignore")

RANDOM_STATE: int = 0
"""再現性を保つために全モデルで共通利用する乱数シード。"""

CV_SPLITS: int = 5
"""交差検証の分割数。"""

IRIS_DATASET: Bunch = load_iris()

DEFAULT_CLASSIFIERS: list[tuple[str, Any]] = [
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("LinearSVC", LinearSVC(max_iter=10000, random_state=RANDOM_STATE)),
    ("SVC", SVC()),
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=4)),
    ("LinearRegression", LinearRegression()),
    ("RandomForestClassifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    (
        "GradientBoostingClassifier",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
    ),
    ("MLPClassifier", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE)),
]

DEFAULT_TREE_CLASSIFIERS: list[tuple[str, Any]] = [
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=RANDOM_STATE)),
    ("RandomForestClassifier", RandomForestClassifier(random_state=RANDOM_STATE)),
    (
        "GradientBoostingClassifier",
        GradientBoostingClassifier(random_state=RANDOM_STATE),
    ),
]

DEFAULT_SCALERS: list[tuple[str, Any]] = [
    ("Original", None),
    ("MinMaxScaler", MinMaxScaler()),
    ("StandardScaler", StandardScaler()),
    ("RobusScaler", RobustScaler()),
    ("Normalizer", Normalizer()),
]
"""``plot_scaled_data`` で比較するスケーラー一覧。"""


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
                行わない。デフォルトは ``None``。

        Raises:
            TypeError: ``value`` が整数以外の型の場合。
            ValueError: ``value`` が1未満、または ``upper_bound`` を超える場合。
        """
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"{variable_name} は1以上のint型です。"
                f" 受け取った値: {value!r} (type={type(value).__name__})"
            )
        if upper_bound is not None and value > upper_bound:
            raise ValueError(
                f"{variable_name} は {upper_bound} 以下にしてください。"
                f" 受け取った値: {value}"
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
                "diag_kind は 'auto', 'hist', 'kde', None のいずれかです。"
                f" 受け取った値: {diag_kind!r}"
            )

    def _validate_n_neighbors(self, n_neighbors: int) -> None:
        """KNeighborsClassifierの近傍数を確認する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。

        Raises:
            ValueError: ``n_neighbors`` が使用可能な範囲外の場合。
        """
        self._validate_positive_int(n_neighbors, "n_neighbors")

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

        Raises:
            TypeError: ``head`` が整数以外の型の場合。
            ValueError: ``head`` が1未満の場合。
        """
        if head is not None:
            self._validate_positive_int(head, "head")

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
        n_neighbors: int = 4,
    ) -> dict[str, dict[str, np.ndarray]]:
        """全分類モデルに対して交差検証スコアを計算する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            dict[str, dict[str, np.ndarray]]: モデル名をキー、cross_validateの
                結果を値とするdict。

        Raises:
            ValueError: ``n_neighbors`` が整数でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        self._validate_n_neighbors(n_neighbors)

        list_classifier: list[tuple[str, Any]] = []
        for str_classifier_name, classifier in DEFAULT_CLASSIFIERS:
            if str_classifier_name == "KNeighborsClassifier":
                list_classifier.append(
                    (
                        str_classifier_name,
                        KNeighborsClassifier(n_neighbors=n_neighbors),
                    )
                )
                continue
            list_classifier.append((str_classifier_name, classifier))

        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target
        dict_results: dict[str, dict[str, np.ndarray]] = {}

        for str_classifier_name, classifier in list_classifier:
            dict_results[str_classifier_name] = cross_validate(
                classifier,
                df_feature,
                ndarray_target,
                cv=CV_SPLITS,
                return_train_score=True,
            )

        return dict_results

    def all_supervised(self, n_neighbors: int = 4) -> None:
        """全分類モデルの交差検証スコアをコンソールに出力する。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Raises:
            TypeError: ``n_neighbors`` が整数以外の型の場合。
            ValueError: ``n_neighbors`` が整数でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        dict_results = self.calc_supervised_scores(n_neighbors)

        for str_classifier_name, dict_score in dict_results.items():
            print("=== {} ===".format(str_classifier_name))
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

    def get_supervised(self, n_neighbors: int = 4) -> pd.DataFrame:
        """全分類モデルのテストスコアをDataFrameで返す。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            pd.DataFrame: モデル名を列名、各foldのテストスコアを行とする
                DataFrame。

        Raises:
            TypeError: ``n_neighbors`` が整数以外の型の場合。
            ValueError: ``n_neighbors`` が整数でないか、1未満か、
                交差検証で扱える件数を超える場合。
        """
        dict_results = self.calc_supervised_scores(n_neighbors)
        dict_test_score: dict[str, np.ndarray] = {}

        for str_classifier_name, dict_score in dict_results.items():
            dict_test_score[str_classifier_name] = dict_score["test_score"]

        return pd.DataFrame(dict_test_score)

    def best_supervised(self, n_neighbors: int = 4) -> tuple[str, float]:
        """平均テストスコアが最も高い分類モデルを返す。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。デフォルトは4。

        Returns:
            tuple[str, float]: 以下の要素を持つタプル。

                - str_best_method (str): 最良モデルの名前。
                - float_best_score (float): 最良モデルの平均テストスコア。

        Raises:
            TypeError: ``n_neighbors`` が整数以外の型の場合。
            ValueError: ``n_neighbors`` が整数でないか、1未満か、
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

        for str_classifier_name, classifier in DEFAULT_TREE_CLASSIFIERS:
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
        StandardScaler / RobusScaler / Normalizer の各スケーリングを適用し、
        LinearSVCのtest/trainスコアと散布図行列を表示する。

        Returns:
            pd.DataFrame: 各fold・各スケーリング手法のtest/trainスコアを
                格納したDataFrame。

        Notes:
            散布図は各foldの訓練データを用いて、特徴量の最初の2次元の
            分布をスケーリング手法ごとに描画する。
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

            for int_idx, (str_scaler_name, scaler) in enumerate(DEFAULT_SCALERS):
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
                ax.scatter(
                    ndarray_x_train_scaled[:, 0],
                    ndarray_x_train_scaled[:, 1],
                    c=ndarray_y_train,
                    cmap="viridis",
                    edgecolor="k",
                    s=40,
                )
                ax.set_title(
                    "fold={} / {}".format(int_fold, str_scaler_name)
                )
                ax.set_xlabel(df_feature.columns[0])
                ax.set_ylabel(df_feature.columns[1])

            list_axes_flat[5].axis("off")
            fig.tight_layout()
            plt.show()

        print(
            "========================================================================="
        )
        return pd.DataFrame(list_records)

    def plot_pca(self, n_components: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
        """StandardScaler後にPCAを適用し、結果を可視化する。

        Args:
            n_components (int): 主成分の数。デフォルトは2。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame, PCA]: 以下の要素を持つタプル。

                - df_x_scaled (pd.DataFrame): 標準化後の特徴量DataFrame。
                - df_pca (pd.DataFrame): 主成分得点のDataFrame。
                - pca (PCA): 学習済みのPCAインスタンス。

        Raises:
            TypeError: ``n_components`` が整数以外の型の場合。
            ValueError: ``n_components`` が1未満、または特徴量数を超える場合。
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

        self._plot_2d_with_labels(
            ndarray_pca,
            self.dataset.target,
            title="PCA",
            xlabel=list_pc_columns[0],
            ylabel=list_pc_columns[1] if n_components >= 2 else "",
        )

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

        Raises:
            TypeError: ``n_components`` が整数以外の型の場合。
            ValueError: ``n_components`` が1未満、または特徴量数を超える場合。
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

        self._plot_2d_with_labels(
            ndarray_nmf,
            self.dataset.target,
            title="NMF",
            xlabel=list_nmf_columns[0],
            ylabel=list_nmf_columns[1] if n_components >= 2 else "",
        )

        return (df_x_scaled, df_nmf, nmf)

    def plot_tsne(self) -> None:
        """スケーリングなしのデータにt-SNEを適用し、2次元で可視化する。

        Notes:
            t-SNEは確率的手法のため、結果は ``RANDOM_STATE`` に依存する。
        """
        ndarray_target: np.ndarray = self.dataset.target

        tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
        ndarray_tsne = tsne.fit_transform(self.df_feature)

        self._plot_2d_with_labels(
            ndarray_tsne,
            ndarray_target,
            title="t-SNE",
            xlabel="t-SNE feature 0",
            ylabel="t-SNE feature 1",
        )

    def plot_k_means(self) -> None:
        """KMeans法でクラスタリングし、結果を表示する。

        クラスタ数はクラス数(=3)に設定する。KMeansが付与したラベルと
        実際のラベルの両方を出力し、PCAで2次元に投影した散布図で
        クラスタリング結果を可視化する。
        """
        df_feature = self.df_feature
        ndarray_target: np.ndarray = self.dataset.target
        int_n_clusters = len(self.dataset.target_names)

        kmeans = KMeans(
            n_clusters=int_n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        kmeans.fit(df_feature)

        print("KMeans法で予測したラベル:")
        print(kmeans.labels_)
        print()
        print("実際のラベル:")
        print(ndarray_target)

        scaler = StandardScaler()
        ndarray_x_scaled = scaler.fit_transform(df_feature)
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        ndarray_pca = pca.fit_transform(ndarray_x_scaled)

        fig, list_axes = plt.subplots(1, 2, figsize=(12, 5))
        list_axes[0].scatter(
            ndarray_pca[:, 0],
            ndarray_pca[:, 1],
            c=kmeans.labels_,
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
            c=ndarray_target,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        list_axes[1].set_title("True labels (PCA projection)")
        list_axes[1].set_xlabel("PC1")
        list_axes[1].set_ylabel("PC2")

        fig.tight_layout()
        plt.show()

    def plot_dendrogram(self, truncate: bool = False) -> None:
        """凝集型階層クラスタリングのデンドログラムを表示する。

        Args:
            truncate (bool): ``True`` の場合は ``truncate_mode='lastp'`` で
                上位の枝のみを表示する。デフォルトは ``False``。

        Raises:
            TypeError: ``truncate`` がbool以外の場合。
        """
        if not isinstance(truncate, bool):
            raise TypeError(
                "truncate は bool 型です。"
                f" 受け取った値: {truncate!r} (type={type(truncate).__name__})"
            )

        ndarray_linkage = linkage(self.df_feature.to_numpy(), method="ward")

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

    def plot_dbscan(
        self,
        scaling: bool = False,
        eps: float = 0.5,
        min_samples: int = 5,
    ) -> None:
        """DBSCANでクラスタリングし、結果を表示する。

        Args:
            scaling (bool): ``True`` のときStandardScalerでスケーリングしてから
                DBSCANを適用する。デフォルトは ``False``。
            eps (float): DBSCANのepsパラメータ。デフォルトは0.5。
            min_samples (int): DBSCANのmin_samplesパラメータ。デフォルトは5。

        Raises:
            TypeError: ``scaling`` がbool以外、``eps`` が数値以外、
                ``min_samples`` が整数以外の場合。
            ValueError: ``eps`` が0以下、``min_samples`` が1未満の場合。
        """
        if not isinstance(scaling, bool):
            raise TypeError(
                "scaling は bool 型です。"
                f" 受け取った値: {scaling!r} (type={type(scaling).__name__})"
            )
        if isinstance(eps, bool) or not isinstance(eps, (int, float)):
            raise TypeError(
                "eps は数値型(int または float)です。"
                f" 受け取った値: {eps!r} (type={type(eps).__name__})"
            )
        if eps <= 0:
            raise ValueError(
                f"eps は正の数値です。受け取った値: {eps}"
            )
        self._validate_positive_int(min_samples, "min_samples")

        df_feature = self.df_feature
        if scaling:
            scaler = StandardScaler()
            ndarray_x = scaler.fit_transform(df_feature)
        else:
            ndarray_x = df_feature.to_numpy()

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        ndarray_clusters = dbscan.fit_predict(ndarray_x)
        print("Cluster Memberships:", ndarray_clusters)

        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        ndarray_pca = pca.fit_transform(ndarray_x)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            ndarray_pca[:, 0],
            ndarray_pca[:, 1],
            c=ndarray_clusters,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        plt.title(
            "DBSCAN (scaling={}, eps={}, min_samples={})".format(
                scaling, eps, min_samples
            )
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.show()

    def _plot_2d_with_labels(
        self,
        ndarray_xy: np.ndarray,
        ndarray_labels: np.ndarray,
        title: str,
        xlabel: str,
        ylabel: str,
    ) -> None:
        """2次元データをラベル別に色分けして散布図表示する内部ヘルパー。

        Args:
            ndarray_xy (np.ndarray): shape=(n_samples, 2) の座標配列。
            ndarray_labels (np.ndarray): クラスラベル配列。
            title (str): グラフタイトル。
            xlabel (str): X軸ラベル。
            ylabel (str): Y軸ラベル。
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(
            ndarray_xy[:, 0],
            ndarray_xy[:, 1] if ndarray_xy.shape[1] >= 2 else np.zeros(len(ndarray_xy)),
            c=ndarray_labels,
            cmap="viridis",
            edgecolor="k",
            s=40,
        )
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()
