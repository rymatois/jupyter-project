# -*- coding: utf-8 -*-
"""分類データセットの探索的分析と分類モデル評価モジュール。

このモジュールはsklearn形式のデータセットを用いて、特徴量の可視化・
相関分析・複数の教師あり学習モデルの交差検証・特徴量重要度の可視化を
行うクラスを提供する。

Example:
    基本的な使い方::

        analyzer = ClassifierAnalyzer()                    # Irisデータセット（デフォルト）
        analyzer = ClassifierAnalyzer(load_wine())         # 他のデータセットも可
        analyzer.plot_pairwise_scatter()
        analyzer.print_cv_scores()
        best_model, best_score = analyzer.get_best_model()

    モデルを追加したい場合（DEFAULT_CLASSIFIERSに連結する）::

        from sklearn.linear_model import RidgeClassifier
        from sklearn.datasets import load_wine
        wine = ClassifierAnalyzer(
            dataset=load_wine(),
            classifiers=DEFAULT_CLASSIFIERS + [("RidgeClassifier", RidgeClassifier())],
        )
        wine.print_cv_scores()
"""

import inspect
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

# 特徴量重要度の表示対象モデル（木ベースのみ）
DEFAULT_TREE_CLASSIFIERS = [
    ("DecisionTreeClassifier", DecisionTreeClassifier(random_state=0)),
    ("RandomForestClassifier", RandomForestClassifier(random_state=0)),
    ("GradientBoostingClassifier", GradientBoostingClassifier(random_state=0)),
]

# ここを編集してモデルを追加・削除・差し替えする
DEFAULT_CLASSIFIERS = [
    ("LogisticRegression", LogisticRegression(max_iter=1000)),
    ("LinearSVC", LinearSVC(max_iter=10000)),
    ("SVC", SVC()),
    ("DecisionTreeClassifier", DecisionTreeClassifier()),
    ("KNeighborsClassifier", KNeighborsClassifier(n_neighbors=4)),
    ("RandomForestClassifier", RandomForestClassifier()),
    ("GradientBoostingClassifier", GradientBoostingClassifier()),
    ("MLPClassifier", MLPClassifier(max_iter=2000)),
]


class ClassifierAnalyzer:
    """分類データセットの分析・可視化・モデル評価を行うクラス。

    Args:
        dataset: sklearn形式のデータセット（Bunch）。
            ``None`` の場合はIrisデータセットを使用する。
        classifiers: (モデル名, モデルインスタンス) のリスト。
            ``None`` の場合は ``DEFAULT_CLASSIFIERS`` を使用する。

    Attributes:
        dataset: 読み込んだデータセット。
        feature_df (pd.DataFrame): 特徴量のみのDataFrame（ラベルなし）。
        classifiers (list): 評価対象の分類モデル一覧。
    """

    def __init__(self, dataset=None, classifiers=None):
        """データセットを読み込み、特徴量DataFrameを初期化する。"""
        if dataset is None:
            dataset = load_iris()
        self.dataset = dataset
        self.feature_df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
        self.classifiers = (
            classifiers if classifiers is not None else DEFAULT_CLASSIFIERS
        )

    def print_methods(self):
        """使用可能なメソッドの一覧と説明を表示する。

        ``_`` で始まる内部メソッドを除いた全パブリックメソッドについて、
        名前・シグネチャ・docstringの先頭行を出力する。
        """
        print("=== {} の使用可能なメソッド ===\n".format(self.__class__.__name__))
        for name in dir(self):
            if name.startswith("_"):
                continue
            try:
                method = getattr(self, name)
            except AttributeError:
                continue
            if not callable(method):
                continue
            sig = inspect.signature(method)
            doc = inspect.getdoc(method)
            summary = doc.splitlines()[0] if doc else "(説明なし)"
            print("  {}{}".format(name, sig))
            print("    → {}\n".format(summary))

    def print_dataset_info(self):
        """データセットの概要をコンソールに出力する。

        サンプル数・特徴量数・クラス数・クラス分布・特徴量の基本統計量を表示する。
        """
        target = self.dataset.target
        target_names = self.dataset.target_names
        n_samples, n_features = self.feature_df.shape
        n_classes = len(target_names)

        print("=== データセット概要 ===\n")
        print("サンプル数   : {}".format(n_samples))
        print("特徴量数     : {}".format(n_features))
        print("クラス数     : {}\n".format(n_classes))

        print("クラス分布:")
        for i, name in enumerate(target_names):
            count = (target == i).sum()
            pct = count / n_samples * 100
            print("  {:12s} : {:3d} サンプル ({:.1f}%)".format(name, count, pct))

        print("\n特徴量の統計:")
        print(self.feature_df.describe().round(3))

    def get_labeled_df(self, head=None):
        """ラベル列を付加したDataFrameを返す。

        Args:
            head (int | None): 先頭から返す行数。``None`` の場合は全行を返す。

        Returns:
            pd.DataFrame: ラベル列（Label）を含むDataFrame。
        """
        labeled_df = self.feature_df.copy()
        labeled_df["Label"] = self.dataset.target
        if head is not None:
            return labeled_df.head(head)
        return labeled_df

    def get_correlation_matrix(self):
        """特徴量間の相関係数行列を返す。

        Returns:
            pd.DataFrame: 特徴量間の相関係数を格納したDataFrame。
        """
        return self.feature_df.corr()

    def plot_pairwise_scatter(self, diag_kind="hist"):
        """全特徴量の組み合わせを散布図で表示する。

        Args:
            diag_kind (str): 対角成分のグラフ種別。``"hist"`` または
                ``"kde"`` を指定する。デフォルトは ``"hist"``。
        """
        labeled_df = self.get_labeled_df()
        label_map = {i: name for i, name in enumerate(self.dataset.target_names)}
        labeled_df["LabelName"] = labeled_df["Label"].map(label_map)
        sns.pairplot(
            labeled_df.drop(columns=["Label"]),
            hue="LabelName",
            diag_kind=diag_kind,
        )
        plt.show()

    def _build_classifier_list(self, n_neighbors):
        """評価対象の分類モデル一覧を生成する。

        ``self.classifiers`` をベースに、KNeighborsClassifier の
        ``n_neighbors`` だけを引数の値で上書きして返す。

        Args:
            n_neighbors (int): KNeighborsClassifierの近傍数。

        Returns:
            list: (モデル名 (str), モデルインスタンス) のタプルのリスト。
        """
        return [
            (
                (name, KNeighborsClassifier(n_neighbors=n_neighbors))
                if name == "KNeighborsClassifier"
                else (name, clf)
            )
            for name, clf in self.classifiers
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
        y = self.dataset.target

        cv_results = {}
        for model_name, classifier in classifiers:
            result = cross_validate(classifier, X, y, cv=5, return_train_score=True)
            cv_results[model_name] = result

        return cv_results

    def print_cv_scores(self, n_neighbors=4):
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

    def get_cv_score_df(self):
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

    def get_best_model(self):
        """平均テストスコアが最も高い分類モデルを返す。

        Returns:
            tuple: 以下の要素を持つタプル。

                - best_model_name (str): 最良モデルの名前。
                - best_mean_score (float): 最良モデルの平均テストスコア。
        """
        score_stats_df = self.get_cv_score_df().describe()
        best_model_name = score_stats_df.loc["mean"].idxmax()
        best_mean_score = score_stats_df.loc["mean"].max()

        return (best_model_name, best_mean_score)

    def plot_model_score_comparison(self):
        """全分類モデルの平均テストスコアを棒グラフで比較表示する。

        各モデルの平均スコアを棒グラフで、標準偏差をエラーバーで表示する。
        """
        score_df = self.get_cv_score_df()
        means = score_df.mean()
        stds = score_df.std()

        plt.figure(figsize=(10, 5))
        plt.bar(means.index, means.values, yerr=stds.values, capsize=4)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean Test Score")
        plt.title("Model Score Comparison (mean ± std, cv=5)")
        plt.ylim(max(0, means.min() - 0.1), 1.05)
        plt.tight_layout()
        plt.show()

    def plot_feature_importances(self, tree_classifiers=None):
        """木ベースの分類モデルの特徴量重要度を棒グラフで表示する。

        Args:
            tree_classifiers: (モデル名, モデルインスタンス) のリスト。
                ``None`` の場合は ``DEFAULT_TREE_CLASSIFIERS`` を使用する。
        """
        X = self.feature_df
        y = self.dataset.target

        if tree_classifiers is None:
            tree_classifiers = DEFAULT_TREE_CLASSIFIERS

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

    def plot_decision_tree(self):
        """決定木の構造を可視化して表示する。

        DecisionTreeClassifierを全データで学習し、木構造を図示する。

        Returns:
            list: plot_treeが返すArtistオブジェクトのリスト。
        """
        X = self.feature_df
        y = self.dataset.target

        classifier = DecisionTreeClassifier(random_state=0)
        classifier.fit(X, y)

        plt.figure(figsize=(24, 15))
        tree_artists = plot_tree(
            classifier,
            feature_names=X.columns,
            class_names=self.dataset.target_names,
            filled=True,
            rounded=True,
            fontsize=10,
        )
        plt.show()

        return tree_artists


# ------------------------------------------------------------
# リネーム履歴
# ------------------------------------------------------------
# メソッド名
#   describe()                     -> print_methods()
#   summary()                      -> print_dataset_info()
#   get()                          -> get_labeled_df()
#   get_correlation()              -> get_correlation_matrix()
#   pair_plot()                    -> plot_pairwise_scatter()
#   all_supervised()               -> print_cv_scores()
#   get_supervised()               -> get_cv_score_df()
#   best_supervised()              -> get_best_model()
#   plot_score_comparison()        -> plot_model_score_comparison()
#   plot_feature_importances_all() -> plot_feature_importances()
#   visualize_decision_tree()      -> plot_decision_tree()
# ------------------------------------------------------------
