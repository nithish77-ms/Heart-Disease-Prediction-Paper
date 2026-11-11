import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MultiCriterionRandomForest:
    def __init__(
        self,
        n_estimators=200,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.criteria = ["gini", "entropy", "log_loss"]
        self.trees = []
        self.tree_criteria = []

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        self.trees = []
        self.tree_criteria = []

        for _ in range(self.n_estimators):
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            criterion = rng.choice(self.criteria)

            tree = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                class_weight=self.class_weight,
                random_state=rng.randint(0, 1000000),
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            self.tree_criteria.append(criterion)

        print("Multi-Criterion Random Forest Trained Successfully.")
        self.print_criterion_distribution()
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = np.array([tree.predict(X) for tree in self.trees])
        final_predictions = []

        for sample in predictions.T:
            values, counts = np.unique(sample, return_counts=True)
            final_predictions.append(values[np.argmax(counts)])

        return np.array(final_predictions)

    def predict_proba(self, X):
        X = np.asarray(X)
        probabilities = np.array([tree.predict_proba(X) for tree in self.trees])
        return probabilities.mean(axis=0)

    def print_criterion_distribution(self):
        print("\nSplitting Criterion Distribution")
        for criterion in self.criteria:
            count = self.tree_criteria.count(criterion)
            percentage = (count / self.n_estimators) * 100
            print(f"{criterion:10s}: {count:3d} trees ({percentage:.2f}%)")
