from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def build_model_registry() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=8),
        "random_forest": RandomForestClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=10,
        ),
        "naive_bayes": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=5),
    }
