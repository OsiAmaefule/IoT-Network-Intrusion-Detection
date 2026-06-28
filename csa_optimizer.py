"""
Crow Search Algorithm (CSA) hyperparameter optimization wrapper.
"""

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from mealpy import FloatVar, IntegerVar, Problem
from mealpy.swarm_based.CSA import OriginalCSA


class ClassifierOptProblem(Problem):
    """Optimization problem that maximizes classifier CV accuracy."""

    def __init__(self, classifier_class, param_grid, X_train, y_train, cv=3, **kwargs):
        self.classifier_class = classifier_class
        self.param_grid = param_grid
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.param_names = list(param_grid.keys())
        self.le = LabelEncoder()
        self.y_encoded = self.le.fit_transform(y_train)

        # Categorical params handled separately — only pass numeric bounds to mealpy
        self.numeric_params = []
        self.categorical_params = {}
        bounds = []

        for name, config in param_grid.items():
            if config["type"] == "int":
                self.numeric_params.append((name, "int"))
                bounds.append(IntegerVar(config["low"], config["high"], name=name))
            elif config["type"] == "float":
                self.numeric_params.append((name, "float"))
                bounds.append(FloatVar(config["low"], config["high"], name=name))
            elif config["type"] == "categorical":
                # Use int index into choices list
                self.numeric_params.append((name, "categorical", config["choices"]))
                bounds.append(IntegerVar(0, len(config["choices"]) - 1, name=name))

        super().__init__(bounds=bounds, minmax="max", **kwargs)

    def obj_func(self, x):
        params = {}
        for i, (item) in enumerate(self.numeric_params):
            name = item[0]
            ptype = item[1]
            val = x[i]
            if ptype == "int":
                params[name] = int(round(val))
            elif ptype == "float":
                params[name] = float(val)
            elif ptype == "categorical":
                choices = item[2]
                params[name] = choices[int(round(val)) % len(choices)]

        try:
            clf = self.classifier_class(**params)
            scores = cross_val_score(
                clf, self.X_train, self.y_encoded,
                cv=self.cv, scoring="accuracy", n_jobs=-1
            )
            return float(np.mean(scores))
        except Exception as e:
            return 0.0


def optimize_classifier(classifier_class, param_grid, X_train, y_train,
                        pop_size=30, epoch=50, cv=3):
    """
    Optimize classifier hyperparameters using CSA.
    Returns: best_params (dict), best_accuracy (float)
    """
    problem = ClassifierOptProblem(
        classifier_class, param_grid, X_train, y_train, cv=cv
    )
    model = OriginalCSA(epoch=epoch, pop_size=pop_size)
    g_best = model.solve(problem)

    best_params = {}
    for i, item in enumerate(problem.numeric_params):
        name = item[0]
        ptype = item[1]
        val = g_best.solution[i]
        if ptype == "int":
            best_params[name] = int(round(val))
        elif ptype == "float":
            best_params[name] = float(val)
        elif ptype == "categorical":
            choices = item[2]
            best_params[name] = choices[int(round(val)) % len(choices)]

    return best_params, g_best.target.fitness
