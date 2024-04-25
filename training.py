from sklearn.model_selection import GridSearchCV


def train_model(model, X_train, X_test, y_train, y_test) -> any:

    #param_grid = {
    #    'n_estimators': [100, 200, 300],
    #    'max_depth': [None, 10, 20],
    #    'min_samples_split': [2, 5, 10],
    #    'min_samples_leaf': [1, 2, 4]
    #}

    param_grid = {
        'n_neighbors': [10, 50, 100, 200, 300, 500],
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    }

    # Perform grid search with cross-validation to find the best combination of hyperparameters
    #grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1,verbose=True)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy',verbose=True)
    grid_search.fit(X_train, y_train)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Train the best model
    best_model.fit(X_train, y_train)
    return best_model