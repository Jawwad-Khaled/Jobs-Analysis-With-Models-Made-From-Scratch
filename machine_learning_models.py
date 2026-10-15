import numpy as np

class Ridge_Regression():
    def __init__(self):
        self.parameters = np.array([])
        self.y_mean = 0
        self.X_mean = 0

    def fit(self, X, y, alpha):
        self.y_mean = np.mean(y)
        self.X_mean = np.mean(X, axis=0)
        X_ = (X - self.X_mean).copy()
        y_ = (y - self.y_mean).reshape(-1, 1).copy()
        self.parameters = np.linalg.inv(X_.T @ X_ + alpha * np.identity((X_.T @ X_).shape[0])) @ X_.T @ y_

    def predict(self, X):
        y_hat = X @ self.parameters + self.y_mean - self.parameters.T @ self.X_mean.reshape(-1, 1).copy()
        y_hat = y_hat.reshape(-1)
        return y_hat

class Logistic_Regression():
    def __init__(self, max_iterations=1000, record_losses=False):
        self.parameters = np.array([])
        self.first_moment_r = np.array([])
        self.second_moment_r = np.array([])
        self.first_moment_ur = 0
        self.second_moment_ur = 0
        self.bias = 0
        self.bias_vector = np.array([])
        self.learning_rate = 0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.max_iterations = max_iterations
        self.record_losses = record_losses
        self.losses = []
        self.iterations = []

    def fit(self, X, y, alpha):
        X_ = X.copy()
        y_ = y.reshape(-1, 1).copy()
        self.parameters = np.zeros(X_.shape[1]).reshape(-1, 1)
        self.first_moment_r = np.zeros(X_.shape[1]).reshape(-1, 1)
        self.second_moment_r = np.zeros(X_.shape[1]).reshape(-1, 1)
        self.first_moment_ur = 0
        self.second_moment_ur = 0
        self.bias_vector =  np.full(shape=y_.shape[0], fill_value=self.bias).reshape(-1, 1)
        for i in range(0, self.max_iterations):
            previous_parameters = self.parameters.copy()
            p = 1/(1+np.exp(-(X_ @ self.parameters + self.bias_vector)))
            grad_w = -(np.transpose(X_) @ (y_ - p) -2 * X_.shape[0] * alpha * self.parameters) * (1/X_.shape[0])
            grad_b = -(np.mean(y_ - p))
            self.first_moment_r = self.beta_1 * self.first_moment_r + (1-self.beta_1) * grad_w 
            self.second_moment_r = self.beta_2 * self.second_moment_r + (1-self.beta_2) * np.square(grad_w)
            self.first_moment_ur = self.beta_1 * self.first_moment_ur + (1-self.beta_1) * grad_b
            self.second_moment_ur = self.beta_2 * self.second_moment_ur + (1-self.beta_2) * np.square(grad_b)
            m_hat_w = self.first_moment_r / (1 - self.beta_1 ** (i+1))
            v_hat_w = self.second_moment_r / (1 - self.beta_2 ** (i+1))
            m_hat_b = self.first_moment_ur / (1 - self.beta_1 ** (i+1))
            v_hat_b = self.second_moment_ur / (1 - self.beta_2 ** (i+1))
            self.parameters = self.parameters - self.learning_rate * (m_hat_w / (np.sqrt(v_hat_w) + 1e-8))
            self.bias = self.bias - self.learning_rate * (m_hat_b / (np.sqrt(v_hat_b) + 1e-8))
            self.bias_vector = np.full(shape=y_.shape[0], fill_value=self.bias).reshape(-1, 1)
            if self.record_losses:
                loss = (-1/y_.shape[0]) * np.sum(y_ * np.log(p) + (np.ones_like(y_) - y_) * np.log(np.ones_like(p) - p))
                self.losses.append(loss)
                self.iterations.append(i+1)
            if np.max(np.abs(self.parameters - previous_parameters)) < 1e-12:
                break

    def predict(self, X):
        bias_vector_prediction = np.full(shape=X.shape[0], fill_value=self.bias).reshape(-1, 1)
        y_hat = 1/(1+np.exp(-(X @ self.parameters + bias_vector_prediction)))
        y_hat = y_hat.reshape(-1)
        return y_hat    

class Node():
    def __init__(self, feature=None, threshold=0, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class Regression_Tree():
    def __init__(self, number_of_random_features, minimum_samples_split=100, max_depth=100, hessian=None):
        self.minimum_samples_split = minimum_samples_split
        self.max_depth = max_depth
        self.features = np.array([])
        self.number_of_random_features = number_of_random_features
        self.hessian = hessian
        self.sse_decreases = np.array([])
        self.sum_of_changes = np.array([])
        self.root = None
        self.total_count = 0

    def choose_best_split(self, X, y):
        self.features = np.random.choice(X.shape[1], self.number_of_random_features, replace=False)
        choices = []
        if len(y) == 0:
            return None, None, False
        initial_sse = np.sum((np.mean(y) - y)**2)
        for feature in self.features:
            sorted_array = np.unique(X[:, feature])
            if len(sorted_array) <= 1:
                continue
            midpoints = 0.5 * (sorted_array[1:] + sorted_array[:-1])
            sse_values = {}
            for midpoint in midpoints:
                sse = 0
                left_mask = np.argwhere(X[:, feature] <= midpoint).flatten()
                right_mask = np.argwhere(X[:, feature] > midpoint).flatten()
                if len(left_mask) == 0 or len(right_mask) == 0:
                    continue
                left_mean = np.mean(y[left_mask])
                sse += np.sum((left_mean - y[left_mask])**2)
                right_mean = np.mean(y[right_mask])
                sse += np.sum((right_mean - y[right_mask])**2)
                sse_values.update({midpoint:sse})
            if not bool(sse_values):
                continue
            best_midpoint = min(sse_values, key=sse_values.get)
            best_sse = sse_values[best_midpoint]
            choices.append([feature, best_midpoint, best_sse])
        if not choices:
            return None, None, False
        best_predictor = 0
        best_midpoint = 0
        best_sse = np.inf
        for choice in choices:
            if choice[2] < best_sse:
                best_sse = choice[2]
                best_predictor = choice[0]
                best_midpoint = choice[1]
        left_mask = X[:, best_predictor] <= best_midpoint
        right_mask = ~left_mask
        self.sse_decreases[best_predictor] += initial_sse - best_sse
        self.sum_of_changes[best_predictor] += len(y) / self.total_count * (np.mean(y[right_mask]) - np.mean(y[left_mask]))
        return best_predictor, best_midpoint, True
    
    def fit(self, X, y):
        self.sse_decreases = np.zeros((X.shape[1]))
        self.sum_of_changes = np.zeros((X.shape[1]))
        self.total_count = len(y)
        if self.hessian is None:
            self.hessian = np.ones(y.shape)
        self.features = X.shape[1]
        self.root = self.grow_tree(X, y, self.hessian)

    def grow_tree(self, X, y, hessian, depth=0):
        if len(y) == 0:
            prediction = 0
            return Node(value=prediction)
        if depth >= self.max_depth or len(y) < self.minimum_samples_split:
            prediction = np.sum(y) / np.sum(hessian)
            return Node(value=prediction)
        best_predictor, best_midpoint, valid_split = self.choose_best_split(X, y)
        if not valid_split:
            prediction = np.sum(y) / np.sum(hessian)
            return Node(value=prediction)
        left_mask = X[:, best_predictor] <= best_midpoint
        right_mask = ~left_mask
        left = self.grow_tree(X[left_mask], y[left_mask], hessian[left_mask], depth+1)
        right = self.grow_tree(X[right_mask], y[right_mask], hessian[right_mask], depth+1)
        return Node(best_predictor, best_midpoint, left, right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def feature_importances(self):
        return self.sse_decreases / np.sum(self.sse_decreases)

    def direction_of_change(self):
        return self.sum_of_changes

    def permutation_importances(self, X, y):
        predictions_full = self.predict(X)
        rmse_full = np.sqrt(np.mean((y - predictions_full)**2))
        permutation_importances = []
        for feature in range(0, X.shape[1]):
            feature_rmse_increase = []
            for i in range(0, 10):
                X_ = X.copy()
                np.random.shuffle(X_[:, feature])
                predictions = self.predict(X_)
                rmse = np.sqrt(np.mean((y - predictions)**2))
                rmse_increase = rmse - rmse_full
                feature_rmse_increase.append(rmse_increase)
            permutation_importances.append(np.mean(feature_rmse_increase))
        permutation_importances = np.array(permutation_importances)
        permutation_importances = np.where(permutation_importances > 0, permutation_importances, 0)
        permutation_importances = permutation_importances / np.sum(permutation_importances)
        return permutation_importances

class Classification_Tree():
    def __init__(self, number_of_random_features, minimum_samples_split=100, max_depth=100):
        self.minimum_samples_split = minimum_samples_split
        self.max_depth = max_depth
        self.features = np.array([])
        self.number_of_random_features = number_of_random_features
        self.impurity_decreases = np.array([])
        self.sum_of_changes = np.array([])
        self.root = None
        self.total_count = 0

    def choose_best_split(self, X, y):
        self.features = np.random.choice(X.shape[1], self.number_of_random_features, replace=False)
        choices = []
        if len(y) == 0:
            return None, None, False
        initial_impurity = 1 - (np.sum(y == 0) / len(y)) ** 2 - (np.sum(y == 1) / len(y)) ** 2
        for feature in self.features:
            sorted_array = np.unique(X[:, feature])
            if len(sorted_array) <= 1:
                continue
            midpoints = 0.5 * (sorted_array[1:] + sorted_array[:-1])
            gini_impurities = {}
            for midpoint in midpoints:
                impurity = 0
                left_mask = X[:, feature] <= midpoint
                right_mask = ~left_mask
                if len(left_mask) == 0 or len(right_mask) == 0:
                    continue
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_impurity = 1 - (np.sum(y[left_mask] == 0) / len(y[left_mask])) ** 2 - (np.sum(y[left_mask] == 1) / len(y[left_mask])) ** 2
                right_impurity = 1 - (np.sum(y[right_mask] == 0) / len(y[right_mask])) ** 2 - (np.sum(y[right_mask] == 1) / len(y[right_mask])) ** 2
                impurity = (len(y[left_mask]) / len(y)) * left_impurity + (len(y[right_mask]) / len(y)) * right_impurity
                gini_impurities.update({midpoint:impurity})
            if not bool(gini_impurities):
                continue
            best_midpoint = min(gini_impurities, key=gini_impurities.get)
            best_impurity = gini_impurities[best_midpoint]
            choices.append([feature, best_midpoint, best_impurity])
        if not choices:
            return None, None, False
        best_predictor = 0
        best_midpoint = 0
        best_impurity = np.inf
        for choice in choices:
            if choice[2] < best_impurity:
                best_impurity = choice[2]
                best_predictor = choice[0]
                best_midpoint = choice[1]
        left_mask = X[:, best_predictor] <= best_midpoint
        right_mask = ~left_mask
        self.impurity_decreases[best_predictor] += initial_impurity - best_impurity
        self.sum_of_changes[best_predictor] += len(y) / self.total_count * (np.mean(y[right_mask]) - np.mean(y[left_mask]))
        return best_predictor, best_midpoint, True
    
    def fit(self, X, y):
        self.impurity_decreases = np.zeros((X.shape[1]))
        self.sum_of_changes = np.zeros((X.shape[1]))
        self.total_count = len(y)
        self.features = X.shape[1]
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        if len(y) == 0:
            prediction = 0
            return Node(value=prediction)
        if depth >= self.max_depth or len(y) < self.minimum_samples_split or len(np.unique(y)) == 1:
            prediction = np.argmax(np.bincount(y))
            return Node(value=prediction)
        best_predictor, best_midpoint, valid_split = self.choose_best_split(X, y)
        if not valid_split:
            prediction = np.argmax(np.bincount(y))
            return Node(value=prediction)
        left_mask = X[:, best_predictor] <= best_midpoint
        right_mask = ~left_mask
        left = self.grow_tree(X[left_mask], y[left_mask], depth+1)
        right = self.grow_tree(X[right_mask], y[right_mask], depth+1)
        return Node(best_predictor, best_midpoint, left, right)

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def feature_importances(self):
        return self.impurity_decreases / np.sum(self.impurity_decreases)

    def direction_of_change(self):
        return self.sum_of_changes

    def permutation_importances(self, X, y):
        predictions_full = self.predict(X)
        predictions_full = [1 if x > 0.5 else 0 for x in predictions_full]
        accuracy_full = np.sum(y == predictions_full) / len(y)
        permutation_importances = []
        for feature in range(0, X.shape[1]):
            feature_accuracy_increase = []
            for i in range(0, 10):
                X_ = X.copy()
                np.random.shuffle(X_[:, feature])
                predictions = self.predict(X_)
                predictions = [1 if x > 0.5 else 0 for x in predictions]
                accuracy = np.sum(y == predictions) / len(y)
                accuracy_decrease = accuracy_full - accuracy
                feature_accuracy_increase.append(accuracy_decrease)
            permutation_importances.append(np.mean(feature_accuracy_increase))
        permutation_importances = np.array(permutation_importances)
        permutation_importances = np.where(permutation_importances > 0, permutation_importances, 0)
        permutation_importances = permutation_importances / np.sum(permutation_importances)
        return permutation_importances

class Regression_Random_Forest():
    def __init__(self, number_of_trees=100, number_of_random_features=1, max_depth=100, minimum_samples_split=20):
        self.number_of_trees = number_of_trees
        self.number_of_random_features = number_of_random_features
        self.max_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
        self.trees = []

    def fit(self, X, y):
        for i in range(0, self.number_of_trees):
            indices = self.select_samples(X, y)
            regression_tree = Regression_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=self.number_of_random_features)
            regression_tree.fit(X[indices], y[indices])
            self.trees.append(regression_tree)
            
    def select_samples(self, X, y):
        return np.random.choice(y.shape[0], y.shape[0], True)
    
    def predict(self, X, y=None):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return np.mean(predictions, axis=0)

    def feature_importances(self):
        feature_importances = np.column_stack([tree.feature_importances() for tree in self.trees])
        return np.mean(feature_importances, axis=1)

    def direction_of_change(self):
        direction_of_change = np.column_stack([tree.direction_of_change() for tree in self.trees])
        return np.mean(direction_of_change, axis=1)

    def permutation_importances(self, X, y):
        permutation_importances = np.column_stack([tree.permutation_importances(X, y) for tree in self.trees])
        return np.mean(permutation_importances, axis=1)

class Classification_Random_Forest():
    def __init__(self, number_of_trees=100, number_of_random_features=1, max_depth=100, minimum_samples_split=20):
            self.number_of_trees = number_of_trees
            self.number_of_random_features = number_of_random_features
            self.max_depth = max_depth
            self.minimum_samples_split = minimum_samples_split
            self.trees = []

    def fit(self, X, y):
        for i in range(0, self.number_of_trees):
            indices = self.select_samples(X, y)
            classification_tree = Classification_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=self.number_of_random_features)
            classification_tree.fit(X[indices], y[indices])
            self.trees.append(classification_tree)

    def select_samples(self, X, y):
        return np.random.choice(y.shape[0], y.shape[0], True)
    
    def predict(self, X, y=None):
        predictions = []
        for tree in self.trees:
            prediction = tree.predict(X)
            predictions.append(prediction)
        predictions = np.array(predictions)
        return  np.mean(predictions, axis=0)

    def feature_importances(self):
        feature_importances = np.column_stack([tree.feature_importances() for tree in self.trees])
        return np.mean(feature_importances, axis=1)

    def direction_of_change(self):
        direction_of_change = np.column_stack([tree.direction_of_change() for tree in self.trees])
        return np.mean(direction_of_change, axis=1)

    def permutation_importances(self, X, y):
        permutation_importances = np.column_stack([tree.permutation_importances(X, y) for tree in self.trees])
        return np.mean(permutation_importances, axis=1)

class Regression_Gradient_Boosting():
    def __init__(self, number_of_trees=100, max_depth=3, minimum_samples_split=25, learning_rate=0.1):
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
        self.learning_rate = learning_rate
        self.initial_prediction = 0
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)
        current_prediction = np.full(shape=y.shape, fill_value=self.initial_prediction)
        for i in range(0, self.number_of_trees):
            residuals = y - current_prediction
            regression_tree = Regression_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=X.shape[1])
            regression_tree.fit(X, residuals)
            current_prediction = current_prediction + self.learning_rate * regression_tree.predict(X)
            self.trees.append(regression_tree)

    def predict(self, X):
        current_prediction = np.full(shape=X.shape[0], fill_value=self.initial_prediction)
        for tree in self.trees:
            current_prediction = current_prediction + self.learning_rate * tree.predict(X)
        return current_prediction

    def feature_importances(self):
        sse_decreases = np.column_stack([tree.sse_decreases for tree in self.trees])
        sse_decrease_per_feature = np.sum(sse_decreases, axis=1)
        return sse_decrease_per_feature / np.sum(sse_decrease_per_feature)

    def direction_of_change(self):
        direction_of_change = np.column_stack([tree.direction_of_change() for tree in self.trees])
        return np.mean(direction_of_change, axis=1)

    def permutation_importances(self, X, y):
        current_prediction = np.full(shape=y.shape, fill_value=self.initial_prediction)
        permutation_importances = []
        for tree in self.trees:
            residuals = y - current_prediction
            current_prediction = current_prediction + self.learning_rate * tree.predict(X)
            permutation_importance = tree.permutation_importances(X, residuals)
            permutation_importances.append(permutation_importance)
        permutation_importances = np.column_stack(permutation_importances)
        return np.mean(permutation_importances, axis=1)

class Classification_Gradient_Boosting():
    def __init__(self, number_of_trees=100, max_depth=3, minimum_samples_split=25, learning_rate=0.1):
        self.number_of_trees = number_of_trees
        self.max_depth = max_depth
        self.minimum_samples_split = minimum_samples_split
        self.learning_rate = learning_rate
        self.initial_prediction = 0
        self.trees = []

    def fit(self, X, y):
        self.initial_prediction = np.log((np.sum(y == 1) + 1e-8) / (np.sum(y == 0) + 1e-8))
        current_prediction = np.full(shape=y.shape, fill_value=self.initial_prediction)
        for i in range(0, self.number_of_trees):
            p = 1 / (1 + np.exp(-current_prediction))
            residuals = y - p
            regression_tree = Regression_Tree(max_depth=self.max_depth, minimum_samples_split=self.minimum_samples_split, number_of_random_features=X.shape[1], hessian=p*(1-p))
            regression_tree.fit(X, residuals)
            current_prediction = current_prediction + self.learning_rate * regression_tree.predict(X)
            self.trees.append(regression_tree)
            
    def predict(self, X):
        current_prediction = np.full(shape=X.shape[0], fill_value=self.initial_prediction)
        for tree in self.trees:
            current_prediction = current_prediction + self.learning_rate * tree.predict(X)
        return 1 / (1 + np.exp(-current_prediction))

    def feature_importances(self):
        sse_decreases = np.column_stack([tree.sse_decreases for tree in self.trees])
        sse_decrease_per_feature = np.sum(sse_decreases, axis=1)
        return sse_decrease_per_feature / np.sum(sse_decrease_per_feature)

    def direction_of_change(self):
        direction_of_change = np.column_stack([tree.direction_of_change() for tree in self.trees])
        return np.mean(direction_of_change, axis=1)

    def permutation_importances(self, X, y):
        current_prediction = np.full(shape=y.shape, fill_value=self.initial_prediction)
        permutation_importances = []
        for tree in self.trees:
            p = 1 / (1 + np.exp(-current_prediction))
            residuals = y - p
            current_prediction = current_prediction + self.learning_rate * tree.predict(X)
            permutation_importance = tree.permutation_importances(X, residuals)
            permutation_importances.append(permutation_importance)
        permutation_importances = np.column_stack(permutation_importances)
        return np.mean(permutation_importances, axis=1)

class Layer_Regression():
    def __init__(self, input_size, hidden_size, input, last=False):
        self.weight = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, hidden_size))
        self.weight_gradient = np.array([])
        self.bias_gradient = np.array([])
        self.input = input
        self.linear = self.input @ self.weight + self.bias
        self.last = last
        if self.last:
            self.activation = self.linear
        else:
            self.activation = np.where(self.linear > 0, self.linear, 0)
        self.previous = None
        self.next = None
        self.first_moment_weight = np.zeros_like(self.weight)
        self.second_moment_weight = np.zeros_like(self.weight)
        self.first_moment_bias = np.zeros_like(self.bias)
        self.second_moment_bias = np.zeros_like(self.bias)

    def recompute(self, input):
        self.input = input
        self.linear = self.input @ self.weight + self.bias
        if self.last:
            self.activation = self.linear
        else:
            self.activation = np.where(self.linear > 0, self.linear, 0)

class Regression_Neural_Network():
    def __init__(self, hidden_layer_number, hidden_size, epochs=10000, learning_rate=0.001):
        self.hidden_layer_number = hidden_layer_number
        self.hidden_size = hidden_size
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999

    def build_network(self, X, y, current_number=1, previous_layer=None):
        if current_number >= self.hidden_layer_number:
            output_layer = Layer_Regression(self.hidden_size, 1, previous_layer.activation, last=True)
            output_layer.previous = previous_layer
            output_layer.next = None
            self.layers.append(output_layer)
            return output_layer
        layer = Layer_Regression(self.hidden_size, self.hidden_size, previous_layer.activation)
        layer.previous = previous_layer
        layer.next = self.build_network(X, y, current_number+1, layer)
        self.layers.append(layer)
        return layer

    def fit(self, X, y):
        y_ = y.reshape(-1, 1).copy()
        input_layer = Layer_Regression(X.shape[1], self.hidden_size, X)
        input_layer.previous = None
        input_layer.next = self.build_network(X, y_, 1, input_layer)
        self.layers.append(input_layer)

        for i in range(0, self.epochs):
            previous_layer = None
            for layer in reversed(self.layers):
                if previous_layer is None:
                    layer.recompute(X)
                    previous_layer = layer
                else:
                    layer.recompute(previous_layer.activation)
                    previous_layer = layer

            y_hat = self.layers[0].activation
            n = len(y_)
            delta = -2 * (y_ - y_hat) / n

            for layer in self.layers:
                layer.bias_gradient = np.sum(delta, axis=0, keepdims=True)
                layer.weight_gradient = layer.input.T @ delta

                if layer.previous is not None:
                    delta = (delta @ layer.weight.T) * (layer.previous.linear > 0)

                layer.first_moment_weight = self.beta_1 * layer.first_moment_weight + (1 - self.beta_1) * layer.weight_gradient
                layer.first_moment_bias = self.beta_1 * layer.first_moment_bias + (1 - self.beta_1) * layer.bias_gradient
                layer.second_moment_weight = self.beta_2 * layer.second_moment_weight + (1 - self.beta_2) * np.square(layer.weight_gradient)
                layer.second_moment_bias = self.beta_2 * layer.second_moment_bias + (1 - self.beta_2) * np.square(layer.bias_gradient)

                m_hat_weight = layer.first_moment_weight / (1 - self.beta_1 ** (i+1))
                v_hat_weight = layer.second_moment_weight / (1 - self.beta_2 ** (i+1))
                m_hat_bias = layer.first_moment_bias / (1 - self.beta_1 ** (i+1))
                v_hat_bias = layer.second_moment_bias / (1 - self.beta_2 ** (i+1))

                layer.weight = layer.weight - self.learning_rate * (m_hat_weight / (np.sqrt(v_hat_weight) + 1e-8))
                layer.bias = layer.bias - self.learning_rate * (m_hat_bias / (np.sqrt(v_hat_bias) + 1e-8))

    def predict(self, X):
        previous_layer = None
        for layer in reversed(self.layers):
            if previous_layer is None:
                layer.recompute(X)
                previous_layer = layer
            else:
                layer.recompute(previous_layer.activation)
                previous_layer = layer
        y_hat = previous_layer.activation.reshape(-1).copy()
        return y_hat

    def permutation_importances(self, X, y):
        predictions_full = self.predict(X)
        rmse_full = np.sqrt(np.mean((y - predictions_full)**2))
        permutation_importances = []
        for feature in range(0, X.shape[1]):
            feature_rmse_increase = []
            for i in range(0, 10):
                X_ = X.copy()
                np.random.shuffle(X_[:, feature])
                predictions = self.predict(X_)
                rmse = np.sqrt(np.mean((y - predictions)**2))
                rmse_increase = rmse - rmse_full
                feature_rmse_increase.append(rmse_increase)
            permutation_importances.append(np.mean(feature_rmse_increase))
        permutation_importances = np.array(permutation_importances)
        permutation_importances = np.where(permutation_importances > 0, permutation_importances, 0)
        permutation_importances = permutation_importances / np.sum(permutation_importances)
        return permutation_importances

class Layer_Classification():
    def __init__(self, input_size, hidden_size, input):
        self.weight = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, hidden_size))
        self.weight_gradient = np.array([])
        self.bias_gradient = np.array([])
        self.input = input
        self.linear = self.input @ self.weight + self.bias
        self.activation = 1 / (1 + np.exp(-self.linear))
        self.previous = None
        self.next = None
        self.first_moment_weight = np.zeros_like(self.weight)
        self.second_moment_weight = np.zeros_like(self.weight)
        self.first_moment_bias = np.zeros_like(self.bias)
        self.second_moment_bias = np.zeros_like(self.bias)

    def recompute(self, input):
        self.input = input
        self.linear = self.input @ self.weight + self.bias
        self.activation = 1 / (1 + np.exp(-self.linear))

class Classification_Neural_Network():
    def __init__(self, hidden_layer_number, hidden_size, epochs=10000, learning_rate=0.001):
        self.hidden_layer_number = hidden_layer_number
        self.hidden_size = hidden_size
        self.layers = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.beta_1 = 0.9
        self.beta_2 = 0.999

    def build_network(self, X, y, current_number=1, previous_layer=None):
        if current_number >= self.hidden_layer_number:
            output_layer = Layer_Classification(self.hidden_size, 1, previous_layer.activation)
            output_layer.previous = previous_layer
            output_layer.next = None
            self.layers.append(output_layer)
            return output_layer
        layer = Layer_Classification(self.hidden_size, self.hidden_size, previous_layer.activation)
        layer.previous = previous_layer
        layer.next = self.build_network(X, y, current_number+1, layer)
        self.layers.append(layer)
        return layer

    def fit(self, X, y):
        y_ = y.reshape(-1, 1).copy()
        input_layer = Layer_Classification(X.shape[1], self.hidden_size, X)
        input_layer.previous = None
        input_layer.next = self.build_network(X, y_, 1, input_layer)
        self.layers.append(input_layer)

        for i in range(0, self.epochs):
            previous_layer = None
            for layer in reversed(self.layers):
                if previous_layer is None:
                    layer.recompute(X)
                    previous_layer = layer
                else:
                    layer.recompute(previous_layer.activation)
                    previous_layer = layer

            y_hat = self.layers[0].activation
            n = len(y_)
            delta = -2 * (y_ - y_hat) / n

            for layer in self.layers:
                layer.bias_gradient = np.sum(delta, axis=0, keepdims=True)
                layer.weight_gradient = layer.input.T @ delta

                if layer.previous is not None:
                    delta = (delta @ layer.weight.T) * (layer.previous.linear > 0)

                layer.first_moment_weight = self.beta_1 * layer.first_moment_weight + (1 - self.beta_1) * layer.weight_gradient
                layer.first_moment_bias = self.beta_1 * layer.first_moment_bias + (1 - self.beta_1) * layer.bias_gradient
                layer.second_moment_weight = self.beta_2 * layer.second_moment_weight + (1 - self.beta_2) * np.square(layer.weight_gradient)
                layer.second_moment_bias = self.beta_2 * layer.second_moment_bias + (1 - self.beta_2) * np.square(layer.bias_gradient)

                m_hat_weight = layer.first_moment_weight / (1 - self.beta_1 ** (i+1))
                v_hat_weight = layer.second_moment_weight / (1 - self.beta_2 ** (i+1))
                m_hat_bias = layer.first_moment_bias / (1 - self.beta_1 ** (i+1))
                v_hat_bias = layer.second_moment_bias / (1 - self.beta_2 ** (i+1))

                layer.weight = layer.weight - self.learning_rate * (m_hat_weight / (np.sqrt(v_hat_weight) + 1e-8))
                layer.bias = layer.bias - self.learning_rate * (m_hat_bias / (np.sqrt(v_hat_bias) + 1e-8))

    def predict(self, X):
        previous_layer = None
        for layer in reversed(self.layers):
            if previous_layer is None:
                layer.recompute(X)
                previous_layer = layer
            else:
                layer.recompute(previous_layer.activation)
                previous_layer = layer
        y_hat = previous_layer.activation.reshape(-1).copy()
        return y_hat

    def permutation_importances(self, X, y):
        predictions_full = self.predict(X)
        predictions_full = [1 if x > 0.5 else 0 for x in predictions_full]
        accuracy_full = np.mean(y == predictions_full)
        permutation_importances = []
        for feature in range(0, X.shape[1]):
            feature_accuracy_decrease = []
            for i in range(0, 10):
                X_ = X.copy()
                np.random.shuffle(X_[:, feature])
                predictions = self.predict(X_)
                predictions = [1 if x > 0.5 else 0 for x in predictions]
                accuracy = np.mean(y == predictions)
                accuracy_decrease = accuracy_full - accuracy
                feature_accuracy_decrease.append(accuracy_decrease)
            permutation_importances.append(np.mean(feature_accuracy_decrease))
        permutation_importances = np.array(permutation_importances)
        permutation_importances = np.where(permutation_importances > 0, permutation_importances, 0)
        permutation_importances = permutation_importances / np.sum(permutation_importances)
        return permutation_importances

def train_test_split(number_of_elements, testing_size):
    training_index = [i for i in range(0, number_of_elements)]
    testing_index = []
    for i in range(1, testing_size + 1):
        choice = np.random.choice(training_index, replace=False)
        training_index.remove(choice)
        testing_index.append(choice)
    return training_index, testing_index

def kfold_cross_validation(number_of_folds, data_size):
    lists = []
    for i in range(0, number_of_folds):
        lists.append([])
    for j in range(0, data_size):
        list_index = j % number_of_folds
        lists[list_index].append(j)
    return lists