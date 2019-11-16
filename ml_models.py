from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier


class Regression:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def feature_scaling(data, scaler=None):
        """
        In layman's terms, fit_transform means to do some calculation and then do transformation (say calculating the
        means of columns from some data and then replacing the missing values). So for training set, you need to both
        calculate and do transformation.

        But for testing set, Machine learning applies prediction based on what was learned during the training set
        and so it doesn't need to calculate, it just performs the transformation.

        Standard Scalar will transform our data with mean =0 and SD=1
        """
        if not scaler:
            scaler = StandardScaler(with_mean=False)
            _data = scaler.fit_transform(data)
        else:
            _data = scaler.transform(data)

        return _data, scaler

    def fit_linear_regression(self):
        regressor = LinearRegression()
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def fit_polynomial_regression(self, degree):

        poly_reg = PolynomialFeatures(degree)
        X_poly = poly_reg.fit_transform(self.X_train)
        poly_reg.fit(X_poly, self.y_train)
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train)
        return regressor, poly_reg

    def fit_sv_regression(self, kernel='rbf', gamma='scale'):

        X_train, scaler_X = self.feature_scaling(self.X_train)
        # X_test, _ = self.feature_scaling(X_test, scaler_X)

        y_train, scaler_y = self.feature_scaling(self.y_train.reshape(-1, 1))
        y_train = y_train.reshape(len(y_train))
        # y_test, _ = self.feature_scaling(y_test.reshape(-1, 1), scaler_y)
        # y_test = y_test.reshape(len(y_test))

        regressor = SVR(kernel=kernel, gamma=gamma)
        regressor.fit(X_train, y_train)
        return regressor, X_train, y_train, scaler_X, scaler_y

    def fit_decision_tree_regression(self, criterion):

        regressor = DecisionTreeRegressor(criterion)
        regressor.fit(self.X_train, self.y_train)
        return regressor

    def fit_random_forest_regression(self, n_estimators, criterion):

        regressor = RandomForestRegressor(n_estimators, criterion)
        regressor.fit(self.X_train, self.y_train)
        return regressor


class Classification:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def fit_logistic_regression(self, solver):

        classifier = LogisticRegression(fit_intercept=True, dual=False, penalty='l2', solver=solver)
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def fit_knn(self, n_neighbors):
        classifier = KNeighborsClassifier(n_neighbors)
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def fit_svm(self, kernel, gamma, degree=3):

        classifier = SVC(C=1.0, kernel=kernel, degree=degree, gamma=gamma, probability=True)
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def fit_naive_bayes(self):

        classifier = GaussianNB()
        try:
            classifier.fit(self.X_train, self.y_train)
        except TypeError:
            # Avoiding error: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a
            # dense numpy array.
            classifier.fit(self.X_train.toarray(), self.y_train)

        return classifier

    def fit_decision_tree_classification(self, criterion, splitter):

        classifier = DecisionTreeClassifier(criterion, splitter)
        classifier.fit(self.X_train, self.y_train)
        return classifier

    def fit_random_forest_classification(self, n_estimators, criterion):

        classifier = RandomForestClassifier(n_estimators, criterion)
        classifier.fit(self.X_train, self.y_train)
        return classifier
