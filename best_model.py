from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif


X = df.drop(columns = ['label'])
y = df['label']

train_size = 0.7
test_size = 0.3

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

select = SelectKBest(score_func = mutual_info_classif, k=5)
X_select = select.fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_select, y, test_size = 0.3, random_state = 42)

rf = RandomForestClassifier(random_state = 42)

para_grid = {'n_estimators': [100, 200, 300],
             'max_depth': [None, 10, 20, 30],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1,2,4] }

grid_search = GridSearchCV(rf, para_grid, cv=5, scoring = 'accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

rf_best = RandomForestClassifier(random_state = 42, **best_params)
rf_best.fit(X_train, y_train)

cv_scores = cross_val_score(rf_best, X_train, y_train, cv = 5, scoring='accuracy')
mean_cv_score = cv_scores.mean()

test_score = rf_best.score(X_test, y_test)

print("Best hyperparameters:", best_params)
print("Mean cross-validation accuracy:", mean_cv_score)
print("Test set accuracy:", test_score)
