import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import brier_score_loss

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Read data from csv
df = pd.read_csv('diabetes.csv')

# Print number of rows and clumns in data
print('\nOur dataset has {} rows and {} attribues'.format(len(df), len(df.columns)))
print('\n')

# Solving for data impurities by replacing all zeroes with median values
df.loc[df['Glucose'] == 0,'Glucose'] = np.nan
df.loc[df['BloodPressure'] == 0,'BloodPressure'] = np.nan
df.loc[df['SkinThickness'] == 0,'SkinThickness'] = np.nan
df.loc[df['Insulin'] == 0,'Insulin'] = np.nan
df.loc[df['BMI'] == 0,'BMI'] = np.nan
df['Glucose'].fillna(df['Glucose'].median(), inplace = True)
df['BloodPressure'].fillna(df['BloodPressure'].median(), inplace = True)
df['SkinThickness'].fillna(df['SkinThickness'].median(), inplace = True)
df['Insulin'].fillna(df['Insulin'].median(), inplace = True)
df['BMI'].fillna(df['BMI'].median(), inplace = True)

# Separating features and label
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

# Plot count to check for data imbalance
Positive, Negative = y.value_counts()
print('----------Checking for Data Imbalance------------')
print('Number of Positive Outcomes: ',Positive,'\nPercentage of Positive Outcomes: {}%'.format(round(Positive/y.count()*100 , 2)))
print('Number of Negative Outcomes : ',Negative,'\nPercentage of Positive Outcomes: {}%'.format(round(Negative/y.count()*100 , 2)))
print('\n')
sns.countplot(y,label="Count")
plt.show()

# Checking for Correlation between attributes
f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()

# Plot Histogram to see the distribution of values for each attribute
X.hist(figsize=(10,10))
plt.show()

# Plot pairplot to plot multiple pairwise bi-variate distributions in our database
sns.pairplot(df, hue='Outcome')
plt.show()

# Train Test Data Split
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X, y, test_size=0.1, random_state=21, stratify = y)

# Reset Index of the split sets of data
X_train_all.reset_index(drop=True,inplace=True)
X_test_all.reset_index(drop=True,inplace=True)
y_train_all.reset_index(drop=True,inplace=True)
y_test_all.reset_index(drop=True,inplace=True)

# Normalize Data
X_train_all_std = (X_train_all - X_train_all.mean())/X_train_all.std()
X_test_all_std = (X_test_all - X_test_all.mean())/X_test_all.std()

# Define required function to fit the model and calculate metrics

def calc_metrics(matrix):
    metrics = []
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[1][1]
    TPR = TP/(TP+FN)
    TNR = TN/(TN+FP)
    FPR = FP/(TN+FP)
    FNR = FN/(TP+FN)
    Precision = TP/(TP+FP)
    F1_measure = (2*TP)/(2*TP+FP+FN)
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Error_rate = (FP+FN)/(TP+FP+FN+TN)
    BACC = (TPR+TNR)/2
    TSS = (TP/(TP+FN))-(FP/(FP+TN))
    HSS = 2*(TP*TN-FP*FN)/((TP+FN)*(FN+TN)+(TP+FP)*(FP+TN))
    metrics.append(TP)
    metrics.append(TN)
    metrics.append(FP)
    metrics.append(FN)
    metrics.append(TPR)
    metrics.append(TNR)
    metrics.append(FPR)
    metrics.append(FNR)
    metrics.append(Precision)
    metrics.append(F1_measure)
    metrics.append(Accuracy)
    metrics.append(Error_rate)
    metrics.append(BACC)
    metrics.append(TSS)
    metrics.append(HSS)
    return metrics


def get_metrics(model, X_train, X_test, y_train, y_test, LSTM_flag):
    if LSTM_flag == 1:
        # Convert data to numpy array
        Xtrain = X_train.to_numpy()
        Xtest = X_test.to_numpy()
        ytrain = y_train.to_numpy()
        ytest = y_test.to_numpy()
        # Reshape data
        shape = Xtrain.shape
        Xtrain_reshaped = Xtrain.reshape(len(Xtrain), shape[1], 1)
        Xtest_reshaped = Xtest.reshape(len(Xtest), shape[1], 1)
        model.fit(Xtrain_reshaped, ytrain, epochs=50, validation_data=(Xtest_reshaped, ytest),verbose=0)
        lstm_scores = model.evaluate(Xtest_reshaped, ytest, verbose=0)
        predict_prob = lstm_model.predict(Xtest_reshaped)
        pred_labels = predict_prob > 0.5
        pred_labels_1 = pred_labels.astype(int)
        matrix = confusion_matrix(ytest, pred_labels_1, labels=[1, 0])
        metrics = calc_metrics(matrix)
        Acc = lstm_scores[1]
        lstm_brier_score = brier_score_loss(ytest, predict_prob)
        lstm_roc_auc = roc_auc_score(ytest, predict_prob)
        metrics.append(lstm_brier_score)
        metrics.append(lstm_roc_auc)
        metrics.append(Acc)

    if LSTM_flag == 0:
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(y_test, predicted, labels=[1, 0])
        metrics = calc_metrics(matrix)
        Acc = model.score(X_test, y_test)
        predict_prob = model.predict_proba(X_test)
        predict_prob_1 = [item[1] for item in predict_prob]
        model_brier_score = brier_score_loss(y_test, predict_prob_1)
        model_roc_auc = roc_auc_score(y_test, predict_prob_1)
        metrics.append(model_brier_score)
        metrics.append(model_roc_auc)
        metrics.append(Acc)

    return metrics

# Parameter Tuning for KNN
knn_parameters = {"n_neighbors": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_model = KNeighborsClassifier()
knn_cv = GridSearchCV(knn_model, knn_parameters, cv=10, n_jobs=-1)
knn_cv.fit(X_train_all_std, y_train_all)
print("\nBest Parameters for KNN based on GridSearchCV : ", knn_cv.best_params_)
print('\n')
n_neighbors = knn_cv.best_params_['n_neighbors']

# Parameter Tuning for RF
rf_parameters = {"n_estimators": [10,20,30,40,50,60,70,80,90,100], "min_samples_split": [2,4,6,8,10]}
rf_model = RandomForestClassifier()
rf_cv = GridSearchCV(rf_model, rf_parameters, cv = 10, n_jobs=-1)
rf_cv.fit(X_train_all_std, y_train_all)
print("\nBest Parameters for Random Forest based on GridSearchCV : ", rf_cv.best_params_)
print('\n')
min_samples_split = rf_cv.best_params_['min_samples_split']
n_estimators = rf_cv.best_params_['n_estimators']

# Parameter Tuning for SVM
svc_parameters = {"kernel":["linear"], "C": [1,2,3,4,5,6,7,8,9,10]}
svc_model = SVC()
svc_cv = GridSearchCV(svc_model,svc_parameters, cv = 10, n_jobs=-1)
svc_cv.fit(X_train_all_std, y_train_all)
print("\nBest Parameters for SVC based on GridSearchCV : ", svc_cv.best_params_)
print('\n')
C = svc_cv.best_params_['C']

# Comparing the classifiers with selected parameters by using 10-Fold Stratified Cross-Validation to calculate all metrics
# Implementing 10-Fold Stratified Cross-Validation

CV = StratifiedKFold(n_splits = 10, shuffle = True, random_state=21)

metrics_knn = []
metrics_rf = []
metrics_svm = []
metrics_lstm = []

metric_columns = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'FPR', 'FNR', 'Precision',
                  'F1_measure', 'Accuracy', 'Error_rate', 'BACC', 'TSS', 'HSS', 'Brier_score',
                  'AUC', 'Acc_by_package_fn']
iter = 1

# 10 Iterations of 10-fold cross validation
for train_index, test_index in CV.split(X_train_all_std, y_train_all):
    # KNN Model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    # Random Forest Model
    rf_model = RandomForestClassifier(min_samples_split=min_samples_split, n_estimators=n_estimators)
    # SVM Classifier Model
    svm_model = SVC(C=C, kernel='linear', probability=True)
    # LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, activation='relu', batch_input_shape=(None, 8, 1), return_sequences=False))
    lstm_model.add(Dense(1, activation='sigmoid'))
    # Compile model
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    X_train, X_test = X_train_all_std.iloc[train_index, :], X_train_all_std.iloc[test_index, :]
    y_train, y_test = y_train_all[train_index], y_train_all[test_index]
    knn_metrics = get_metrics(knn_model, X_train, X_test, y_train, y_test, 0)
    rf_metrics = get_metrics(rf_model, X_train, X_test, y_train, y_test, 0)
    svm_metrics = get_metrics(svm_model, X_train, X_test, y_train, y_test, 0)
    lstm_metrics = get_metrics(lstm_model, X_train, X_test, y_train, y_test, 1)
    metrics_knn.append(knn_metrics)
    metrics_rf.append(rf_metrics)
    metrics_svm.append(svm_metrics)
    metrics_lstm.append(lstm_metrics)
    metrics_all = []
    metrics_all.append(knn_metrics)
    metrics_all.append(rf_metrics)
    metrics_all.append(svm_metrics)
    metrics_all.append(lstm_metrics)
    metric_all_index = ['KNN', 'RF', 'SVM', 'LSTM']
    metrics_all_df = pd.DataFrame(metrics_all, columns=metric_columns, index=metric_all_index)
    print('\nIteration {}: \n'.format(iter))
    print('\n----- Metrics for all Algorithms in Interation {} -----\n'.format(iter))
    print(metrics_all_df.round(decimals=2).T)
    print('\n')
    iter = iter + 1


metric_index = ['iter1','iter2','iter3','iter4','iter5','iter6','iter7','iter8','iter9','iter10']
knn_metrics = pd.DataFrame(metrics_knn, columns =metric_columns, index= metric_index)
rf_metrics = pd.DataFrame(metrics_rf, columns =metric_columns, index= metric_index)
svm_metrics = pd.DataFrame(metrics_svm, columns =metric_columns, index= metric_index)
lstm_metrics = pd.DataFrame(metrics_lstm, columns =metric_columns, index= metric_index)

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 12)

print('\nPerformance Metrics for K-Nearest Neighbour: \n')
print(knn_metrics.round(decimals=2).T)
print('\nPerformance Metrics for Random Forest: \n')
print(rf_metrics.round(decimals=2).T)
print('\nPerformance Metrics for SVM: \n')
print(svm_metrics.round(decimals=2).T)
print('\nPerformance Metrics for LSTM: \n')
print(lstm_metrics.round(decimals=2).T)
print('\n')


# Average metrics for all four classifiers
knn_avg = knn_metrics.mean()
rf_avg = rf_metrics.mean()
svm_avg = svm_metrics.mean()
lstm_avg = lstm_metrics.mean()
avg_performance = pd.DataFrame({'KNN': knn_avg,'RF': rf_avg,'SVM': svm_avg, 'LSTM': lstm_avg}, index=metric_columns)
print('\nAverage metrics for four classifiers: \n')
print(avg_performance.round(decimals=2))
print('\n')


# Comparing Performance of Algorithms using ROC curve and AUC on test data

# KNN Model ROC
knn_model = KNeighborsClassifier(n_neighbors = n_neighbors)
knn_model.fit(X_train_all_std, y_train_all)
predict_prob_knn = knn_model.predict_proba(X_test_all_std)
predict_prob_knn_1 = [item[1] for item in predict_prob_knn]
knn_roc_auc = roc_auc_score(y_test_all, predict_prob_knn_1)
fpr, tpr, thresholds = roc_curve(y_test_all, predict_prob_knn_1)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", label = "AUC (area = %0.2f)" %knn_roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN ROC")
plt.legend(loc="lower right")
plt.show()

# Random Forest Model ROC
rf_model = RandomForestClassifier(min_samples_split = min_samples_split, n_estimators = n_estimators)
rf_model.fit(X_train_all_std, y_train_all)
predict_prob_rf = rf_model.predict_proba(X_test_all_std)
predict_prob_rf_1 = [item[1] for item in predict_prob_rf]
rf_roc_auc = roc_auc_score(y_test_all, predict_prob_rf_1)
fpr, tpr, thresholds = roc_curve(y_test_all, predict_prob_rf_1)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", label = "AUC (area = %0.2f)" %rf_roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC")
plt.legend(loc="lower right")
plt.show()

# SVM Classifier Model ROC
svm_model = SVC(C=C, kernel = 'linear', probability=True )
svm_model.fit(X_train_all_std, y_train_all)
predict_prob_svm = svm_model.predict_proba(X_test_all_std)
predict_prob_svm_1 = [item[1] for item in predict_prob_svm]
svm_roc_auc = roc_auc_score(y_test_all, predict_prob_svm_1)
fpr, tpr, thresholds = roc_curve(y_test_all, predict_prob_svm_1)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", label = "AUC (area = %0.2f)" %svm_roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC")
plt.legend(loc="lower right")
plt.show()

# LSTM model ROC
lstm_model = Sequential()
lstm_model.add(LSTM(64, activation='relu', batch_input_shape = (None, 8,1), return_sequences = False))
lstm_model.add(Dense(1, activation='sigmoid'))
# Compile model
lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Convert data to numpy array
Xtrain = X_train_all_std.to_numpy()
Xtest = X_test_all_std.to_numpy()
ytrain = y_train_all.to_numpy()
ytest = y_test_all.to_numpy()
# Reshape data
input_shape = Xtrain.shape
input_train = Xtrain.reshape(len(Xtrain), input_shape[1],1)
input_test  = Xtest.reshape(len(Xtest), input_shape[1],1)
output_train = ytrain
output_test = ytest
lstm_model.fit(input_train, output_train, epochs = 50, validation_data=(input_test,output_test),verbose=0)
predict_lstm = lstm_model.predict(input_test)
lstm_roc_auc = roc_auc_score(y_test_all, predict_lstm)
fpr, tpr, thresholds = roc_curve(y_test_all, predict_lstm)
plt.figure()
plt.plot(fpr, tpr, color="darkorange", label = "AUC (area = %0.2f)" %lstm_roc_auc)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("LSTM ROC")
plt.legend(loc="lower right")
plt.show()


# Final DataFrame with all classifier's average metrics printed again
print('-----------Average Performance Metrics Comparison-----------')
print(avg_performance.round(decimals=2))
print('\n')