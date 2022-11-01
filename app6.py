import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

@st.cache(allow_output_mutation=True)

def load_data():
    file_path = "crunchbase.csv"
    df = pd.read_csv(file_path)
    #df.drop('Unnamed: 0', axis = 1, inplace = True)
    return df

df = load_data()

# Select a target (or create a target)
def success(rows):
    if rows['ipo'] == 1:
        return 1
    elif rows['is_acquired'] == 1:
        return 1
    elif rows['is_acquired'] == 0:
        return 0
# Apply success function to each row (axis=1)
df['success'] = df.apply(success, axis=1) 

df['number_degrees']= df.iloc[:, 10:14].sum(axis=1)

df['age_new'] = df['age'] //365

df = df[(df['age_new']>=3) & (df['age_new'] <=7)]

df[['ipo', 'is_acquired', 'is_closed']] = df[['ipo', 'is_acquired', 'is_closed']].replace({True : 1, False: 0})

df['average_funded'] = df['average_funded'].round(2)


#droping the columns which have more than 50% of missing values
df_clean = df.drop(["products_number", "acquired_companies", "mba_degree","phd_degree", "ms_degree", "other_degree", "age", 'ipo', 'is_acquired', 'is_closed'], axis = 1)

# dropping the rows with missing values which cannot be replaced
df_clean = df_clean.dropna()
df_clean.isnull().sum()

# Numerical data frame
df_num = df_clean.select_dtypes(exclude = ['object'])
df_cat = df_clean.select_dtypes(include = ['object'])
cnt = df_clean['success'].value_counts()
st.title('Start up prediction')
st.sidebar.title('Predicting the success of start up companies')

st.sidebar.subheader("Visualization Selector")
multiselect = st.sidebar.multiselect('Select the Plots', ('Correlation Heatmap', 'Target Distribution'))
st.set_option('deprecation.showPyplotGlobalUse', False)
if "Correlation Heatmap"in multiselect:
    st.subheader('Correlation Heatmap')
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_num.corr(), cmap = 'icefire', annot=True, fmt=".2f", ax=ax)
    ax.set_title("Correlations of numerical features and target")
    st.pyplot(fig)

if "Target Distribution" in multiselect:
	st.subheader("Bar plot")
	plt.style.use('ggplot')
	fig, ax = plt.subplots(figsize=(14, 10))
	sns.barplot(x = cnt.index, y=cnt.values, ax=ax)
	ax.set_xticklabels(['Unsucessful', 'Successful'])
	ax.set_xlabel("Successful Startup")
	ax.set_ylabel("Number of Startups")
	ax.set_title("Distribution of Success variable")
	st.pyplot(fig)
	successful = cnt[1]
	unuccessful = cnt[0]

df_series = round(df.isnull().sum() / df.shape[0], 2)
if st.sidebar.checkbox('Missing Values'):
	st.subheader('Percentage of Missing values')
	st.dataframe(df_series)
	st.write("Interpretation:")
	st.write("1. All variables with 50% or more missing values should be removed from the dataframe because they cannot be imputed and will not be useful for the machine learning algorithm.")
	st.write("2. The company's category contains around 10% missing values; however, because it is a categorical variable, it cannot be imputed, hence the corresponding rows will be discarded.")
	st.write("3. Since 'average funded' and 'offices' are numerical variables, the missing values can be replaced with mean or median.")
df[['average_funded', 'offices']] = df[['average_funded','offices']].fillna(df[['average_funded', 'offices']].mean()) 


df_cat = df_clean.select_dtypes(include = ['object'])
df_cat_encode = pd.get_dummies(df_cat, drop_first=True)
df_num1 = df_clean.select_dtypes(exclude = ['object'])

df_clean = pd.concat([df_num1, df_cat_encode], axis = 1)
X = df_clean.drop(columns = ['success'], axis = 1)
Y = df_clean['success']
smote = SMOTE()
x, y = smote.fit_resample(X,Y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, stratify=y, random_state = 30)

classifier_type = st.sidebar.selectbox('Select a Classifier Model', ('Random Forest Classifier', 'Decision Tree Classifier'))

#Random Forest Model

if classifier_type =='Random Forest Classifier':
	st.sidebar.subheader('Hyperparameters')
	max_depth = st.sidebar.number_input('Max Depth', 1, 100, step=1)
	decision_trees = st.sidebar.number_input('Decision Trees', 100, 5000, step=10)
	st.sidebar.write("Click on Classify button to generate results")
	if st.sidebar.button('Classify'):
		st.subheader('Random Forest')
		plt.style.use('ggplot')
		pipeline = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = 0)),('std_scaler', StandardScaler()), ('clf', RandomForestClassifier(class_weight = 'balanced'))])
		pipeline.fit(x_train, y_train)
		y_pred = pipeline.predict(x_test)
		st.write("Accuracy score: ", round(pipeline.score(x_test, y_test), 2), "%")
		#st.write("Classification Report: ","\n", classification_report(y_test, y_pred))

		conf_matrix = metrics.confusion_matrix(y_test, y_pred)
		fig, ax = plt.subplots(figsize=(6.4, 4.8))
		sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Pastel2_r)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('Actual')
		ax.set_title('Confusion matrix')
		ax.xaxis.set_ticklabels(['No', 'Yes'])
		ax.yaxis.set_ticklabels(['No', 'Yes'])
		st.pyplot(fig)

		fpr_un, tpr_un, threshold_un = roc_curve(y_test, y_pred)
		fig, ax = plt.subplots(figsize=(6.4, 4.8))
		plt.plot([0, 1], [0, 1], label = 'Baseline', linestyle = '--', color= 'red')
		plt.plot(fpr_un, tpr_un,color = 'yellow', label = 'Random Forest')
		plt.title('ROC curve')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive rate')
		plt.legend(loc='lower right')
		plt.grid(False)
		st.pyplot(fig)




#Decision Tree Model 
if classifier_type =='Decision Tree Classifier':
	st.sidebar.subheader('Hyperparameters')
	
	criterion = st.sidebar.radio("Criterion", ("gini", "entropy"))
	max_depth = st.sidebar.number_input('Max Depth', 1, 100, step=1)
	st.sidebar.write("Click on Classify button to generate results")
	if st.sidebar.button('Classify'):
		st.subheader('Decision Tree')
		plt.style.use('ggplot')
		pipeline_1 = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'constant', fill_value = 0)),('std_scaler', StandardScaler()), ('clf', DecisionTreeClassifier(class_weight = 'balanced'))])
		pipeline_1.fit(x_train, y_train)
		y_pred = pipeline_1.predict(x_test)
		st.write("Accuracy score: ", round(pipeline_1.score(x_test, y_test), 2), "%")
		#st.write("Classification Report: ","\n", classification_report(y_test, y_pred))

		conf_matrix = metrics.confusion_matrix(y_test, y_pred)
		fig, ax = plt.subplots(figsize=(6.4, 4.8))
		sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap=plt.cm.Pastel2_r)
		ax.set_xlabel('Predicted')
		ax.set_ylabel('Actual')
		ax.set_title('Confusion matrix')
		ax.xaxis.set_ticklabels(['No', 'Yes'])
		ax.yaxis.set_ticklabels(['No', 'Yes'])
		st.pyplot(fig)

		fpr_un, tpr_un, threshold_un = roc_curve(y_test, y_pred)
		fig, ax = plt.subplots(figsize=(6.4, 4.8))
		plt.plot([0, 1], [0, 1], label = 'Baseline', linestyle = '--', color= 'red')
		plt.plot(fpr_un, tpr_un,color = 'yellow', label = 'Random Forest')
		plt.title('ROC curve')
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive rate')
		plt.legend(loc='lower right')
		plt.grid(False)
		st.pyplot(fig)


if st.sidebar.checkbox("Comparision of both the models"):
	x = df_clean.drop(columns = ['success'], axis = 1)
	y = df_clean['success']
	smote = SMOTE()
	x, y = smote.fit_resample(X,Y)
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, stratify=y, random_state = 30)
	models=[]
	models.append(('Random Forest Classifier', RandomForestClassifier()))
	models.append(('Decision Tree Classifieri', DecisionTreeClassifier()))
	results = []
	names = []
	scoring = 'accuracy'
	for name, model in models:
		kfold = StratifiedKFold(n_splits=5, shuffle=False)
		cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
		results.append(cv_results)
		names.append(name)
		msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
		st.write(msg)
	fig, ax = plt.subplots(figsize=(6.4, 4.8))
	plt.boxplot(results)
	ax.set_xticklabels(names)
	plt.title('Random Forest vs Decision Tree Classifier')
	plt.grid(False)
	st.pyplot(fig)

if st.sidebar.checkbox("Important features"):
	x = df_clean.drop(columns = ['success'], axis = 1)
	y = df_clean['success']
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True, stratify=y, random_state = 30)
	rf_clf = RandomForestClassifier()
	rf_clf.fit(x_train, y_train)
	y_train_pred_rf = rf_clf.predict(x_train)
	y_test_pred_rf = rf_clf.predict(x_test)	
	plt.style.use('ggplot')
	fig, ax = plt.subplots(figsize=(11, 8))
	feat_importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
	n_10 = feat_importances.nlargest(10).plot(kind='barh')
	ax.set_xlabel('Feature importance')
	ax.set_ylabel('Feature')
	ax.set_title("Feature importance")
	st.pyplot(fig)

