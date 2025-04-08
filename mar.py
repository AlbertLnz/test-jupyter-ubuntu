import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sweetviz as sv
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# CSV
file_path = 'diabetes_binary_health_indicators_BRFSS2015.csv'
df_original = pd.read_csv(file_path)

print(df_original.info())

print(df_original.describe())

print(df_original.isnull().sum())

duplicados = df_original.duplicated().sum()
print(f"Número de filas duplicadas: {duplicados}")

df = df_original.drop_duplicates()

print(f"Número de filas después de eliminar duplicados: {df.shape[0]}")

# OUTLIERS
plt.figure(figsize = (15,15))
for i,col in enumerate(['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']):
    plt.subplot(4,2,i+1)
    sns.boxplot(x = col, data = df)
plt.show()

# Detect outliers using IQR
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f'{col}: {len(outliers)} outliers detectados')
    return outliers, lower_bound, upper_bound

# Aplicar la función
outlier_dict = {}
for col in ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth']:
    outliers, lb, ub = detect_outliers(df, col)
    outlier_dict[col] = {'outliers': outliers, 'lower_bound': lb, 'upper_bound': ub}

# Acceder a los outliers de una variable específica, por ejemplo BMI
outlier_dict['BMI']['outliers'].head()

# DROP OUTLIERS BMI
def detect_outliers_bmi(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrar outliers
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f'{col}: {len(outliers)} outliers detectados')
    return lower_bound, upper_bound

lower_bound, upper_bound = detect_outliers_bmi(df, 'BMI')
df_cleaned = df[(df['BMI'] >= lower_bound) & (df['BMI'] <= upper_bound)]

# Verifica el tamaño del DataFrame antes y después de eliminar outliers
print(f"Cantidad de registros antes de eliminar outliers: {len(df)}")
print(f"Cantidad de registros después de eliminar outliers: {len(df_cleaned)}")

df = df_cleaned

df.describe()

data2 = df.copy()

data2["Diabetes_binary_str"]= data2["Diabetes_binary"].replace({0:"NOn-Diabetic",1:"Diabetic"})

# Convertir las columnas a tipo 'object' para evitar problemas de asignación
data2['Age'] = data2['Age'].astype('object')
data2['Diabetes_binary'] = data2['Diabetes_binary'].astype('object')
data2['HighBP'] = data2['HighBP'].astype('object')
data2['HighChol'] = data2['HighChol'].astype('object')
data2['CholCheck'] = data2['CholCheck'].astype('object')
data2['Smoker'] = data2['Smoker'].astype('object')
data2['Stroke'] = data2['Stroke'].astype('object')
data2['HeartDiseaseorAttack'] = data2['HeartDiseaseorAttack'].astype('object')
data2['PhysActivity'] = data2['PhysActivity'].astype('object')
data2['Fruits'] = data2['Fruits'].astype('object')
data2['Veggies'] = data2['Veggies'].astype('object')
data2['HvyAlcoholConsump'] = data2['HvyAlcoholConsump'].astype('object')
data2['AnyHealthcare'] = data2['AnyHealthcare'].astype('object')
data2['NoDocbcCost'] = data2['NoDocbcCost'].astype('object')
data2['GenHlth'] = data2['GenHlth'].astype('object')
data2['DiffWalk'] = data2['DiffWalk'].astype('object')
data2['Sex'] = data2['Sex'].astype('object')

# Mapeo de valores numéricos a categorías
age_map = {
    1: '18 to 24', 2: '25 to 29', 3: '30 to 34', 4: '35 to 39', 5: '40 to 44',
    6: '45 to 49', 7: '50 to 54', 8: '55 to 59', 9: '60 to 64', 10: '65 to 69',
    11: '70 to 74', 12: '75 to 79', 13: '80 or older'
}

genhlth_map = {5: 'Excellent', 4: 'Very Good', 3: 'Good', 2: 'Fair', 1: 'Poor'}

binary_map = {0: 'No', 1: 'Yes'}

sex_map = {0: 'Female', 1: 'Male'}

# Aplicar el mapeo a las columnas correspondientes
data2['Age'] = data2['Age'].map(age_map)
data2['Diabetes_binary'] = data2['Diabetes_binary'].map({0: 'No Diabetes', 1: 'Diabetes'})
data2['HighBP'] = data2['HighBP'].map({0: 'No High BP', 1: 'High BP'})
data2['HighChol'] = data2['HighChol'].map({0: 'No High Cholesterol', 1: 'High Cholesterol'})
data2['CholCheck'] = data2['CholCheck'].map({0: 'No Cholesterol Check in 5 Years', 1: 'Cholesterol Check in 5 Years'})
data2['Smoker'] = data2['Smoker'].map(binary_map)
data2['Stroke'] = data2['Stroke'].map(binary_map)
data2['HeartDiseaseorAttack'] = data2['HeartDiseaseorAttack'].map(binary_map)
data2['PhysActivity'] = data2['PhysActivity'].map(binary_map)
data2['Fruits'] = data2['Fruits'].map(binary_map)
data2['Veggies'] = data2['Veggies'].map(binary_map)
data2['HvyAlcoholConsump'] = data2['HvyAlcoholConsump'].map(binary_map)
data2['AnyHealthcare'] = data2['AnyHealthcare'].map(binary_map)
data2['NoDocbcCost'] = data2['NoDocbcCost'].map(binary_map)
data2['GenHlth'] = data2['GenHlth'].map(genhlth_map)
data2['DiffWalk'] = data2['DiffWalk'].map(binary_map)
data2['Sex'] = data2['Sex'].map(sex_map)

# CORRELATION MATRIX
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

data2.hist(figsize=(20,15));

cols = ['HighBP', 'HighChol', 'CholCheck','Smoker', 'Stroke', 'HeartDiseaseorAttack', 'PhysActivity', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'DiffWalk']

def create_plot_pivot(data2, x_column):
    _df_plot = data2.groupby([x_column, 'Diabetes_binary']).size() \
    .reset_index().pivot(columns='Diabetes_binary', index=x_column, values=0)
    return _df_plot

fig, ax = plt.subplots(3, 4, figsize=(20,20))
axe = ax.ravel()

c = len(cols)

for i in range(c):
    create_plot_pivot(data2, cols[i]).plot(kind='bar',stacked=True, ax=axe[i])
    axe[i].set_xlabel(cols[i])

fig.show()

sns.boxplot(x = 'Diabetes_binary_str', y = 'BMI', data = data2)
plt.title('BMI vs Diabetes_binary_str')
plt.grid()
plt.show()

# CORRELATION WITH DIABETES

df.drop(columns=['Diabetes_binary']).corrwith(df['Diabetes_binary']).plot(
    kind='bar',
    grid=True,
    figsize=(20, 8),
    color="purple",
    title="Correlation with Diabetes_binary"
)
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.show()

# VIF TEST
vif = df.select_dtypes(include=['number'])

vif_data = pd.DataFrame()
vif_data["Variable"] = vif.columns
vif_data["VIF"] = [variance_inflation_factor(vif.values, i) for i in range(vif.shape[1])]

print(vif_data)

X_reducedd = vif.drop(columns=['Education', 'Income', 'CholCheck', 'AnyHealthcare', 'Fruits', 'Veggies', 'NoDocbcCost', 'Sex'])

vif_data_reduced = pd.DataFrame()
vif_data_reduced["Variable"] = X_reducedd.columns
vif_data_reduced["VIF"] = [variance_inflation_factor(X_reducedd.values, i) for i in range(X_reducedd.shape[1])]

print(vif_data_reduced)

# Chi-square
data3 = df.copy()

X = data3.iloc[:,1:]
Y = data3.iloc[:,0]

#apply SelectKBest class to extract top 10 best features
BestFeatures = SelectKBest(score_func=chi2, k=10)
fit = BestFeatures.fit(X,Y)

df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)

#concatenating two dataframes for better visualization
f_Scores = pd.concat([df_columns,df_scores],axis=1)
f_Scores.columns = ['Feature','Score']

f_Scores

columnas = ["Fruits" , "Veggies" , "Sex" , "CholCheck" , "AnyHealthcare", "NoDocbcCost" , "Education", "Income"]
df_models = df.copy()
df_models.drop(columnas, axis=1, inplace=True)

print(df_models.info())

X = df_models.iloc[:,1:]
Y = df_models.iloc[:,0]

Y.value_counts()

from imblearn.under_sampling import ClusterCentroids
from collections import Counter

# Ver distribución original
print("Antes del undersampling:", Counter(Y))

# Aplicar undersampling
cc = ClusterCentroids(random_state=42)
X_resampled, Y_resampled = cc.fit_resample(X, Y)

# Ver distribución balanceada
print("Después del undersampling:", Counter(Y_resampled))

X_train , X_test , Y_train , Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.2 , random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Alternative PCA
df_pca = df.copy()

X_pca= df_pca.drop("Diabetes_binary",axis=1)
Y_pca= df_pca["Diabetes_binary"]

X_trainn , X_testt , Y_trainn , Y_testt = train_test_split(X_pca, Y_pca, test_size=0.2 , random_state=42)

X_trainn_scaled = scaler.fit_transform(X_trainn)
X_testt_scaled = scaler.transform(X_testt)

pca_95 = PCA(n_components=0.95)
X_train_pca = pca_95.fit_transform(X_train_scaled)
X_test_pca = pca_95.transform(X_test_scaled)

models = {
    "Regresión Logística": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, Y_train)
    Y_pred = model.predict(X_test_scaled)

    results.append({
        "Modelo": name,
        "Accuracy": accuracy_score(Y_test, Y_pred),
        "Precision": precision_score(Y_test, Y_pred),
        "Recall": recall_score(Y_test, Y_pred),
        "F1-Score": f1_score(Y_test, Y_pred)
    })

results_df = pd.DataFrame(results)
results_df

# Alternative PCA

for name, model in models.items():
    model.fit(X_train_pca, Y_trainn)
    Y_predd = model.predict(X_test_pca)

    results.append({
        "Modelo": name,
        "Accuracy": accuracy_score(Y_testt, Y_predd),
        "Precision": precision_score(Y_testt, Y_predd),
        "Recall": recall_score(Y_testt, Y_predd),
        "F1-Score": f1_score(Y_testt, Y_predd)
    })

results_pca = pd.DataFrame(results)
results_pca

dt = DecisionTreeClassifier( max_depth= 12)
dt.fit(X_train , Y_train)

# make predictions on test set
y_pred=dt.predict(X_test)

print('Training set score: {:.4f}'.format(dt.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(dt.score(X_test, Y_test)))

matrix = classification_report(Y_test,y_pred )
print(matrix)

xg = XGBClassifier(eval_metric= 'error', learning_rate= 0.1)
xg.fit(X_train , Y_train)

y_pred=xg.predict(X_test)

print('Training set score: {:.4f}'.format(xg.score(X_train, Y_train)))

print('Test set score: {:.4f}'.format(xg.score(X_test, Y_test)))

matrix = classification_report(Y_test,y_pred )
print(matrix)

# Red neuronal simple
model = Sequential()
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Entrenamiento
history = model.fit(X_train_scaled, Y_train,
                    validation_data=(X_test_scaled, Y_test),
                    epochs=50, batch_size=32, verbose=0)

# Predicción y evaluación
Y_pred_dl = (model.predict(X_test_scaled) > 0.5).astype("int32")

accuracy = accuracy_score(Y_test, Y_pred_dl)
precision = precision_score(Y_test, Y_pred_dl)
recall = recall_score(Y_test, Y_pred_dl)
f1 = f1_score(Y_test, Y_pred_dl)

print("Modelo Deep Learning")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

