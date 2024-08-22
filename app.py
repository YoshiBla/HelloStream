
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


#Page configuration
st.set_page_config (
  page_title = 'Iris prediccion',
  layout = 'wide',
  initial_sidebar_state = 'expanded'
)

# Titulo de la app
st.title ('Prediccion clase de orquidea')

# Lectura de datos
iris = load_iris ()
df = pd.DataFrame (iris.data, columns = iris.feature_names)

st.write (df)

maximo = df.iloc[0:].max()
minimo = df.iloc[0:].min()

sepalL = st.sidebar.slider ('sepal length (cm)', minimo[0], maximo[0], minimo[0])
sepalW = st.sidebar.slider ('sepal width (cm)', minimo[1], maximo[1], minimo[1])
petalL = st.sidebar.slider ('petal length (cm)', minimo[2], maximo[2], minimo[2])
petalW = st.sidebar.slider ('petal width (cm)', minimo[3], maximo[3], minimo[3])

df ['target']=iris.target

X_train, X_test, y_train, y_test = train_test_split (
    df.drop (['target'], axis='columns'), iris.target,
    test_size = 0.2
)

model = RandomForestClassifier ()
model.fit (X_train, y_train)

y_predicted = model.predict (X_test)
st.write (iris.target_names[y_predicted[0]])

st.write ('Los datos seleccionados son:')
st.table (pd.DataFrame ({
    'sepal length (cm)': [sepalL],
    'sepal width (cm)': [sepalW],
    'petal length (cm)': [petalL],
    'petal width (cm)': [petalW]
}))

