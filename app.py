import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load the "iris" dataset from Scikit-learn
iris = load_iris()

# Prepare data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)

# Streamlit app
st.title('Iris Species Prediction')

# Sidebar with additional options
st.sidebar.header('Customize Prediction')
sepal_length = st.sidebar.slider('Sepal Length', 4.0, 8.0, 5.4)
sepal_width = st.sidebar.slider('Sepal Width', 2.0, 5.0, 3.4)
petal_length = st.sidebar.slider('Petal Length', 1.0, 7.0, 1.3)
petal_width = st.sidebar.slider('Petal Width', 0.1, 2.5, 0.2)

show_data = st.sidebar.checkbox('Show Input Data')
if show_data:
    st.sidebar.write('Input Data:')
    st.sidebar.write(f'Sepal Length: {sepal_length}')
    st.sidebar.write(f'Sepal Width: {sepal_width}')
    st.sidebar.write(f'Petal Length: {petal_length}')
    st.sidebar.write(f'Petal Width: {petal_width}')

new_data = [[sepal_length, sepal_width, petal_length, petal_width]]

if st.sidebar.button('Predict'):
    prediction = clf.predict(new_data)
    species_names = iris.target_names
    st.write('---')
    st.write('**Prediction Result**')
    st.write(f'Predicted Species: {species_names[prediction][0]}')

# Generate and show correlation heatmap
if st.checkbox('Show Correlation Heatmap'):
    st.write('---')
    st.write('**Correlation Heatmap**')
    
    # Convert X_train to a DataFrame
    df_train = pd.DataFrame(X_train, columns=iris.feature_names)
    
    # Calculate correlation matrix
    correlation_matrix = df_train.corr()
    
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    st.pyplot()  # Display the heatmap using st.pyplot()


