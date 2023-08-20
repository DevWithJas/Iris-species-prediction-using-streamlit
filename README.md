# Iris-species-prediction-using-streamlit

Data Loading and Preparation:

The Iris dataset is loaded using scikit-learn's load_iris() function.
The data is split into features (X) and target labels (y) using train_test_split().
Classifier Training:

A K-Nearest Neighbors (KNN) classifier is initialized and trained on the training data using the fit() method.
Streamlit App Structure:

The Streamlit app starts with a title using st.title().
The sidebar is used for customization options and interactive elements.
Customization sliders for sepal length, sepal width, petal length, and petal width are added to the sidebar.
A checkbox is added to show or hide input data in the sidebar.
The "Predict" button triggers the prediction using the trained classifier when clicked.
Prediction Result Display:

The predicted species is displayed using the trained classifier's predictions and the Iris species names.
Correlation Heatmap Display:

The checkbox "Show Correlation Heatmap" is used to toggle the display of the correlation heatmap.
If the checkbox is selected, the heatmap is generated and displayed using Seaborn and Matplotlib.
X_train data is converted to a DataFrame df_train.
The correlation matrix of df_train is calculated using df_train.corr().
The correlation heatmap is created using sns.heatmap() and displayed using st.pyplot().
Footer:

A footer is added at the end of the app to give credit to the creator.

![heatmap](https://github.com/CreateJas/Iris-species-prediction-using-streamlit/assets/91935368/5e3f4351-d77c-4cc8-a669-ddbfc2ac702c)
