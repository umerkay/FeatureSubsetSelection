# streamlit_app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from deap import base, creator, tools, algorithms
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to preprocess the dataset
# def preprocess_data(df, target_column):
#     # Separate features and target
#     X = df.drop(columns=[target_column])
#     y = df[target_column]

#     # Identify categorical and numeric columns
#     numeric_features = X.select_dtypes(include=[float, int]).columns
#     categorical_features = X.select_dtypes(include=[object]).columns

#     # Preprocessing pipelines for numeric and categorical data
#     numeric_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='mean'))
#     ])

#     categorical_transformer = Pipeline(steps=[
#         ('imputer', SimpleImputer(strategy='most_frequent')),
#         ('onehot', OneHotEncoder(handle_unknown='ignore'))
#     ])

#     # Combine preprocessors
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])

#     # Apply transformations
#     X_processed = preprocessor.fit_transform(X)
    
#     return X_processed, y

# Load and preprocess data
def load_data(file):
    data = pd.read_csv(file, sep=';')
    return data

def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data.select_dtypes(include=[float, int])), columns=data.select_dtypes(include=[float, int]).columns)
    
    # Encode categorical variables
    categorical_columns = data.select_dtypes(include=[object]).columns
    le = LabelEncoder()
    for col in categorical_columns:
        data_imputed[col] = le.fit_transform(data[col])
    
    return data_imputed
# Function to run NSGA-II
def run_nsga_ii(X_train, y_train, num_generations, population_size):
    def evaluate(individual):
        selected_features = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected_features) == 0:
            return 1, 0  # Penalize if no features are selected

        X_selected = X_train.iloc[:, selected_features]
        classifier = DecisionTreeClassifier(random_state=42)
        accuracy = np.mean(cross_val_score(classifier, X_selected, y_train, cv=2))
        return 1 - accuracy, sum(individual)  # Minimize (1 - accuracy) and minimize the number of features

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))  # Minimize both objectives
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X_train.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate)

    population = toolbox.population(n=population_size)
    algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=40, cxpb=0.7, mutpb=0.2, ngen=num_generations, 
                              stats=None, halloffame=None, verbose=True)
    
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
    best_solution = min(pareto_front, key=lambda x: x.fitness.values)

    return pareto_front, best_solution

# Streamlit UI
st.title("NSGA-II for Feature Selection")
st.write("This app runs NSGA-II for feature selection using Random Forest as a benchmark.")

# Upload dataset
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Function to calculate baseline accuracy using all features
def calculate_baseline_accuracy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    return baseline_accuracy

if uploaded_file is not None:
    
    data = load_data(uploaded_file)
    data = preprocess_data(data)

    st.write("Dataset Preview:")
    st.write(data.head())

    # Define features and target
    target_column = st.selectbox("Select the target column", data.columns)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    st.write("Calculating baseline accuracy with all features...")
    baseline_accuracy = calculate_baseline_accuracy(X, y)
    st.write(f"Baseline Accuracy with all features: {baseline_accuracy:.4f}")
    
    # Split data into training and test sets
    test_size = st.sidebar.slider("Test Size", 0.1, 0.5, 0.3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Set NSGA-II parameters
    st.sidebar.title("NSGA-II Parameters")
    num_generations = st.sidebar.slider("Number of Generations", 10, 100, 20)
    population_size = st.sidebar.slider("Population Size", 10, 100, 20)

    # Run NSGA-II
    if st.sidebar.button("Run NSGA-II"):
        pareto_front, best_solution = run_nsga_ii(X_train, y_train, num_generations, population_size)

        # Visualize Pareto Front
        pareto_accuracies = [1 - ind.fitness.values[0] for ind in pareto_front]
        pareto_feature_counts = [ind.fitness.values[1] for ind in pareto_front]

        st.subheader("Pareto Front: Accuracy vs Number of Features")
        fig, ax = plt.subplots()
        sns.scatterplot(x=pareto_feature_counts, y=pareto_accuracies, ax=ax, s=100)
        ax.set_xlabel('Number of Features')
        ax.set_ylabel('Accuracy')
        ax.set_title('Pareto Front')
        st.pyplot(fig)

        # Show best solution
        selected_features = [i for i, bit in enumerate(best_solution) if bit == 1]
        st.subheader("Best Solution")
        st.write("Selected Features:", X.columns[selected_features])
        st.write("Number of Features Selected:", sum(best_solution))
        st.write("Fitness Values (1 - Accuracy, Number of Features):", best_solution.fitness.values)

        # Evaluate on test set
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_train.iloc[:, selected_features], y_train)
        test_accuracy = classifier.score(X_test.iloc[:, selected_features], y_test)
        st.write("Test Accuracy of the Best Solution:", test_accuracy)

        # Feature Importance Visualization
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1]
        selected_feature_names = X.columns[selected_features]

        st.subheader("Feature Importances of Selected Features")
        fig, ax = plt.subplots()
        ax.bar(range(len(selected_features)), importances[indices], align="center")
        ax.set_xticks(range(len(selected_features)))
        ax.set_xticklabels(selected_feature_names[indices], rotation=90)
        ax.set_title("Feature Importances")
        st.pyplot(fig)

        # Distribution of Selected Features Across Solutions
        feature_selection_counts = np.sum([ind for ind in pareto_front], axis=0)

        st.subheader("Frequency of Feature Selection Across Pareto-Optimal Solutions")
        fig, ax = plt.subplots()
        ax.bar(range(len(X.columns)), feature_selection_counts, align="center")
        ax.set_xticks(range(len(X.columns)))
        ax.set_xticklabels(X.columns, rotation=90)
        ax.set_ylabel('Number of Times Selected')
        ax.set_title("Feature Selection Frequency")
        st.pyplot(fig)

# To run the app locally:
# 1. Save this code in a file named `streamlit_app.py`.
# 2. Open a terminal and navigate to the directory containing this file.
# 3. Run the command `streamlit run streamlit_app.py`.
