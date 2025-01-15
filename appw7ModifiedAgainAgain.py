import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
from deap import base, creator, tools, algorithms
from pyswarm import pso
import time
import random

# Load and preprocess data
def load_data(file):
    data = pd.read_csv(file)
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

# Define the fitness function for GA
def evaluate(individual, X_train, X_test, y_train, y_test):
    selected_features = [index for index in range(len(individual)) if individual[index] == 1]
    if len(selected_features) == 0:
        return 0,
    
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    mutual_info = mutual_info_classif(X_train, y_train)
    relevancy = np.sum(mutual_info[selected_features]) / len(selected_features)
    
    return accuracy,

# Define the fitness function for PSO
def fitness_function(x, X_train, X_test, y_train, y_test):
    selected_features = [i for i in range(len(x)) if x[i] > 0.5]
    if len(selected_features) == 0:
        return 1.0
    
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    
    mutual_info = mutual_info_classif(X_train, y_train)
    relevancy = np.sum(mutual_info[selected_features]) / len(selected_features)
    
    return 1.0 - accuracy


def systematic_initialization(lb, ub, num_particles):
    dim = len(lb)
    particles = []
    
    # Calculate step size for systematic sampling
    step = (np.array(ub) - np.array(lb)) / (num_particles - 1)
    
    for i in range(num_particles):
        particle = lb + step * i
        particles.append(particle)
    
    return np.array(particles)


# Genetic Algorithm
def run_ga(X_train, X_test, y_train, y_test):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.randint, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X_train.columns))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    population = toolbox.population(n=50)
    NGEN = 20
    CXPB = 0.5
    MUTPB = 0.2

    start_time = time.time()
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = list(map(toolbox.evaluate, offspring))
        
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        population[:] = toolbox.select(offspring, len(population))

    best_individual_ga = tools.selBest(population, k=1)[0]
    end_time = time.time()

    return best_individual_ga, end_time - start_time

# Particle Swarm Optimization
def run_pso(X_train, X_test, y_train, y_test):
    lb = [0] * len(X_train.columns)
    ub = [1] * len(X_train.columns)

    start_time = time.time()
    xopt, _ = pso(lambda x: fitness_function(x, X_train, X_test, y_train, y_test), lb, ub, swarmsize=50, maxiter=20)
    end_time = time.time()

    return xopt, end_time - start_time

# Mixed Initialization
def mixed_initialization(swarmsize, num_features):
    particles = []
    for _ in range(swarmsize):
        if random.random() < 0.75:  # 75% chance of small initialization
            # Small Initialization
            particle = np.zeros(num_features)
            selected_features = random.sample(range(num_features), k=random.randint(1, num_features//2))
            for idx in selected_features:
                particle[idx] = random.uniform(0, 1)
        else:
            # Large Initialization
            particle = np.ones(num_features)
            selected_features = random.sample(range(num_features), k=random.randint(num_features//2, num_features))
            for idx in selected_features:
                particle[idx] = random.uniform(0, 1)
        particles.append(particle)
    return np.array(particles)

# Custom PSO implementation
def run_mod_pso(X_train, X_test, y_train, y_test):
    lb = np.zeros(len(X_train.columns))
    ub = np.ones(len(X_train.columns))
    
    swarmsize = 50
    maxiter = 20
    num_features = len(X_train.columns)
    
    # Initialize particles with Mixed Initialization
    particles = mixed_initialization(swarmsize, num_features)
    velocities = np.random.uniform(-1, 1, (swarmsize, num_features))
    personal_best_positions = particles.copy()
    personal_best_scores = np.array([fitness_function(p, X_train, X_test, y_train, y_test) for p in particles])
    global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    
    start_time = time.time()
    
    # PSO loop
    for _ in range(maxiter):
        for i in range(swarmsize):
            r1, r2 = np.random.random(size=2)
            velocities[i] = 0.5 * velocities[i] + 1.5 * r1 * (personal_best_positions[i] - particles[i]) + 1.5 * r2 * (global_best_position - particles[i])
            particles[i] = np.clip(particles[i] + velocities[i], lb, ub)
            
            score = fitness_function(particles[i], X_train, X_test, y_train, y_test)
            
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i].copy()
        
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
    
    end_time = time.time()
    
    return global_best_position, end_time - start_time


# Streamlit App
st.title("Feature Selection with GA and PSO")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    data = preprocess_data(data)

    st.write("Dataset Preview:")
    st.write(data.head())

    # Define features and target
    target_column = st.selectbox("Select the target column", data.columns)
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Baseline accuracy
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Baseline Accuracy: {baseline_accuracy:.4f}")

    if st.button("Run Feature Selection"):
        # GA
        # st.write("Running Genetic Algorithm...")
        # best_individual_ga, ga_time = run_ga(X_train, X_test, y_train, y_test)
        # selected_features_ga = [index for index in range(len(best_individual_ga)) if best_individual_ga[index] == 1]

        # X_train_selected_ga = X_train.iloc[:, selected_features_ga]
        # X_test_selected_ga = X_test.iloc[:, selected_features_ga]

        # dt = DecisionTreeClassifier(random_state=42)
        # dt.fit(X_train_selected_ga, y_train)
        # y_pred_ga = dt.predict(X_test_selected_ga)
        # selected_accuracy_ga = accuracy_score(y_test, y_pred_ga)

        # # PSO
        # st.write("Running Particle Swarm Optimization...")
        # xopt, pso_time = run_pso(X_train, X_test, y_train, y_test)
        # selected_features_pso = [i for i in range(len(xopt)) if xopt[i] > 0.5]

        # X_train_selected_pso = X_train.iloc[:, selected_features_pso]
        # X_test_selected_pso = X_test.iloc[:, selected_features_pso]

        # dt = DecisionTreeClassifier(random_state=42)
        # dt.fit(X_train_selected_pso, y_train)
        # y_pred_pso = dt.predict(X_test_selected_pso)
        # selected_accuracy_pso = accuracy_score(y_test, y_pred_pso)

        # Modified PSO
        st.write("Running Modified Particle Swarm Optimization...")
        xopt_mod, mod_pso_time = run_mod_pso(X_train, X_test, y_train, y_test)
        selected_features_mod_pso = [i for i in range(len(xopt_mod)) if xopt_mod[i] > 0.5]

        X_train_selected_mod_pso = X_train.iloc[:, selected_features_mod_pso]
        X_test_selected_mod_pso = X_test.iloc[:, selected_features_mod_pso]

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X_train_selected_mod_pso, y_train)
        y_pred_mod_pso = dt.predict(X_test_selected_mod_pso)
        selected_accuracy_mod_pso = accuracy_score(y_test, y_pred_mod_pso)

        # Display results
        st.write("Results:")
        results = pd.DataFrame({
            "Method": ["Baseline", "Modified PSO"],
            "Accuracy": [baseline_accuracy, selected_accuracy_mod_pso],
            "Features Selected": [len(X.columns), len(selected_features_mod_pso)],
            "Computation Time": ["N/A", f"{mod_pso_time:.2f} seconds"]
        })
        st.table(results)

        # Display selected features
        st.write("Selected Features:")
        # st.write("GA:", [X.columns[i] for i in selected_features_ga])
        # st.write("PSO:", [X.columns[i] for i in selected_features_pso])
        st.write("Modified PSO:", [X.columns[i] for i in selected_features_mod_pso])

else:
    st.write("Please upload a CSV file to begin.")