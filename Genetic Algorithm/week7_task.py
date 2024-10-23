import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random

# 讀取數據集
file_path = 'The_Cancer_data_1500_V2.csv'
data = pd.read_csv(file_path)

# 定義特徵和目標變數
X = data.drop(columns=['Diagnosis'])
y = data['Diagnosis']

# 將數據分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 遺傳算法的參數
population_size = 10  # 種群大小
num_generations = 10  # 進化代數
mutation_rate = 0.1   # 突變率

# 初始化種群（每個染色體長度為特徵數量，0 和 1 表示是否選擇該特徵）
def initialize_population(pop_size, num_features):
    return [np.random.randint(0, 2, num_features).tolist() for _ in range(pop_size)]

# 計算適應度（用隨機森林的準確率作為適應度函數）
def fitness_function(chromosome, X_train, X_test, y_train, y_test):
    selected_features = [i for i, bit in enumerate(chromosome) if bit == 1]
    
    if len(selected_features) == 0:  # 如果沒有選擇任何特徵，返回適應度0
        return 0
    
    # 訓練隨機森林分類器
    clf = RandomForestClassifier()
    clf.fit(X_train.iloc[:, selected_features], y_train)
    
    # 預測準確率作為適應度值
    predictions = clf.predict(X_test.iloc[:, selected_features])
    return accuracy_score(y_test, predictions)

# 選擇交配的父母（輪盤賽選擇法）
def select_parents(population, fitnesses):
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population), random.choice(population)
    probs = [fit / total_fitness for fit in fitnesses]
    parent1 = random.choices(population, probs)[0]
    parent2 = random.choices(population, probs)[0]
    return parent1, parent2

# 交配（單點交叉）
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# 突變
def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]  # 0 變 1 或 1 變 0
    return chromosome

# 主遺傳算法過程
def genetic_algorithm(X_train, X_test, y_train, y_test, pop_size, num_generations, mutation_rate):
    num_features = X_train.shape[1]
    population = initialize_population(pop_size, num_features)
    
    for generation in range(num_generations):
        fitnesses = [fitness_function(chrom, X_train, X_test, y_train, y_test) for chrom in population]
        
        # 找到當前最好的個體
        best_fitness = max(fitnesses)
        best_chromosome = population[fitnesses.index(best_fitness)]
        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Chromosome = {best_chromosome}")
        
        # 選擇父母並生成新一代
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1, mutation_rate))
            new_population.append(mutate(child2, mutation_rate))
        
        population = new_population[:pop_size]  # 保持種群大小不變
    
    return best_chromosome

# 運行遺傳算法
best_chromosome = genetic_algorithm(X_train, X_test, y_train, y_test, population_size, num_generations, mutation_rate)

# 打印最終結果
selected_features = [i for i, bit in enumerate(best_chromosome) if bit == 1]
print(f"最佳特徵選擇: {X.columns[selected_features].tolist()}")
