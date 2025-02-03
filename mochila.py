import numpy as np
import random
import matplotlib.pyplot as plt

# Definição do problema
tam_populacao = 200
taxa_crossover = 0.05
taxa_mutacao = 0.7
num_geracoes = 250
capacidade_mochila = 50
penalidade = 10

# Definição dos itens (valor, peso)
itens = [(random.randint(10, 100), random.randint(1, 20)) for _ in range(20)]
num_itens = len(itens)

def fitness(individuo):
    valor_total = sum(itens[i][0] for i in range(num_itens) if individuo[i] == 1)
    peso_total = sum(itens[i][1] for i in range(num_itens) if individuo[i] == 1)
    if peso_total > capacidade_mochila:
        return valor_total - penalidade * (peso_total - capacidade_mochila)
    return valor_total

def gerar_populacao():
    return [np.random.randint(2, size=num_itens).tolist() for _ in range(tam_populacao)]

def selecao_roleta(populacao):
    soma_fitness = sum(fitness(ind) for ind in populacao)
    
    # If total fitness is 0, return two random individuals to avoid ZeroDivisionError
    if soma_fitness == 0:  
        return random.choices(populacao, k=2)
    
    selecionados = []
    for _ in range(2):
        pick = random.uniform(0, soma_fitness)
        atual = 0
        for ind in populacao:
            atual += fitness(ind)
            if atual >= pick:
                selecionados.append(ind)
                break
        # If no individual was selected, append a random one
        if len(selecionados) < _ + 1:  
            selecionados.append(random.choice(populacao))
    return selecionados

def crossover(pai1, pai2):
    if random.random() < taxa_crossover:
        ponto = random.randint(1, num_itens - 1)
        filho1 = pai1[:ponto] + pai2[ponto:]
        filho2 = pai2[:ponto] + pai1[ponto:]
        return filho1, filho2
    return pai1, pai2

def mutacao(individuo):
    for i in range(num_itens):
        if random.random() < taxa_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo

def algoritmo_genetico():
    populacao = gerar_populacao()
    historico_fitness = []
    
    for _ in range(num_geracoes):
        nova_populacao = []
        populacao = sorted(populacao, key=fitness, reverse=True)
        elite = populacao[:10]
        while len(nova_populacao) < tam_populacao - len(elite):
            pai1, pai2 = selecao_roleta(populacao)
            filho1, filho2 = crossover(pai1, pai2)
            nova_populacao.extend([mutacao(filho1), mutacao(filho2)])
        nova_populacao.extend(elite)
        populacao = nova_populacao[:tam_populacao]
        historico_fitness.append(max(fitness(ind) for ind in populacao))
    
    melhor_solucao = max(populacao, key=fitness)
    print(f'Melhor solução encontrada: {melhor_solucao}, Fitness: {fitness(melhor_solucao)}')
    
    plt.plot(historico_fitness)
    plt.xlabel('Gerações')
    plt.ylabel('Melhor Fitness')
    plt.title('Evolução do Fitness')
    plt.show()
    

algoritmo_genetico()