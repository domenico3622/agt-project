import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import os

from SecurityGame import SecurityGame
from BestResponseDynamics import BestResponseDynamics
from FictitiousPlay import FictitiousPlay
from RegretMatching import RegretMatching
from CoalitionalSecurityGame import CoalitionalSecurityGame

# --- FUNZIONE AGGIUNTA PER STAMPARE LA CONVERGENZA ---
def plot_convergence(history, algorithm_name, filename="convergence.png"):
    """
    Traccia l'andamento del numero di nodi che giocano '1' nel tempo.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history, linewidth=2, label='Size of Security Set (Strategy = 1)')
    plt.xlabel('Iterations')
    plt.ylabel('Number of Nodes playing 1')
    plt.title(f'Convergence Analysis: {algorithm_name}')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Salva l'immagine
    plt.savefig(filename, dpi=300)
    plt.close() # Chiude la figura per liberare memoria
    print(f"Grafico di convergenza salvato come: {filename}")

# --- FUNZIONE AGGIUNTA PER STAMPARE IL GRAFO ---
def visualize_graph(graph, strategies, algorithm_name, filename="graph_viz.png"):
    """
    Visualizza il grafo colorando i nodi in base alla strategia.
    Rosso: Security Set (1)
    Blu: Non protetto (0)
    """
    plt.figure(figsize=(10, 8))
    
    # Definisci i colori
    node_colors = []
    for node in graph.nodes():
        if strategies.get(node) == 1:
            node_colors.append('#FF6B6B') # Rosso pastello per Security Set
        else:
            node_colors.append('#4ECDC4') # Turchese per gli altri
            
    # Layout del grafo (spring layout è spesso buono per visualizzare cluster)
    # Fissiamo il seed per coerenza visiva se necessario, o lasciamo libero
    pos = nx.spring_layout(graph, seed=42) 
    
    nx.draw(graph, pos, 
            node_color=node_colors, 
            with_labels=False, 
            node_size=50, 
            edge_color='gray', 
            alpha=0.7,
            width=0.5)
            
    # Legenda personalizzata
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Security Set (1)', markerfacecolor='#FF6B6B', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Unprotected (0)', markerfacecolor='#4ECDC4', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f'Graph Visualization: {algorithm_name}')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Visualizzazione grafo salvata come: {filename}")

# --- FUNZIONE AGGIUNTA PER HEATMAP SHAPLEY ---
def visualize_shapley_heatmap(graph, shapley_values, algorithm_name, filename="shapley_heatmap.png"):
    """
    Visualizza il grafo colorando i nodi in base al loro Shapley Value (Heatmap).
    """
    plt.figure(figsize=(10, 8))
    
    values = [shapley_values[n] for n in graph.nodes()]
    
    pos = nx.spring_layout(graph, seed=42)
    
    nodes = nx.draw_networkx_nodes(graph, pos, 
                                   node_color=values, 
                                   cmap=plt.cm.coolwarm, # Colormap 'coolwarm' (Blue -> Red)
                                   node_size=80,
                                   alpha=0.9)
    nx.draw_networkx_edges(graph, pos, edge_color='gray', alpha=0.3)
    
    plt.colorbar(nodes, label='Shapley Value')
    plt.title(f'Shapley Value Heatmap: {algorithm_name}')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Heatmap Shapley salvata come: {filename}")

# --- Validation Functions ---
def is_security_set(graph, security_set):
    if not security_set: 
        return graph.number_of_nodes() == 0
    if not all(node in graph.nodes for node in security_set):
        return False
    for node in graph.nodes:
        if node in security_set:
            continue
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return False
        if not all(neighbor in security_set for neighbor in neighbors):
            return False
    return True

def is_minimal_security_set(graph, security_set):
    if not is_security_set(graph, security_set):
        print("Error: The set is not a security set.")
        return False
    for node_to_remove in security_set:
        temp_set = security_set - {node_to_remove}
        if is_security_set(graph, temp_set):
            return False
    return True

# --- Main Execution ---
def create_regular_graph(n: int, k: int):
    return nx.random_regular_graph(k, n)

def create_erdos_renyi(n: int, p: float):
    return nx.gnp_random_graph(n, p)

def create_barabasi_albert(n: int, m: int):
    return nx.barabasi_albert_graph(n, m)

def run_and_report(game, algo_class, algo_name, plot_file, graph_file, max_iter=5000, **algo_kwargs):
    print(f"\n----- RUNNING {algo_name} -----")
    algo = algo_class(game, max_iterations=max_iter, **algo_kwargs)
    strats, is_pne, history = algo.run()
    
    # Plot convergenza
    plot_convergence(history, algo_name, plot_file)
    
    # Visualizzazione Grafo
    if game.graph.number_of_nodes() <= 2000: # Evita di plottare grafi troppo grandi se non necessario
        visualize_graph(game.graph, strats, algo_name, graph_file)
    
    result_set = {node for node, s in strats.items() if s == 1}
    print(f"Result Set Size: {len(result_set)}")
    if is_minimal_security_set(game.graph, result_set):
        print("The resulting set is a Minimal security Set.")
    else:
        print("The resulting set is NOT a Minimal security Set.")
    return result_set

def run_coalitional_game(graph, graph_name, heatmap_file, result_file):
    print(f"\n----- RUNNING Coalitional Game (Shapley + Reverse Greedy) on {graph_name} -----")
    cg = CoalitionalSecurityGame(graph)
    
    # 1. Calcolo Shapley Values
    print("Calculating Shapley Values (Monte Carlo)...")
    shapley_vals = cg.calculate_shapley_monte_carlo(num_permutations=500) # 500 per velocità
    
    # 2. Visualizza Heatmap
    visualize_shapley_heatmap(graph, shapley_vals, f"Shapley Values ({graph_name})", heatmap_file)
    
    # 3. Build Security Set (Reverse Greedy)
    print("Building Security Set via Reverse Greedy...")
    security_list = cg.build_security_set_from_shapley(shapley_vals)
    security_set = set(security_list)
    
    # Converti in formato strategies {node: 0/1} per visualizzazione
    strategies = {n: (1 if n in security_set else 0) for n in graph.nodes()}
    
    # 4. Visualizza Risultato
    visualize_graph(graph, strategies, f"Coalitional Result ({graph_name})", result_file)
    
    print(f"Result Set Size: {len(security_set)}")
    if is_minimal_security_set(graph, security_set):
        print("The resulting set is a Minimal security Set.")
    else:
        print("The resulting set is NOT a Minimal security Set.")

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # --- Step 1: Create a k-regular Graph ---
    num_nodes = 200 # Reduced node count for better visualization
    k = 3
    max_iter = 200
    update_fraction_fictitious = 0.2

    G_reg = create_regular_graph(num_nodes, k)
    game_reg = SecurityGame(G_reg, alpha=10, c=4)
    run_and_report(game_reg, BestResponseDynamics, "Best Response Dynamics (Regular)", "results/brd_regular_convergence.png", "results/brd_regular_graph.png", max_iter)
    run_and_report(game_reg, FictitiousPlay, "Batch Fictitious Play (Regular)", "results/fictitious_play_regular_convergence.png", "results/fictitious_play_regular_graph.png", max_iter, update_fraction=update_fraction_fictitious)
    run_and_report(game_reg, RegretMatching, "Regret Matching (Regular)", "results/regret_matching_regular_convergence.png", "results/regret_matching_regular_graph.png", max_iter)
    run_coalitional_game(G_reg, "Regular", "results/shapley_regular_heatmap.png", "results/shapley_regular_result.png")

    G_erdos = create_erdos_renyi(num_nodes, p=0.05)
    game_erdos = SecurityGame(G_erdos, alpha=10, c=4)
    run_and_report(game_erdos, BestResponseDynamics, "Best Response Dynamics (Erdős-Rényi)", "results/brd_erdos_convergence.png", "results/brd_erdos_graph.png", max_iter)
    run_and_report(game_erdos, FictitiousPlay, "Batch Fictitious Play (Erdős-Rényi)", "results/fictitious_play_erdos_convergence.png", "results/fictitious_play_erdos_graph.png", max_iter, update_fraction=update_fraction_fictitious)
    run_and_report(game_erdos, RegretMatching, "Regret Matching (Erdős-Rényi)", "results/regret_matching_erdos_convergence.png", "results/regret_matching_erdos_graph.png", max_iter)
    run_coalitional_game(G_erdos, "Erdős-Rényi", "results/shapley_erdos_heatmap.png", "results/shapley_erdos_result.png")

    G_ba = create_barabasi_albert(num_nodes, m=2)
    game_ba = SecurityGame(G_ba, alpha=10, c=4)
    run_and_report(game_ba, BestResponseDynamics, "Best Response Dynamics (Barabasi-Albert)", "results/brd_barabasi_convergence.png", "results/brd_barabasi_graph.png", max_iter)
    run_and_report(game_ba, FictitiousPlay, "Batch Fictitious Play (Barabasi-Albert)", "results/fictitious_play_barabasi_convergence.png", "results/fictitious_play_barabasi_graph.png", max_iter, update_fraction=update_fraction_fictitious)
    run_and_report(game_ba, RegretMatching, "Regret Matching (Barabasi-Albert)", "results/regret_matching_barabasi_convergence.png", "results/regret_matching_barabasi_graph.png", max_iter)
    run_coalitional_game(G_ba, "Barabasi-Albert", "results/shapley_barabasi_heatmap.png", "results/shapley_barabasi_result.png")