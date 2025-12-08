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
from SecurityMarketplace import SecurityMarketplace
from VCGPathAuction import VCGPathAuction

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

# --- FUNZIONE AGGIUNTA PER MARKET ALLOCATION ---
def plot_market_allocation(buyers, vendors, matches, output_path, title):
    """Visualizza l'allocazione del marketplace (con Istogramma Buyer e Bubble Chart Vendor)"""
    # Crea la figura con 2 subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7)) # Reso un po' più largo
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # --- Subplot 1: Istogramma Impilato dei Budget (INVARIATO) ---
    matched_buyers_ids = {m[0] for m in matches if m[1] is not None}
    
    budgets_matched = [b['budget'] for b in buyers if b['id'] in matched_buyers_ids]
    budgets_unmatched = [b['budget'] for b in buyers if b['id'] not in matched_buyers_ids]
    
    bins = np.arange(0, 101, 10) 
    
    ax1.hist([budgets_matched, budgets_unmatched], 
             bins=bins, 
             stacked=True, 
             color=['#FF6B6B', '#95E1D3'], 
             label=['Matched', 'Unmatched'],
             edgecolor='black')

    ax1.set_xlabel('Fascia di Budget', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Numero di Buyer (Count)', fontsize=12, fontweight='bold')
    ax1.set_title('Distribuzione Budget (Rosso=Matched)', fontsize=12, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_xticks(bins) 
    ax1.legend()
    # --- Fine Subplot 1 ---

    
    # --- MODIFICA Subplot 2: Sostituito con Bubble Chart ---
    
    # 1. Raccogli i dati per ogni vendor
    vendor_data = {v['id']: {
        'price': v['price'],
        'security_level': v['security_level'],
        'sales_count': 0,
        'utility_sum': 0
    } for v in vendors}

    for buyer_id, vendor_id, utility in matches:
        if vendor_id is not None:
            vendor_data[vendor_id]['sales_count'] += 1
            vendor_data[vendor_id]['utility_sum'] += utility

    # 2. Estrai le liste per il plotting
    prices = [d['price'] for d in vendor_data.values()]
    security_levels = [d['security_level'] for d in vendor_data.values()]
    sales_counts = [d['sales_count'] for d in vendor_data.values()]
    
    # Calcola l'utilità media, gestendo la divisione per zero se un vendor non vende nulla
    avg_utilities = [
        (d['utility_sum'] / d['sales_count']) if d['sales_count'] > 0 else 0 
        for d in vendor_data.values()
    ]
    
    # 3. Disegna il Bubble Chart
    # La dimensione 's' è scalata per leggibilità (es. vendite^2 * 10)
    sizes = [(s**1.5 * 20) + 10 for s in sales_counts] # +10 per vedere anche chi non vende
    
    scatter = ax2.scatter(prices, security_levels, s=sizes, c=avg_utilities, 
                          cmap='viridis', alpha=0.7, edgecolors='black')
    
    # Aggiungi etichette per i Vendor ID
    for v_id, data in vendor_data.items():
        ax2.text(data['price'] + 1, data['security_level'] + 0.1, str(v_id), 
                 fontsize=9, ha='left')

    # Aggiungi una Color Bar per l'utilità
    cbar = fig.colorbar(scatter, ax=ax2, pad=0.05)
    cbar.set_label('Utilità Media del Match', fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Prezzo (€)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Livello di Sicurezza', fontsize=12, fontweight='bold')
    ax2.set_title('Analisi Venditori (Dimensione = Vendite)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 11) # Limiti per livello di sicurezza 1-10
    ax2.set_xlim(0, 101) # Limiti per prezzo 1-100
    # --- Fine Modifica Subplot 2 ---
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta per il titolo generale
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Market Allocation plot saved as: {output_path}")

# --- FUNZIONE AGGIUNTA PER VCG PATH AUCTION (TASK 4) ---
def visualize_vcg_path(graph, security_set, optimal_path, source, target, output_path, title):
    """Visualizza il percorso trovato dal VCG (con colori di sicurezza sul percorso)"""
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42)
    
    # --- 1. Mappatura Colori per TUTTI i nodi ---
    # Creiamo una mappa {nodo: colore} per un accesso facile
    node_color_map = {}
    for node in graph.nodes():
        if node == source:
            node_color_map[node] = "#4853B1"  # 1. Blu per source
        elif node == target:
            node_color_map[node] = "#DA9A1B"  # 2. Arancione per target
        
        # 3. Logica per i nodi SUL PERCORSO (non source/target)
        elif node in optimal_path:
            if node in security_set:
                node_color_map[node] = '#6BCB77'  # Verde (Sicuro sul percorso)
            else:
                node_color_map[node] = "#FF0000"  # Rosso (Insicuro sul percorso)
        
        # 4. Logica per i nodi NON SUL PERCORSO
        elif node in security_set:
            node_color_map[node] = '#95E1D3'  # Azzurro (Sicuro, non percorso)
        else:
            node_color_map[node] = '#E5E5E5'  # Grigio (Insicuro, non percorso)
    
    # Lista di colori nell'ordine di graph.nodes(), per lo sfondo
    all_node_colors = [node_color_map[node] for node in graph.nodes()]

    
    # --- 2. Disegna lo sfondo "fantasma" ---
    nx.draw_networkx_nodes(graph, pos, node_color=all_node_colors, node_size=500, alpha=0.2)
    nx.draw_networkx_edges(graph, pos, alpha=0.05, width=1)
    
    # --- 3. Evidenzia il percorso (disegnando sopra) ---
    path_edges = [(optimal_path[i], optimal_path[i+1]) for i in range(len(optimal_path)-1)]
    path_nodes = list(optimal_path)
    
    # --- MODIFICA CHIAVE ---
    # Costruisci la lista dei colori del percorso NELLO STESSO ORDINE di path_nodes
    path_node_colors = [node_color_map[node] for node in path_nodes]
    # --- Fine Modifica ---
    
    # Ridisegna solo i nodi del percorso (opaco)
    nx.draw_networkx_nodes(graph, pos, nodelist=path_nodes, node_color=path_node_colors, node_size=700, alpha=1.0, 
                           edgecolors='black', linewidths=0.5)
    
    # Ridisegna solo gli archi del percorso (opaco)
    nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='#FF6B6B', width=4, alpha=1.0)
    
    # --- Etichette Selettive ---
    labels = {node: node for node in optimal_path}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=10, font_weight='bold')
    
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"VCG Path Visualization saved as: {output_path}")

def run_vcg_auction(graph, security_set, network_name, output_dir="results"):
    print(f"\n[TASK 4] VCG PATH AUCTION - {network_name}")
    print("-" * 70)
    
    # Seleziona Source e Target casuali (assicurandosi che siano diversi)
    nodes_list = list(graph.nodes())
    if len(nodes_list) < 2:
        print("Grafo troppo piccolo per VCG.")
        return

    source = random.choice(nodes_list)
    target = random.choice(nodes_list)
    while target == source:
        target = random.choice(nodes_list)
        
    print(f"Source: {source} -> Target: {target}")
    
    # Inizializza l'asta VCG
    vcg = VCGPathAuction(graph, security_set, penalty_weight=10)
    
    # Esegui il meccanismo
    optimal_path, total_cost, payments = vcg.run_vcg_mechanism(source, target)
    
    if optimal_path is None:
        print("Nessun percorso trovato tra Source e Target.")
        return

    print(f"Optimal Path found: {optimal_path}")
    print(f"Total Social Cost: {total_cost}")
    
    print("\nPayments to Winning Nodes:")
    for node, data in payments.items():
        print(f"  - Node {node}: Bid={data['bid']}, Payment={data['payment']:.2f}")
        
    # Visualizza il risultato
    visualize_vcg_path(
        graph, security_set, optimal_path, source, target,
        f"{output_dir}/{network_name}_vcg_path.png",
        f"VCG Optimal Path - {network_name}"
    )

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

def run_and_report(game, algo_class, algo_name, plot_file, graph_file, max_iter=100, **algo_kwargs):
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
    return security_set

def run_market_simulation(final_set, network_name, output_dir="results"):
    print(f"\n[TASK 3] MARKET SIMULATION (Matching) - {network_name}")
    print("-" * 70)
    market = SecurityMarketplace(final_set)

    # --- Scenario 1: Capacità Infinita ---
    print("SCENARIO 1: Capacità Infinita")
    matches_inf, welfare = market.run_scenario_infinite_capacity()
    
    matched_count = sum(1 for m in matches_inf if m[1] is not None)
    print(f"Social Welfare Totale: {welfare}")
    if len(final_set) > 0:
        tasso_match = (matched_count / len(final_set) * 100)
        print(f"Tasso di Match: {tasso_match:.1f}%")
    else:
        print("Tasso di Match: N/A (Il set di sicurezza è vuoto)")
        print("AVVISO: La dinamica ha prodotto un set di sicurezza vuoto, il che è un risultato anomalo se il grafo ha archi.")

    # Plot market INFINITO
    plot_market_allocation(
        market.buyers, market.vendors, matches_inf,
        f"{output_dir}/{network_name}_market_infinite.png", 
        f"Market Allocation (Unlimited Capacity) - {network_name}"
    )

    # --- Scenario 2: Capacità Limitata ---
    print("\nSCENARIO 2: Capacità Limitata (max_items=2)")
    matches_lim, welfare_lim = market.run_scenario_limited_capacity()
    
    matched_count_lim = sum(1 for m in matches_lim if m[1] is not None)
    print(f"  Social Welfare Totale: {welfare_lim:.2f}")
    print(f"  Buyers Matched: {matched_count_lim}/{len(final_set)}")
    if len(final_set) > 0:
        tasso_match = (matched_count_lim / len(final_set) * 100)
        print(f"  Tasso di Match: {tasso_match:.1f}%")
    else:
        print("Tasso di Match: N/A (Il set di sicurezza è vuoto)")
        print("AVVISO: La dinamica ha prodotto un set di sicurezza vuoto, il che è un risultato anomalo se il grafo ha archi.")

    print(f"\nDettagli Matches (Limited):")
    for buyer_id, vendor_id, utility in matches_lim[:10]:  # Prime 10
        if vendor_id is not None:
            print(f"  - Buyer {buyer_id} -> Vendor {vendor_id} (Utility: {utility:.2f})")
        else:
            print(f"  - Buyer {buyer_id} -> Non assegnato")
    if len(matches_lim) > 10:
        print(f"  ... e altri {len(matches_lim)-10} buyers")
    print("")
    
    # Plot market LIMITATO
    plot_market_allocation(
        market.buyers, market.vendors, matches_lim,
        f"{output_dir}/{network_name}_market_limited.png", 
        f"Market Allocation (Limited Capacity) - {network_name}"
    )

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # --- Step 1: Create a k-regular Graph ---
    num_nodes = 200 # Reduced node count for better visualization
    k = 3
    max_iter = 100
    update_fraction_fictitious = 0.2

    G_reg = create_regular_graph(num_nodes, k)
    game_reg = SecurityGame(G_reg, alpha=10, c=4)
    run_and_report(game_reg, BestResponseDynamics, "Best Response Dynamics (Regular)", "results/brd_regular_convergence.png", "results/brd_regular_graph.png", max_iter)
    run_and_report(game_reg, FictitiousPlay, "Batch Fictitious Play (Regular)", "results/fictitious_play_regular_convergence.png", "results/fictitious_play_regular_graph.png", max_iter, update_fraction=update_fraction_fictitious)
    run_and_report(game_reg, RegretMatching, "Regret Matching (Regular)", "results/regret_matching_regular_convergence.png", "results/regret_matching_regular_graph.png", max_iter)
    security_set_reg = run_coalitional_game(G_reg, "Regular", "results/shapley_regular_heatmap.png", "results/shapley_regular_result.png")
    run_market_simulation(security_set_reg, "Regular")
    run_vcg_auction(G_reg, security_set_reg, "Regular")

    G_erdos = create_erdos_renyi(num_nodes, p=0.05)
    game_erdos = SecurityGame(G_erdos, alpha=10, c=4)
    run_and_report(game_erdos, BestResponseDynamics, "Best Response Dynamics (Erdős-Rényi)", "results/brd_erdos_convergence.png", "results/brd_erdos_graph.png", max_iter)
    run_and_report(game_erdos, FictitiousPlay, "Batch Fictitious Play (Erdős-Rényi)", "results/fictitious_play_erdos_convergence.png", "results/fictitious_play_erdos_graph.png", max_iter, update_fraction=update_fraction_fictitious)
    run_and_report(game_erdos, RegretMatching, "Regret Matching (Erdős-Rényi)", "results/regret_matching_erdos_convergence.png", "results/regret_matching_erdos_graph.png", max_iter)
    security_set_erdos = run_coalitional_game(G_erdos, "Erdős-Rényi", "results/shapley_erdos_heatmap.png", "results/shapley_erdos_result.png")
    run_market_simulation(security_set_erdos, "Erdos_Renyi")
    run_vcg_auction(G_erdos, security_set_erdos, "Erdos_Renyi")

    G_ba = create_barabasi_albert(num_nodes, m=2)
    game_ba = SecurityGame(G_ba, alpha=10, c=4)
    run_and_report(game_ba, BestResponseDynamics, "Best Response Dynamics (Barabasi-Albert)", "results/brd_barabasi_convergence.png", "results/brd_barabasi_graph.png", max_iter)
    run_and_report(game_ba, FictitiousPlay, "Batch Fictitious Play (Barabasi-Albert)", "results/fictitious_play_barabasi_convergence.png", "results/fictitious_play_barabasi_graph.png", max_iter, update_fraction=update_fraction_fictitious)
    run_and_report(game_ba, RegretMatching, "Regret Matching (Barabasi-Albert)", "results/regret_matching_barabasi_convergence.png", "results/regret_matching_barabasi_graph.png", max_iter)
    security_set_ba = run_coalitional_game(G_ba, "Barabasi-Albert", "results/shapley_barabasi_heatmap.png", "results/shapley_barabasi_result.png")
    run_market_simulation(security_set_ba, "Barabasi_Albert")
    run_vcg_auction(G_ba, security_set_ba, "Barabasi_Albert")