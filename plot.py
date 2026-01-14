import matplotlib.pyplot as plt
import numpy as np

# Dati
n_values = [200, 1000, 5000]

# Dati per Topologia
regular_data = {
    'Best Response': [0.0028, 0.0419, 0.8566],
    'Fictitious Play': [0.0127, 0.0674, 0.4003],
    'Regret Matching': [0.0641, 0.6532, 11.6361],
    'Coalitional (Shapley)': [0.6366, 4.2071, 52.6264],
}

erdos_data = {
    'Best Response': [0.0033, 0.0510, 1.6301],
    'Fictitious Play': [0.0224, 0.3420, 13.9256],
    'Regret Matching': [0.0848, 0.8966, 24.7239],
    'Coalitional (Shapley)': [0.7910, 9.8318, 309.2375],
}

barabasi_data = {
    'Best Response': [0.0026, 0.0492, 1.2743],
    'Fictitious Play': [0.0118, 0.0533, 0.4694],
    'Regret Matching': [0.0804, 0.7527, 18.6542],
    'Coalitional (Shapley)': [0.7985, 4.1929, 81.7217],
}

# Creazione subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
topologies = [('Regular Graph', regular_data), ('Erdős-Rényi', erdos_data), ('Barabasi-Albert', barabasi_data)]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

for idx, (title, data) in enumerate(topologies):
    ax = axes[idx]
    for i, (algo, values) in enumerate(data.items()):
        # Filtra i NaN per Barabasi SAT
        clean_n = [n for n, v in zip(n_values, values) if not np.isnan(v)]
        clean_v = [v for v in values if not np.isnan(v)]
        ax.plot(clean_n, clean_v, marker=markers[i], color=colors[i], label=algo, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Nodes (N)')
    ax.set_yscale('log')
    ax.grid(True, which="both", ls="--", alpha=0.3)
    if idx == 0: ax.set_ylabel('Time (s) [Log Scale]')

# Legenda unica
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5, fontsize=12)
plt.tight_layout()
plt.savefig('performance_by_topology.png', dpi=300, bbox_inches='tight')