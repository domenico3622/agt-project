import networkx as nx

class SecurityGame:
    def __init__(self, graph, alpha=10, c=4):
        self.graph = graph
        self.num_players = graph.number_of_nodes()
        self.alpha = alpha
        self.c = c

        if not (self.alpha > self.c > 0):
            raise ValueError("Parameters must satisfy alpha > c > 0")

    def get_payoff(self, player_id, strategies):
        s_i = strategies[player_id]
        is_covered = False
        if s_i == 1:
            is_covered = True
        else:
            neighbors = list(self.graph.neighbors(player_id))
            if not neighbors:
                is_covered = False
            else:
                is_covered = all(strategies.get(neighbor) == 1 for neighbor in neighbors)
        
        if s_i == 1:
            return self.alpha - self.c
        elif s_i == 0 and is_covered:
            return self.alpha
        elif s_i == 0 and not is_covered:
            return 0
        else:
            raise ValueError("Invalid strategy or state encountered.")

    def is_nash_equilibrium(self, strategies):
        for player_id in range(self.num_players):
            original_payoff = self.get_payoff(player_id, strategies)
            alt_strategies = strategies.copy()
            alt_strategies[player_id] = 1 - strategies[player_id]
            alt_payoff = self.get_payoff(player_id, alt_strategies)
            if alt_payoff > original_payoff:
                return False
        return True
