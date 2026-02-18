import random
import math

class FictitiousPlay:
    def __init__(self, game, max_iterations=5000, update_fraction=0.7):
        self.game = game
        self.max_iterations = max_iterations
        self.num_players = game.num_players
        self.graph = game.graph
        self.update_fraction = update_fraction
        self.neighbor_counts = {
            i: {v: 0 for v in self.graph.neighbors(i)} for i in range(self.num_players)
        }
        self.total_counts = {i: 0 for i in range(self.num_players)}
        self.current_strategies = {i: random.choice([0, 1]) for i in range(self.num_players)}

    def run(self):
        history = []
        no_change_counter = 0
        # can be lowered if the algorithm is deterministic
        required_stability = self.num_players * 5 
        
        # Pre-calculate the convenience threshold
        # If the probability of being covered exceeds this threshold, it is better to play 0.
        threshold = (self.game.alpha - self.game.c) / self.game.alpha

        for t in range(self.max_iterations):
            current_set_size = sum(self.current_strategies.values())
            history.append(current_set_size)
            
            changed = False
            
            # Batch node selection
            num_to_update = max(1, int(self.num_players * self.update_fraction))
            nodes_to_update = random.sample(range(self.num_players), num_to_update)
            
            # Phase 1: Belief Update
            # In Asynchronous FP, beliefs are often updated only when activated,
            # or updated globally. Here we keep your logic:
            # we update counts only for nodes that "think" in this turn.
            for i in nodes_to_update:
                self.total_counts[i] += 1
                for v in self.graph.neighbors(i):
                    if self.current_strategies[v] == 1:
                        self.neighbor_counts[i][v] += 1

            # Phase 2: Calculate Best Response
            for i in nodes_to_update:
                neighbors = list(self.graph.neighbors(i))
                old_strategy = self.current_strategies[i]
                new_strategy = old_strategy

                if not neighbors:
                    # If I have no neighbors, I cannot be covered -> I must play 1
                    new_strategy = 1
                else:
                    # Calculate empirical probabilities that each neighbor plays 1
                    probs = []
                    for v in neighbors:
                        if self.total_counts[i] > 0:
                            p = self.neighbor_counts[i][v] / self.total_counts[i]
                        else:
                            p = 0.5 # Uniform prior if first iteration
                        probs.append(p)
                    
                    # Calculate Joint Probability (PRODUCT, not average)
                    prob_all_neighbors_1 = math.prod(probs)
                    
                    
                    # --- DETERMINISTIC BEST RESPONSE ---
                    # If the probability that everyone covers me is high, I risk and play 0.
                    # Otherwise, I protect myself and play 1.
                    if prob_all_neighbors_1 > threshold:
                        new_strategy = 0
                    else:
                        new_strategy = 1
                    """
                    # --- STOCHASTIC BEST RESPONSE ---
                    if random.random() <= prob_all_neighbors_1:
                        new_strategy = 0
                    else:
                        new_strategy = 1
                    """

                # Apply the change
                if new_strategy != old_strategy:
                    self.current_strategies[i] = new_strategy
                    changed = True
            
            # Convergence check
            if not changed:
                no_change_counter += 1
            else:
                no_change_counter = 0
                
            if no_change_counter >= required_stability:
                print(f"Convergence detected at step {t}.")
                history.extend([history[-1]] * (self.max_iterations - t - 1))
                break
        
        print(f"Final Strategies (Size S: {sum(self.current_strategies.values())})")
        return self.current_strategies, self.game.is_nash_equilibrium(self.current_strategies), history