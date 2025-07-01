import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Circle, Rectangle
import random
import time

# Configuration matplotlib
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (12, 8)


class PetriNet:
    """
    Classe pour modéliser et simuler un réseau de Pétri pour la gestion des colis
    """

    def __init__(self):
        # Définition des places
        self.places = {
            'p1': 'Colis_Attente_Reception',
            'p2': 'Colis_En_Verification',
            'p3': 'Colis_Stocke',
            'p4': 'Colis_En_Preparation',
            'p5': 'Colis_Pret_Expedition',
            'p6': 'Colis_Expedie',
            'p7': 'Colis_Refuse',
            'p8': 'Agents_Libres',
            'p9': 'Espaces_Stockage_Disponibles',
            'p10': 'Vehicules_Expedition'
        }

        # Définition des transitions
        self.transitions = {
            't1': 'Reception_Colis',
            't2': 'Verification_Conforme',
            't3': 'Verification_Non_Conforme',
            't4': 'Mise_En_Stock',
            't5': 'Preparation_Expedition',
            't6': 'Chargement_Vehicule',
            't7': 'Expedition',
            't8': 'Liberer_Agent',
            't9': 'Liberer_Espace'
        }

        # Matrice d'incidence
        self.incidence_matrix = np.array([
            [-1, 0, 0, 0, 0, 0, 0, 0, 0],  # p1
            [1, -1, -1, 0, 0, 0, 0, 0, 0],  # p2
            [0, 1, 0, -1, 0, 0, 0, 0, 0],  # p3
            [0, 0, 0, 1, -1, 0, 0, 0, 0],  # p4
            [0, 0, 0, 0, 1, -1, 0, 0, 0],  # p5
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # p6
            [0, 0, 1, 0, 0, 0, 0, 0, 0],  # p7
            [-1, 0, 0, 0, 0, 0, 0, 1, 0],  # p8
            [0, 0, 0, -1, 0, 0, 0, 0, 1],  # p9
            [0, 0, 0, 0, 0, -1, 0, 0, 0]  # p10
        ])

        # Marquage initial
        self.initial_marking = np.array([5, 0, 0, 0, 0, 0, 0, 3, 10, 2])
        self.current_marking = self.initial_marking.copy()

        # Durées des transitions (en minutes)
        self.transition_times = {
            't1': 2,  # Reception
            't2': 5,  # Verification conforme
            't3': 3,  # Verification non conforme
            't4': 1,  # Stockage
            't5': 10,  # Preparation
            't6': 15,  # Chargement
            't7': 60,  # Expedition
            't8': 1,  # Liberer agent
            't9': 1  # Liberer espace
        }

        # Historique
        self.marking_history = [self.current_marking.copy()]
        self.transition_history = []
        self.time_history = [0]

    def is_enabled(self, transition_idx):
        """Vérifie si une transition est franchissable"""
        pre_conditions = self.incidence_matrix[:, transition_idx] < 0
        required_tokens = -self.incidence_matrix[pre_conditions, transition_idx]
        available_tokens = self.current_marking[pre_conditions]
        return np.all(available_tokens >= required_tokens)

    def fire_transition(self, transition_idx):
        """Tire une transition si elle est franchissable"""
        if self.is_enabled(transition_idx):
            self.current_marking += self.incidence_matrix[:, transition_idx]
            transition_name = list(self.transitions.keys())[transition_idx]
            self.transition_history.append(transition_name)
            return True
        return False

    def get_enabled_transitions(self):
        """Retourne la liste des transitions franchissables"""
        return [i for i in range(len(self.transitions)) if self.is_enabled(i)]

    def simulate_step(self, current_time):
        """Simule une étape de la simulation"""
        enabled_transitions = self.get_enabled_transitions()

        if not enabled_transitions:
            return current_time + 1

        selected_transition = random.choice(enabled_transitions)

        if self.fire_transition(selected_transition):
            transition_name = list(self.transitions.keys())[selected_transition]
            duration = self.transition_times[transition_name]
            new_time = current_time + duration

            self.marking_history.append(self.current_marking.copy())
            self.time_history.append(new_time)

            return new_time

        return current_time + 1

    def simulate(self, max_time=480, max_steps=1000):
        """Simule le réseau de Pétri"""
        current_time = 0
        step = 0

        print("=== Début de la simulation ===")
        print(f"Marquage initial: {self.current_marking}")

        while current_time < max_time and step < max_steps:
            old_time = current_time
            current_time = self.simulate_step(current_time)

            if current_time > old_time:
                step += 1
                if step % 50 == 0:
                    print(f"Étape {step}, Temps: {current_time:.1f}min")
                    print(f"Marquage actuel: {self.current_marking}")

        print("=== Fin de la simulation ===")
        print(f"Nombre d'étapes: {step}")
        print(f"Temps final: {current_time:.1f} minutes")
        print(f"Marquage final: {self.current_marking}")

    def plot_petri_net(self):
        """Visualise la structure du réseau de Pétri"""
        plt.figure(figsize=(16, 12))

        # Positions des éléments
        pos = {
            'p1': (1, 8), 'p2': (3, 8), 'p3': (5, 8), 'p4': (7, 8),
            'p5': (9, 8), 'p6': (11, 6), 'p7': (3, 6),
            'p8': (1, 4), 'p9': (5, 4), 'p10': (9, 4),
            't1': (2, 8), 't2': (4, 8), 't3': (3, 7), 't4': (6, 8),
            't5': (8, 8), 't6': (10, 8), 't7': (11, 5),
            't8': (2, 4), 't9': (6, 4)
        }

        # Création du graphe
        G = nx.DiGraph()

        # Ajout des noeuds
        for node in pos.keys():
            G.add_node(node)

        # Ajout des arcs
        arcs = [
            ('p1', 't1'), ('t1', 'p2'), ('t1', 'p8'),
            ('p2', 't2'), ('t2', 'p3'),
            ('p2', 't3'), ('t3', 'p7'),
            ('p3', 't4'), ('t4', 'p4'), ('t4', 'p9'),
            ('p4', 't5'), ('t5', 'p5'),
            ('p5', 't6'), ('t6', 'p6'), ('t6', 'p10'),
            ('p6', 't7'),
            ('p8', 't8'), ('p9', 't9')
        ]

        for arc in arcs:
            G.add_edge(*arc)

        # Dessin du graphe
        nx.draw_networkx_nodes(G, pos, nodelist=self.places.keys(),
                               node_shape='o', node_color='lightblue', node_size=2000)
        nx.draw_networkx_nodes(G, pos, nodelist=self.transitions.keys(),
                               node_shape='s', node_color='lightgreen', node_size=1000)

        nx.draw_networkx_edges(G, pos, edgelist=arcs, arrowsize=20)

        # Étiquettes
        labels = {**self.places, **self.transitions}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')

        # Jetons
        for i, (place, coords) in enumerate(pos.items()):
            if place.startswith('p'):
                tokens = self.current_marking[int(place[1:]) - 1]
                if tokens > 0:
                    plt.text(coords[0], coords[1] - 0.3, f'•{tokens}',
                             ha='center', va='center', color='red', fontsize=12)

        plt.title("Réseau de Pétri - Gestion des Colis", fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def plot_results(self):
        """Affiche les résultats de la simulation"""
        if len(self.marking_history) < 2:
            print("Pas assez de données pour visualiser")
            return

        # Convertir l'historique en listes séparées
        time_points = self.time_history
        markings = np.array(self.marking_history)

        plt.figure(figsize=(16, 10))

        # Évolution des états des colis
        plt.subplot(2, 2, 1)
        colis_indices = [0, 1, 2, 3, 4, 5, 6]  # p1 à p7
        for i in colis_indices:
            plt.plot(time_points, markings[:, i], label=self.places[f'p{i + 1}'])
        plt.title("Évolution des états des colis")
        plt.xlabel("Temps (min)")
        plt.ylabel("Nombre de colis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Utilisation des ressources
        plt.subplot(2, 2, 2)
        resource_indices = [7, 8, 9]  # p8 à p10
        for i in resource_indices:
            plt.plot(time_points, markings[:, i], label=self.places[f'p{i + 1}'])
        plt.title("Utilisation des ressources")
        plt.xlabel("Temps (min)")
        plt.ylabel("Quantité disponible")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Débit de traitement
        plt.subplot(2, 2, 3)
        expedited = np.diff(markings[:, 5], prepend=0).cumsum()  # p6
        refused = np.diff(markings[:, 6], prepend=0).cumsum()  # p7
        plt.plot(time_points, expedited, label="Colis expédiés")
        plt.plot(time_points, refused, label="Colis refusés")
        plt.title("Débit cumulé")
        plt.xlabel("Temps (min)")
        plt.ylabel("Nombre de colis")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Taux d'utilisation
        plt.subplot(2, 2, 4)
        total_agents = self.initial_marking[7]
        utilization = (total_agents - markings[:, 7]) / total_agents * 100
        plt.plot(time_points, utilization, color='purple')
        plt.title("Taux d'utilisation des agents")
        plt.xlabel("Temps (min)")
        plt.ylabel("Utilisation (%)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Génère un rapport textuel de la simulation"""
        if len(self.marking_history) < 2:
            print("Simulation non effectuée")
            return

        print("\n" + "=" * 60)
        print("           RAPPORT DE SIMULATION")
        print("=" * 60)

        # Statistiques générales
        print(f"Durée totale de simulation: {self.time_history[-1]:.1f} minutes")
        print(f"Nombre d'étapes simulées: {len(self.marking_history)}")
        print(f"Nombre de transitions tirées: {len(self.transition_history)}")

        # Performance des colis
        final_expedited = self.marking_history[-1][5]  # p6
        final_refused = self.marking_history[-1][6]  # p7
        total_processed = final_expedited + final_refused

        print(f"\nPERFORMANCE:")
        print(f"- Colis expédiés: {final_expedited}")
        print(f"- Colis refusés: {final_refused}")
        print(f"- Total traités: {total_processed}")
        if total_processed > 0:
            print(f"- Taux de réussite: {final_expedited / total_processed * 100:.1f}%")

        # Utilisation des ressources
        initial_agents = self.initial_marking[7]
        final_agents = self.marking_history[-1][7]
        agents_used = initial_agents - final_agents

        print(f"\nRESSOURCES:")
        print(f"- Agents initiaux: {initial_agents}")
        print(f"- Agents libres finaux: {final_agents}")
        print(f"- Agents utilisés: {agents_used}")
        print(f"- Taux d'utilisation: {agents_used / initial_agents * 100:.1f}%")

        # Transitions les plus fréquentes
        transition_counts = {}
        for trans in self.transition_history:
            transition_counts[trans] = transition_counts.get(trans, 0) + 1

        print(f"\nTRANSITIONS LES PLUS FRÉQUENTES:")
        sorted_transitions = sorted(transition_counts.items(), key=lambda x: x[1], reverse=True)
        for trans, count in sorted_transitions[:5]:
            print(f"- {self.transitions[trans]}: {count} fois")

        print("=" * 60)


def main():
    """Fonction principale"""
    # Initialisation
    random.seed(42)

    # Création du réseau
    pn = PetriNet()

    # Simulation
    pn.simulate(max_time=240)  # 4 heures

    # Visualisation
    pn.plot_petri_net()
    pn.plot_results()

    # Rapport
    pn.generate_report()


if __name__ == "__main__":
    main()