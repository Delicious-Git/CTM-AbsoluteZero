import numpy as np

class NeuralTransferAdapter:
    """
    Adapter qui permet le transfert de connaissances entre domaines en modifiant
    les paramètres des tâches.
    """
    def __init__(self, domains):
        """
        Initialise l'adaptateur de transfert.
        
        Args:
            domains: Liste des domaines supportés
        """
        self.domains = domains
        self.domain_performance = {}
        self.correlation = 0.5  # Initialisation avec une corrélation modérée
    
    def transfer_parameters(self, current_domain, current_params, all_domains_perf):
        """
        Applique le transfert de connaissances entre domaines.
        
        Args:
            current_domain: Domaine de la tâche actuelle
            current_params: Paramètres de la tâche actuelle
            all_domains_perf: Métriques de performance de tous les domaines
            
        Returns:
            Paramètres de tâche modifiés avec transfert de connaissances
        """
        # Version simplifiée sans composants neuronaux
        if current_domain == 'maze' and 'qamnist' in all_domains_perf:
            # Transfert de connaissances visuelles vers la génération de labyrinthe
            qamnist_perf = all_domains_perf.get('qamnist', {}).get('avg', 0.5)
            if qamnist_perf > 0.7:
                current_params['params']['visual_patterns'] = True
                current_params['params']['pattern_complexity'] = qamnist_perf
        
        elif current_domain == 'quantum' and 'parity' in all_domains_perf:
            # Transfert de connaissances de parité vers les opérations quantiques
            parity_perf = all_domains_perf.get('parity', {}).get('avg', 0.5)
            if parity_perf > 0.6:
                current_params['params']['algorithm'] = 'qft' if parity_perf > 0.8 else 'grover'
        
        elif current_domain == 'rl' and 'sorting' in all_domains_perf:
            # Transfert de connaissances de tri vers la longueur des épisodes RL
            sorting_perf = all_domains_perf.get('sorting', {}).get('avg', 0.5)
            if sorting_perf > 0.7:
                # Épisodes plus longs pour les modèles de tri performants
                current_params['params']['max_episode_steps'] = int(200 + 300 * sorting_perf)
        
        # Mise à jour de la corrélation basée sur les métriques de performance
        self._update_correlation(all_domains_perf)
        
        return current_params
    
    def _update_correlation(self, all_domains_perf):
        """
        Calcule la métrique de corrélation inter-domaines.
        
        Args:
            all_domains_perf: Métriques de performance de tous les domaines
        """
        if len(all_domains_perf) > 1:
            # Proxy de corrélation simple: variance dans les performances entre domaines
            means = [data['avg'] for data in all_domains_perf.values()]
            variance = np.var(means) if len(means) > 1 else 0
            # Une variance plus faible indique des performances plus homogènes entre domaines
            self.correlation = max(0.1, 1.0 - variance)
    
    def get_correlation(self):
        """
        Récupère la métrique de corrélation inter-domaines actuelle.
        
        Returns:
            Score de corrélation entre 0 et 1
        """
        return self.correlation
    
    def get_transfer_explanation(self, from_domain, to_domain, params_before, params_after):
        """
        Génère une explication du transfert effectué entre domaines.
        
        Args:
            from_domain: Domaine source
            to_domain: Domaine cible
            params_before: Paramètres avant transfert
            params_after: Paramètres après transfert
            
        Returns:
            Explication textuelle du transfert
        """
        explanation = f"Transfert de {from_domain} vers {to_domain}:\n"
        
        # Trouver les différences
        if params_before == params_after:
            return f"Aucun transfert appliqué de {from_domain} vers {to_domain}"
        
        # Analyser les modifications
        for key, after_val in params_after.get('params', {}).items():
            before_val = params_before.get('params', {}).get(key)
            if before_val != after_val:
                explanation += f"  - {key}: {before_val} -> {after_val}\n"
        
        return explanation