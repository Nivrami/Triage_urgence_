"""
Cost Calculator - Calcul des coûts API
"""

from typing import Dict


class CostCalculator:
    """Calcule les coûts d'utilisation des APIs."""

    # Tarifs Mistral (€ par 1M tokens)
    MISTRAL_PRICING = {
        "mistral-small-latest": {
            "input": 1.00,  # 1€ / 1M tokens input
            "output": 3.00,  # 3€ / 1M tokens output
        },
        "mistral-medium-latest": {"input": 2.70, "output": 8.10},
        "mistral-large-latest": {"input": 4.00, "output": 12.00},
    }

    # Tarifs SentenceTransformers (gratuit, mais on simule)
    EMBEDDING_PRICING = {
        "sentence-transformers": {"input": 0.10, "output": 0.00}  # 0.10€ / 1M tokens (simulé)
    }

    def __init__(self):
        pass

    def calculate_mistral_cost(
        self, model: str, tokens_input: int, tokens_output: int
    ) -> Dict[str, float]:
        """Calcule coût Mistral."""
        pricing = self.MISTRAL_PRICING.get(model, self.MISTRAL_PRICING["mistral-small-latest"])

        cost_input = (tokens_input / 1_000_000) * pricing["input"]
        cost_output = (tokens_output / 1_000_000) * pricing["output"]
        cost_total = cost_input + cost_output

        return {
            "cost_input": cost_input,
            "cost_output": cost_output,
            "cost_total": cost_total,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
        }

    def calculate_embedding_cost(
        self, num_embeddings: int, avg_tokens_per_embedding: int = 50
    ) -> Dict[str, float]:
        """Calcule coût embeddings (simulé)."""
        total_tokens = num_embeddings * avg_tokens_per_embedding
        pricing = self.EMBEDDING_PRICING["sentence-transformers"]

        cost = (total_tokens / 1_000_000) * pricing["input"]

        return {"cost_total": cost, "num_embeddings": num_embeddings, "total_tokens": total_tokens}

    def calculate_total_cost(self, api_calls: list) -> Dict:
        """Calcule coût total de tous les appels."""
        mistral_calls = [c for c in api_calls if c.get("service") == "mistral"]
        embedding_calls = [c for c in api_calls if c.get("service") == "embeddings"]

        # Coûts Mistral
        mistral_cost = 0
        mistral_tokens_in = 0
        mistral_tokens_out = 0

        for call in mistral_calls:
            cost = self.calculate_mistral_cost(
                call.get("model", "mistral-small-latest"),
                call["tokens_input"],
                call["tokens_output"],
            )
            mistral_cost += cost["cost_total"]
            mistral_tokens_in += call["tokens_input"]
            mistral_tokens_out += call["tokens_output"]

        # Coûts Embeddings
        embedding_cost = 0
        total_embeddings = len(embedding_calls)

        if embedding_calls:
            avg_tokens = sum(c["tokens_input"] for c in embedding_calls) / len(embedding_calls)
            cost = self.calculate_embedding_cost(total_embeddings, int(avg_tokens))
            embedding_cost = cost["cost_total"]

        total_cost = mistral_cost + embedding_cost

        return {
            "total_cost": total_cost,
            "mistral": {
                "cost": mistral_cost,
                "calls": len(mistral_calls),
                "tokens_input": mistral_tokens_in,
                "tokens_output": mistral_tokens_out,
            },
            "embeddings": {"cost": embedding_cost, "calls": total_embeddings},
            "breakdown": {
                "mistral_pct": (mistral_cost / total_cost * 100) if total_cost > 0 else 0,
                "embeddings_pct": (embedding_cost / total_cost * 100) if total_cost > 0 else 0,
            },
        }

    def estimate_monthly_cost(self, current_cost: float, days_elapsed: int) -> float:
        """Estime coût mensuel basé sur utilisation actuelle."""
        if days_elapsed == 0:
            return 0

        daily_avg = current_cost / days_elapsed
        return daily_avg * 30

    def format_cost(self, cost: float) -> str:
        """Formate coût en format lisible."""
        if cost < 0.01:
            # Convertir en centimes pour plus de clarté
            centimes = cost * 100
            if centimes < 0.01:
                return f"{centimes:.3f}Millièmes de centime€"  # Millièmes de centime
            return f"{centimes:.2f}centimes"  # Centimes
        elif cost < 1.0:
            return f"{cost*100:.1f}centimes"  # Centimes
        return f"{cost:.2f}euro"  # Euros


# Instance globale
_calculator = None


def get_calculator() -> CostCalculator:
    """Récupère l'instance globale du calculateur."""
    global _calculator
    if _calculator is None:
        _calculator = CostCalculator()
    return _calculator
