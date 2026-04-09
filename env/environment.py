import random
from typing import Tuple, Dict, Any, Optional
from env.models import Order, EnvironmentState, Quotation
from env.data import get_products_by_generic
from duckduckgo_search import DDGS

# ─── Task Registry ────────────────────────────────────────────────────────────
TASKS = {
    "quotation": {
        "description": "Full B2B quotation: search brand, select brand+supplier, confirm, calculate price, finalize.",
        "max_reward": 14.5,   # sum of all positive step rewards (0.5+1+1+1+1+10)
        "min_reward": -15.0,
    },
    "brand-selection": {
        "description": "Optimized brand selection: pick correct brand+supplier for Pharma distribution.",
        "max_reward": 3.0,    # search(0) + brand(1) + supplier(1) + confirm(1)
        "min_reward": -3.0,
    },
    "margin-check": {
        "description": "B2B Profitability logic: calculate price meeting 8% margin requirement.",
        "max_reward": 11.0,   # brand(1)+supplier(1)+confirm(1)+calculateprice(1)+finalize(10) - but we pre-seed brand/supplier
        "min_reward": -7.0,
    },
}

def normalize_score(raw: float, task: str) -> float:
    """Clamp and normalise raw cumulative reward → strictly (0.0, 1.0)."""
    info = TASKS.get(task, TASKS["quotation"])
    mn, mx = info["min_reward"], info["max_reward"]
    if mx == mn:
        return 0.01
    score = (raw - mn) / (mx - mn)
    # Clamp to [0.01, 0.99] to ensure scores are strictly within (0, 1)
    return max(0.01, min(0.99, score))


class PharmaQuotationEnv:
    def __init__(self, task: str = "quotation"):
        self.task = task if task in TASKS else "quotation"
        self.state: Optional[EnvironmentState] = None
        self.min_margin = 0.08        # 8% minimum margin for B2B profitability
        self._cumulative_reward = 0.0

    # ── OpenEnv-compatible API ─────────────────────────────────────────────
    def reset(self) -> EnvironmentState:
        self._cumulative_reward = 0.0

        if self.task == "quotation":
            orders = [
                Order(generic_name="Pantoprazole", strength="40mg", dosage_form="tablet",
                      quantity=500, brand_policy="any valid brand"),
                Order(generic_name="Paracetamol", strength="650mg", dosage_form="tablet",
                      quantity=1000, brand_policy="specific brand only", target_brand="Dolo 650"),
            ]
            self.state = EnvironmentState(order=random.choice(orders))

        elif self.task == "brand-selection":
            # Fixed order – agent must select the cheapest valid brand+supplier
            self.state = EnvironmentState(
                order=Order(generic_name="Pantoprazole", strength="40mg",
                            dosage_form="tablet", quantity=500, brand_policy="any valid brand")
            )

        elif self.task == "margin-check":
            # Pre-seed brand and supplier; agent only needs to calculate price + finalize
            self.state = EnvironmentState(
                order=Order(generic_name="Pantoprazole", strength="40mg",
                            dosage_form="tablet", quantity=500, brand_policy="any valid brand"),
                selected_brand="Pan 40",
                selected_supplier="MediSupplies",
                supplier_confirmed=True,
            )

        self.state.shortlisted_products = get_products_by_generic(self.state.order.generic_name)
        self.state.search_results = []
        return self.state

    def step(self, action: str) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        Supported actions:
          search_brands:<generic_name>
          select_brand:<brand_name>
          select_supplier:<supplier_name>
          request_confirmation
          calculate_price:<number>
          finalize
        """
        action = action.strip()
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        if not self.state:
            return self.state, 0.0, True, {"error": "Environment not reset."}

        self.state.action_history.append(action)

        try:
            if action.startswith("select_brand:"):
                brand = action.split(":", 1)[1].strip()
                if (self.state.order.brand_policy == "specific brand only"
                        and brand.lower() != (self.state.order.target_brand or "").lower()):
                    reward = -1.0
                    info["error"] = "Selected incorrect brand for policy 'specific brand only'."
                else:
                    valid_brands = [p.brand_name.lower() for p in self.state.shortlisted_products]
                    if brand.lower() in valid_brands:
                        self.state.selected_brand = brand
                        reward = 1.0
                    else:
                        reward = -1.0
                        info["error"] = "Invalid brand selected for the given generic."

            elif action.startswith("select_supplier:"):
                supplier = action.split(":", 1)[1].strip()
                if not self.state.selected_brand:
                    reward = -1.0
                    info["error"] = "Must select a brand first."
                else:
                    valid_suppliers = [
                        p.supplier_name.lower()
                        for p in self.state.shortlisted_products
                        if p.brand_name.lower() == self.state.selected_brand.lower()
                    ]
                    if supplier.lower() in valid_suppliers:
                        self.state.selected_supplier = supplier
                        self.state.supplier_confirmed = False
                        reward = 1.0
                    else:
                        reward = -1.0
                        info["error"] = f"Supplier not available for {self.state.selected_brand}."

            elif action == "request_confirmation":
                if not self.state.selected_supplier:
                    reward = -0.5
                    info["error"] = "No supplier selected to confirm."
                else:
                    self.state.supplier_confirmed = True
                    reward = 1.0
                    info["message"] = "Supplier confirmed stock and buy rate."

            elif action.startswith("calculate_price:"):
                price_str = action.split(":", 1)[1].strip()
                try:
                    price = float(price_str)
                    self.state.calculated_price = price
                    reward = 1.0
                except ValueError:
                    reward = -1.0
                    info["error"] = "Invalid price format."

            elif action == "finalize":
                done = True
                if (not self.state.selected_brand
                        or not self.state.selected_supplier
                        or not self.state.calculated_price):
                    reward = -5.0
                    info["error"] = "Incomplete quotation."
                elif not self.state.supplier_confirmed:
                    reward = -2.0
                    info["error"] = "Finalized without supplier confirmation."
                else:
                    product = next(
                        (p for p in self.state.shortlisted_products
                         if p.brand_name.lower() == self.state.selected_brand.lower()
                         and p.supplier_name.lower() == self.state.selected_supplier.lower()),
                        None
                    )
                    if not product:
                        reward = -5.0
                        info["error"] = "Invalid final product configuration."
                    elif not product.is_valid_supplier:
                        reward = -5.0
                        info["error"] = "Selected a known invalid/unapproved supplier."
                    else:
                        margin = (self.state.calculated_price - product.buy_rate) / product.buy_rate
                        if margin < self.min_margin:
                            reward = -2.0
                            info["error"] = (
                                f"Margin {margin:.1%} is lower than required {self.min_margin:.1%}."
                            )
                        else:
                            valid_competing = [
                                p for p in self.state.shortlisted_products
                                if p.brand_name.lower() == self.state.selected_brand.lower()
                                and p.is_valid_supplier
                            ]
                            cheapest = min(valid_competing, key=lambda p: p.buy_rate)
                            if product.buy_rate > cheapest.buy_rate:
                                reward = 5.0
                                info["message"] = "Valid quotation, but cheaper valid supplier was available."
                            else:
                                reward = 10.0
                                info["message"] = "Perfect quotation generated!"

            elif action.startswith("search_brands:"):
                query = action.split(":", 1)[1].strip()
                try:
                    results = DDGS().text(f"{query} medicine brands in india", max_results=3)
                    search_str = "\n".join([r["body"] for r in results])
                    self.state.search_results.append(search_str)
                    reward = 0.5
                    info["message"] = "Search completed. Check state for results."
                except Exception as e:
                    reward = -1.0
                    info["error"] = f"Search failed: {str(e)}"

            else:
                reward = -1.0
                info["error"] = f"Unknown action: {action}"

        except Exception as e:
            reward = -1.0
            info["error"] = str(e)

        self._cumulative_reward += reward
        info["score"] = normalize_score(self._cumulative_reward, self.task)
        return self.state, reward, done, info

    # ── Grader helper ─────────────────────────────────────────────────────
    def score(self) -> float:
        """Return normalised final score in [0.0, 1.0]."""
        return normalize_score(self._cumulative_reward, self.task)

    def close(self) -> None:
        pass
