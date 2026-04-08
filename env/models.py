from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Product:
    generic_name: str
    brand_name: str
    composition: str
    supplier_name: str
    buy_rate: float
    mrp: float
    manufacturer: str
    pack_size: str
    stock: int
    is_valid_supplier: bool = True


@dataclass
class Order:
    generic_name: str
    strength: str
    dosage_form: str
    quantity: int
    brand_policy: str          # "any valid brand" | "specific brand only"
    target_brand: Optional[str] = None


@dataclass
class Quotation:
    brand_name: str
    supplier_name: str
    quote_price: float
    margin: float


@dataclass
class EnvironmentState:
    order: Optional[Order] = None
    selected_brand: Optional[str] = None
    selected_supplier: Optional[str] = None
    calculated_price: Optional[float] = None
    supplier_confirmed: bool = False
    shortlisted_products: List[Product] = field(default_factory=list)
    action_history: List[str] = field(default_factory=list)
    search_results: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        search_text = ""
        if self.search_results:
            search_text = (
                "\nSearch Results (Web Data):\n  "
                + "\n  ".join(self.search_results)
                + "\n"
            )
        past = "\n  ".join(self.action_history) if self.action_history else "None"
        return (
            f"Order Details:\n"
            f"  Generic Name: {self.order.generic_name if self.order else 'None'}\n"
            f"  Strength:     {self.order.strength if self.order else 'None'}\n"
            f"  Quantity:     {self.order.quantity if self.order else 'None'}\n"
            f"  Brand Policy: {self.order.brand_policy if self.order else 'None'}\n"
            f"\nCurrent Selections:\n"
            f"  Selected Brand:    {self.selected_brand or 'None'}\n"
            f"  Selected Supplier: {self.selected_supplier or 'None'}\n"
            f"  Quote Price:       {self.calculated_price or 'None'}\n"
            f"  Supplier Confirmed:{self.supplier_confirmed}\n"
            f"\nPast Actions Taken:\n  {past}\n"
            f"{search_text}"
        )
