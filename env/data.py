from env.models import Product

MOCK_CATALOG = [
    # ── Pantoprazole ───────────────────────────────────────────────────────
    Product(
        generic_name="Pantoprazole",
        brand_name="Pantocid",
        composition="Pantoprazole 40mg",
        supplier_name="PharmaCorp",
        buy_rate=45.0,
        mrp=65.0,
        manufacturer="Sun Pharma",
        pack_size="10 tabs",
        stock=1000,
        is_valid_supplier=True,
    ),
    Product(
        generic_name="Pantoprazole",
        brand_name="Pan 40",
        composition="Pantoprazole 40mg",
        supplier_name="MediSupplies",
        buy_rate=40.0,   # cheapest valid supplier
        mrp=60.0,
        manufacturer="Alkem",
        pack_size="10 tabs",
        stock=500,
        is_valid_supplier=True,
    ),
    Product(
        generic_name="Pantoprazole",
        brand_name="Pan 40",
        composition="Pantoprazole 40mg",
        supplier_name="ShadySupplier",
        buy_rate=35.0,   # cheapest but INVALID
        mrp=60.0,
        manufacturer="Alkem",
        pack_size="10 tabs",
        stock=500,
        is_valid_supplier=False,
    ),
    # ── Paracetamol ────────────────────────────────────────────────────────
    Product(
        generic_name="Paracetamol",
        brand_name="Dolo 650",
        composition="Paracetamol 650mg",
        supplier_name="MediSupplies",
        buy_rate=20.0,
        mrp=30.0,
        manufacturer="Micro Labs",
        pack_size="15 tabs",
        stock=2000,
        is_valid_supplier=True,
    ),
    Product(
        generic_name="Paracetamol",
        brand_name="Calpol 650",
        composition="Paracetamol 650mg",
        supplier_name="PharmaCorp",
        buy_rate=22.0,
        mrp=32.0,
        manufacturer="GSK",
        pack_size="15 tabs",
        stock=800,
        is_valid_supplier=True,
    ),
]


def get_products_by_generic(generic_name: str):
    return [p for p in MOCK_CATALOG if p.generic_name.lower() == generic_name.lower()]
