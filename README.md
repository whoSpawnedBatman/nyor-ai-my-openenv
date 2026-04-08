---
title: Pharma B2B Quotation
emoji: 🌍
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---
# Nyor AI: B2B Pharma Quotation Hub

OpenEnv Compatible Benchmark | License: MIT

## Environment Description

Nyor AI provides an OpenEnv-compatible benchmark environment that simulates a professional Pharmaceutical B2B Quotation workflow. Pharmaceutical distribution involves a complex supply chain, and this environment provides a standardized interface to evaluate AI agents attempting to automate the quotation process. 

The environment tests an agent's ability to handle:
- Sourcing: Finding the right manufacturer for generic requirements.
- Supplier Verification: Confirming inventory availability and buy rates.
- Margin Calculation: Ensuring competitive pricing while maintaining required profitability thresholds.

The environment utilizes a goal-based evaluation mechanism across multiple tasks to benchmark an AI agent's decision-making over sequential steps.

## Setup Instructions

1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configuration
Create a `.env` file in the root directory and configure your LLM details:
```env
OPENAI_API_KEY=your_api_key_here
API_BASE_URL=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

3. Run Benchmark Inference
Execute the inference script to evaluate the baseline LLM agent:
```bash
python inference.py
```

4. Start OpenEnv Server
Launch the FastAPI server to expose OpenEnv-compatible HTTP endpoints:
```bash
python app.py
```

## Action Space

The environment uses a discrete, text-based action space. The agent must return exactly one of the following strings per step:

- `search_brands:<generic_name>`: Search for available brands matching a generic medicine name.
- `select_brand:<brand_name>`: Select a specific manufacturer brand.
- `select_supplier:<supplier_name>`: Choose a supplier for the selected brand.
- `request_confirmation`: Request confirmation of stock and buy rates from the selected supplier.
- `calculate_price:<number>`: Propose a final sell price.
- `finalize`: Finalize and submit the quotation.

## Observation Space

The environment provides a structured representation of the current state at each step containing the following fields:

- `order`: Details of the current request including generic name, strength, dosage, quantity, and specific brand policies.
- `selected_brand`: The brand currently selected by the agent.
- `selected_supplier`: The supplier currently selected by the agent.
- `calculated_price`: The currently proposed sell price.
- `supplier_confirmed`: Boolean indicating whether the selected supplier has confirmed stock and pricing.
- `action_history`: A list of past actions taken during the current episode.
- `score`: The current normalized cumulative reward score.

## Tasks

- `quotation`: Full sourcing-to-finalize workflow. Evaluates accuracy and margin constraints.
- `brand-selection`: Optimized brand picking evaluating correct brand/supplier pairings.
- `margin-check`: Mathematical selling price calculation based on minimum margin constraints.
