import pulp
import pandas as pd

def optimize_supervisor_allocation(df, total_agents=5):
    """
    Minimize expected escalation cost by allocating supervisors across issue types
    """

    # Aggregate by category
    summary = (
        df.groupby("Category")
        .agg(
            volume=("Category", "count"),
            escalation_rate=("Escalated", lambda x: (x == "Yes").mean())
        )
        .reset_index()
    )

    # Estimated cost per escalation (example business assumption)
    summary["cost"] = summary["escalation_rate"] * summary["volume"]

    # Optimization model
    model = pulp.LpProblem("Supervisor_Allocation", pulp.LpMinimize)

    # Decision variables: number of agents per category
    x = {
        row.Category: pulp.LpVariable(f"x_{row.Category}", lowBound=0, cat="Integer")
        for _, row in summary.iterrows()
    }

    # Objective: minimize expected escalation cost
    model += pulp.lpSum(summary.loc[i, "cost"] * x[row.Category]
                        for i, row in summary.iterrows())

    # Constraint: limited agents
    model += pulp.lpSum(x.values()) <= total_agents

    model.solve()

    summary["Allocated_Agents"] = summary["Category"].apply(lambda c: x[c].value())

    return summary
