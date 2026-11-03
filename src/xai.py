
from sklearn.tree import _tree


def explain_decision(model, flight):
    """Return the decision rules followed by one flight."""

    if len(flight) != 1:
        raise ValueError("Provide exactly one flight.")

    feature_names = list(model.feature_names_in_)

    node_indicator = model.decision_path(flight)

    node_indices = node_indicator.indices[
        node_indicator.indptr[0]:node_indicator.indptr[1]
    ]

    decisions = []

    for node_id in node_indices:
        feature_index = model.tree_.feature[node_id]

        if feature_index == _tree.TREE_UNDEFINED:
            continue

        name = feature_names[feature_index]
        value = float(flight.iloc[0, feature_index])
        threshold = float(model.tree_.threshold[node_id])

        operator = "<=" if value <= threshold else ">"

        decisions.append({
            "feature": name,
            "value": value,
            "operator": operator,
            "threshold": threshold,
        })

    return decisions
