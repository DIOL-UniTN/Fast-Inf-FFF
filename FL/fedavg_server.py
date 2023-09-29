import flwr as fl
import numpy as np
from typing import List, Tuple
from flwr.common import Metrics


fname = "".join(np.random.choice([chr(i) for i in range(97, 123)], size=8)) + ".log"


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):

        print("AGGREGATE CALLED")

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"round-{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics

# Create strategy and run server
strategy = SaveModelStrategy(
    # (same arguments as FedAvg here)
)


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    metrics = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    with open(fname, "a") as f:
        f.write(f"{metrics}\n")
        print(metrics)

    # Aggregate and return custom metric (weighted average)
    return metrics


# Define strategy
strategy = fl.server.strategy.FedAvg(min_fit_clients=10,evaluate_metrics_aggregation_fn=weighted_average)

print("Starting server")

with open("log.txt", "w") as f:
    f.write(f"accuracy\n")
# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=300),
    strategy=strategy,
)
