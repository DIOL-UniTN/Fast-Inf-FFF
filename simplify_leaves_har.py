import torch
import typer
import pickle
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from fastfeedforward import FFF
from matplotlib import pyplot as plt
# Load local modules
from fff_trainer import test
from fff_experiment_har import load_data


# Set numpy print precision to 2 decimal digits
np.set_printoptions(precision=2)
# Set device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {DEVICE}")


def get_dist(net, testloader):
    """
    Returns the distribution of samples throughout the tree.
    """

    y = []
    l = []
    with torch.no_grad():
        # Iterate over data
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs, leaves = net.forward(images, return_nodes=True)
            y.append(labels)
            l.append(leaves)
    y = torch.concat(y, 0)
    l = torch.concat(l, 0)
    return y, l


class FFFWrapper(torch.nn.Module):
    def __init__(self, fff):
        super(FFFWrapper, self).__init__()
        self._fff = fff
        self._fastinference = [None for i in range(2 ** (self._fff.fff.depth.item()))]

    def forward(self, x, return_nodes=False):
        """
        Override the forward method in order to log the data distribution.
        """
        x = x.view(len(x), -1)
        original_shape = x.shape
        batch_size = x.shape[0]
        last_node = torch.zeros(len(x))

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self._fff.fff.depth.item()):
            plane_coeffs = self._fff.fff.node_weights.index_select(dim=0, index=current_nodes)
            plane_offsets = self._fff.fff.node_biases.index_select(dim=0, index=current_nodes)
            plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))
            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets
            plane_choices = (plane_score.squeeze(-1) >= 0).long()

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform

        leaves = current_nodes - next_platform
        new_logits = torch.empty((batch_size, self._fff.fff.output_width), dtype=torch.float, device=x.device)
        last_node = leaves

        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            if self._fastinference[leaf_index] is not None:
                new_logits[i] = self._fastinference[leaf_index]
            else:
                logits = torch.matmul( x[i].unsqueeze(0), self._fff.fff.w1s[leaf_index])
                logits += self._fff.fff.b1s[leaf_index].unsqueeze(-2)
                activations = self._fff.fff.activation(logits)
                new_logits[i] = torch.matmul( activations, self._fff.fff.w2s[leaf_index]).squeeze(-2)

        if return_nodes:
            return new_logits.view(*original_shape[:-1], self._fff.fff.output_width), last_node
        return new_logits.view(*original_shape[:-1], self._fff.fff.output_width)


    def simplify_leaves(self, trainloader):
        y, leaves = (get_dist(self, trainloader))
        y = y.cpu().detach().numpy()
        outputs = y.max() + 1
        leaves = leaves.cpu().detach().numpy()

        n_simplifications = 0
        ratios = {}
        for l in np.unique(leaves):
            ratios[l] = torch.zeros(outputs)
            indices = leaves == l

            for i in range(outputs):
                ratios[l][i] = (np.sum(y[indices] == i) / np.sum(indices))

            argmax = np.argmax(ratios[l])
            if ratios[l][argmax] > 0.7:
                output = torch.zeros(outputs)
                output[argmax] = 1
                self._fastinference[l] = output
                n_simplifications += 1
                print(f"Leaf {l} has been replaced with {argmax}")
        print(self._fastinference)
        return n_simplifications



def test_splitting(inputs: int, l_w: int, outputs: int, d: int, norm_weight: float = None):
    # Load data
    trainloader, testloader, n = load_data()


    # Retrieve all the runs having "leaf_width" = "4" and "depth" = "4"
    if norm_weight is not None:
        filter_string = f'params.leaf_width = "{l_w}" and params.depth = "{d}" and params.norm_weight = "{norm_weight}"'
        runs = mlflow.search_runs(
            experiment_ids=["2"],
            filter_string=filter_string,
            output_format='pandas'
        )
    else:
        filter_string = f'params.leaf_width = "{l_w}" and params.depth = "{d}"'
        runs = mlflow.search_runs(
            experiment_ids=["2"],
            filter_string=filter_string,
            output_format='pandas'
        )

    print(f"{len(runs)} runs found with {filter_string}")

    for run_id in runs['run_id']:
        already_ran = mlflow.search_runs(
            experiment_ids=["3"],
            output_format='pandas'
        )

        if len(already_ran) > 0 and run_id in already_ran["params.starting_run"].values:
            print(f"Skipping {run_id}")
            continue

        try:
            with mlflow.start_run(experiment_id="3"):
                # Get the run with the current run_id
                run = mlflow.get_run(run_id)
                mlflow.log_param("starting_run", run_id)
                print(f"Testing run {run.info.run_id}")
                # Load the state dict from the model
                model_uri = f"runs:/{run.info.run_id}/model"
                model = mlflow.pytorch.load_model(model_uri)
                model.eval()
                model = FFFWrapper(model)

                # Test the model
                y, leaves = (get_dist(model, trainloader))
                y = y.cpu().detach().numpy()
                leaves = leaves.cpu().detach().numpy()

                # Testing time and accuracy 
                t = time()
                train_loss, train_acc = test(model, trainloader)
                test_loss, test_acc = test(model, testloader)
                t = time() - t
                print(f"Inference time: {t}s")
                print(f"[Before simplification] Train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
                print(f"[Before simplification] Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")

                mlflow.log_metric("train_loss_before", train_loss)
                mlflow.log_metric("test_loss_before", test_loss)
                mlflow.log_metric("train_acc_before", train_acc)
                mlflow.log_metric("test_acc_before", test_acc)
                mlflow.log_metric("inference_time_before", t)

                n_simplifications = model.simplify_leaves(trainloader)
                mlflow.log_metric("n_simplified_leaves", n_simplifications)

                t = time()
                train_loss, train_acc = test(model, trainloader)
                test_loss, test_acc = test(model, testloader)
                t = time() - t
                print(f"Inference time: {t}s")
                print(f"[After simplification] Train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
                print(f"[After simplification] Test loss: {test_loss:.4f}, test acc: {test_acc:.4f}")
                mlflow.log_metric("train_loss_after", train_loss)
                mlflow.log_metric("test_loss_after", test_loss)
                mlflow.log_metric("train_acc_after", train_acc)
                mlflow.log_metric("test_acc_after", test_acc)
                mlflow.log_metric("inference_time_after", t)

                pickle.dump(model, open("truncated_model.pkl", "wb"))
                mlflow.log_artifact("./truncated_model.pkl")
        except:
            print(f"Cannot run on {run_id}")


if __name__ == '__main__':
    typer.run(test_splitting)
