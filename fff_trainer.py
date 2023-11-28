import math
import torch
import typer
import numpy as np
from fastfeedforward import FFF


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "mps"
print(f"Training on {DEVICE}")


def train(net, trainloader, epochs, norm_weight=0.0):
    """Train the network on the training set."""
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

    # Train the network for the given number of epochs
    for _ in range(epochs):
        # Iterate over data
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            l2loss = 0.0
            if hasattr(net, 'fff'):
                l2loss += norm_weight * net.fff.w1s.pow(2).sum()
                l2loss += norm_weight * net.fff.w2s.pow(2).sum()
            else:
                l2loss = 0.0
                if norm_weight != 0:
                    for x in net.parameters():
                        l2loss += x.pow(2).sum()
            loss += norm_weight * l2loss
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    # Train the network for the given number of epochs
    with torch.no_grad():
        # Iterate over data
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class Net(torch.nn.Module):
    def __init__(self, input_width, leaf_width, output_width, depth, dropout, region_leak):
        super(Net, self).__init__()
        self.fff = FFF(input_width, leaf_width, output_width, depth, torch.nn.ReLU(), dropout, train_hardened=False, region_leak=region_leak)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = self.fff(x)
        x = torch.nn.functional.softmax(x, -1)
        return x

    def parameters(self):
        return self.fff.parameters()


class FF(torch.nn.Module):
    def __init__(self, input_width, layer_width, output_width):
        super(FF, self).__init__()
        self.fc1 = torch.nn.Linear(input_width, layer_width)
        self.fc2 = torch.nn.Linear(layer_width, output_width)

    def forward(self, x):
        x = x.view(len(x), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.softmax(self.fc2(x), -1)
        return x

    def parameters(self):
        return [*self.fc1.parameters(), *self.fc2.parameters()]


class SelfAttention(torch.nn.Module):
    def __init__(self, latent_size, leaf_width, depth):
        super(SelfAttention, self).__init__()
        self._latent_size = latent_size
        self.q_net = FFF(latent_size, leaf_width, latent_size, depth, torch.nn.ReLU(), dropout=0, train_hardened=False, region_leak=0)
        self.k_net = FFF(latent_size, leaf_width, latent_size, depth, torch.nn.ReLU(), dropout=0, train_hardened=False, region_leak=0)
        self.v_net = FFF(latent_size, leaf_width, latent_size, depth, torch.nn.ReLU(), dropout=0, train_hardened=False, region_leak=0)

    def forward(self, x):
        q = self.q_net(x)
        k = self.k_net(x)
        v = self.v_net(x)

        attention_map = torch.nn.functional.softmax(q @ torch.transpose(k, 1, 2) / (self._latent_size ** 0.5), -1)
        y = attention_map @ v
        return y


class Transformer(torch.nn.Module):
    def __init__(self, latent_size, leaf_width, depth):
        super(Transformer, self).__init__()
        self._latent_size = latent_size
        self.sa = SelfAttention(latent_size, leaf_width, depth)
        self.fff = FFF(latent_size, leaf_width, latent_size, depth, torch.nn.GeLU(), dropout=0, train_hardened=False, region_leak=0)

    def forward(self, x):
        # Normalize
        x = (x - x.mean(-1, keepdim=True))/(x.std(-1, keepdim=True) + 1e-5) ** 0.5

        x = x + self.sa(x)
        x = x + self.fff(x)
        return x


class ViTFFF(torch.nn.Module):
    def __init__(self, patch_size, latent_size, leaf_width, output_width, depth):
        super(ViTFFF, self).__init__()
        self._latent_size = latent_size
        self._leaf_width = leaf_width
        self._output_width = output_width
        self._depth = depth
        self._patch_size = patch_size
        self.tokenizer = FFF(np.product(patch_size), leaf_width, latent_size, depth, torch.nn.ReLU(), dropout=0, train_hardened=False, region_leak=0)
        self.transformer1 = Transformer(latent_size, leaf_width, depth)
        self.transformer2 = Transformer(latent_size, leaf_width, depth)
        self.output_layer = FFF(latent_size, leaf_width, output_width, depth, torch.nn.ReLU(), dropout=0, train_hardened=False, region_leak=0)

    def make_patches(self, img):
        batch_size, ch, h, w = img.shape
        h_patches = int(h / self._patch_size[1])
        w_patches = int(w / self._patch_size[2])

        patches = torch.zeros((batch_size, h_patches * w_patches, ch * self._patch_size[1] * self._patch_size[2]))

        for j in range(h_patches):
            for k in range(w_patches):
                patches[:, j * w_patches + k] = img[:, :, j*self._patch_size[1]:(j+1)*self._patch_size[1], k*self._patch_size[2]:(k+1)*self._patch_size[2]].reshape(patches.shape[0], -1)
        return patches

    def forward(self, imgs):
        x = self.make_patches(imgs)
        x = x.to(DEVICE)

        x = self.tokenizer(x)

        # Add positional embeddings
        for i in range(x.shape[-1]):
            if i % 2 == 0:
                fun = torch.sin
            else:
                fun = torch.cos
            x[..., i] += fun(torch.arange(x.shape[-2]) / (10000 ** (torch.arange(x.shape[-1] / self._latent_size)))).to(DEVICE)

        # Call transformer
        x = self.transformer1(x)
        x = self.transformer2(x)

        # To 2d
        x = x.mean(1)

        # Output mapping
        x = self.output_layer(x)
        return x


class ConvFFF(torch.nn.Module):
    def __init__(self, input_width: tuple, leaf_width: int, output_width: int, depth: int, activation=torch.nn.ReLU(), dropout: float=0.0, train_hardened: bool=False, region_leak: float=0.0, usage_mode: str = 'none', conv_size=5, n_channels=3, n_filters=32):
        super().__init__()
        assert isinstance(input_width, tuple)
        assert len(input_width) == 3
        self.input_width = input_width
        self.leaf_width = leaf_width
        self.output_width = output_width
        self.dropout = dropout
        self.activation = activation
        self.train_hardened = train_hardened
        self.region_leak = region_leak
        self.usage_mode = usage_mode
        self._n_filters = n_filters

        if depth < 0 or np.product(input_width) <= 0 or leaf_width <= 0 or output_width <= 0:
            raise ValueError("input/leaf/output widths and depth must be all positive integers")
        if dropout < 0 or dropout > 1:
            raise ValueError("dropout must be in the range [0, 1]")
        if region_leak < 0 or region_leak > 1:
            raise ValueError("region_leak must be in the range [0, 1]")
        if usage_mode not in ['hard', 'soft', 'none']:
            raise ValueError("usage_mode must be one of ['hard', 'soft', 'none']")

        self.depth = torch.nn.Parameter(torch.tensor(depth, dtype=torch.long), requires_grad=False)
        self.n_leaves = 2 ** depth
        self.n_nodes = 2 ** depth - 1

        l1_init_factor = 1.0 / math.sqrt(np.product(self.input_width))
        self.node_weights = torch.nn.Parameter(torch.empty((self.n_nodes, 1, n_channels, conv_size, conv_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.node_biases = torch.nn.Parameter(torch.empty((self.n_nodes, 1), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)

        l2_init_factor = 1.0 / math.sqrt(self.leaf_width)
        self.cw1s = torch.nn.Parameter(torch.empty((self.n_leaves, n_filters, n_channels, conv_size, conv_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.cw2s = torch.nn.Parameter(torch.empty((self.n_leaves, n_filters, n_filters, conv_size, conv_size), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w1s = torch.nn.Parameter(torch.empty((self.n_leaves, n_filters, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.b1s = torch.nn.Parameter(torch.empty((self.n_leaves, leaf_width), dtype=torch.float).uniform_(-l1_init_factor, +l1_init_factor), requires_grad=True)
        self.w2s = torch.nn.Parameter(torch.empty((self.n_leaves, leaf_width, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)
        self.b2s = torch.nn.Parameter(torch.empty((self.n_leaves, output_width), dtype=torch.float).uniform_(-l2_init_factor, +l2_init_factor), requires_grad=True)

        self.leaf_dropout = torch.nn.Dropout(dropout)

        if usage_mode != 'none':
            self.node_usage = torch.nn.Parameter(torch.zeros((self.n_nodes,), dtype=torch.float), requires_grad=False)
            self.leaf_usage = torch.nn.Parameter(torch.zeros((self.n_leaves,), dtype=torch.float), requires_grad=False)

    def get_node_param_group(self) -> dict:
        """
        Returns the parameters of the nodes of this FFF, coupled with their usage tensor.

        Returns
        -------
        dict
            The parameters of the nodes of this FFF, coupled with their usage tensor.
            Will have the following keys:
                - "params": a list containing the node parameters
                - "usage": the node usage tensor
        """

        return {
            "params": [self.node_weights, self.node_biases],
            "usage": self.node_usage,
        }
    
    def get_leaf_param_group(self) -> dict:
        """
        Returns the parameters of the leaves of this FFF, coupled with their usage tensor.

        Returns
        -------
        dict
            The parameters of the leaves of this FFF, coupled with their usage tensor.
            Will have the following keys:
                - "params": a list containing the leaf parameters
                - "usage": the node usage tensor
        """
        
        return {
            "params": [self.w1s, self.b1s, self.w2s, self.b2s],
            "usage": self.leaf_usage,
        }

    def training_forward(self, x: torch.Tensor, return_entropies: bool=False, use_hard_decisions: bool=False):
        """
        Computes the forward pass of this FFF during training.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must have shape (..., input_width).
        return_entropies : bool, optional
            Whether to return the entropies of the decisions made at each node. Defaults to False.
            If True, the mean batch entropies for each node will be returned as a tensor of shape (n_nodes,).
        use_hard_decisions : bool, optional
            Whether to use hard decisions during the forward pass. Defaults to False.
            If True, the decisions will be rounded to the nearest integer. This will effectively make the FFF tree non-differentiable.

        Returns
        -------
        torch.Tensor
            The output tensor. Will have shape (..., output_width).
        torch.Tensor, optional
            The mean batch entropies for each node. Will be returned with shape (n_nodes,) if `return_entropies` is True.
            Will not be returned if `return_entropies` is False.

        Notes
        -----
        - The FFF tree is traversed from the root to the leaves.
            At each node, the input is multiplied by the node's weight matrix and added to the node's bias vector.
            The result is passed through a sigmoid function to obtain a probability.
            The probability is used to modify the mixture of the current batch of inputs.
            The modified mixture is passed to the next node.
            Finally, the outputs of all leaves are mixed together to obtain the final output.
        - If `use_hard_decisions` is True and `return_entropies` is True, the entropies will be computed before the decisions are rounded.
        - If self.training is False, region leaks and dropout will not be applied in this function.
        - Node usage, when tracked, is computed after node leaks have been applied (but is of course also applied when there is no node leaks).
        
        Raises
        ------
        ValueError
            - if `x` does not have shape (..., input_width)

        See Also
        --------
        `eval_forward()`

        """
        # x has shape (batch_size, input_width)
        original_shape = x.shape
        # x = x.view(-1, x.shape[-1])
        batch_size = x.shape[0]

        hard_decisions = use_hard_decisions or self.train_hardened
        current_mixture = torch.ones((batch_size, self.n_leaves), dtype=torch.float, device=x.device)
        entropies = None if not return_entropies else torch.zeros((batch_size, self.n_nodes), dtype=torch.float, device=x.device)

        if self.usage_mode != 'none' and self.depth.item() > 0:
            self.node_usage[0] += batch_size

        for current_depth in range(self.depth.item()):
            platform = torch.tensor(2 ** current_depth - 1, dtype=torch.long, device=x.device)
            next_platform = torch.tensor(2 ** (current_depth+1) - 1, dtype=torch.long, device=x.device)

            n_nodes = 2 ** current_depth
            current_weights = self.node_weights[platform:next_platform]    # (n_nodes, input_width)    
            current_biases = self.node_biases[platform:next_platform]    # (n_nodes, 1)

            # boundary_plane_coeff_scores = torch.matmul(x, current_weights.transpose(0, 1))        # (batch_size, n_nodes)
            boundary_plane_coeff_scores = torch.max(torch.nn.functional.conv2d(x, current_weights[platform:next_platform].squeeze(1), padding='same').view(batch_size, -1), -1).values.view(-1, 1)
            boundary_plane_logits = boundary_plane_coeff_scores + current_biases.transpose(0, 1)# (batch_size, n_nodes)
            boundary_effect = torch.sigmoid(boundary_plane_logits)                                # (batch_size, n_nodes)

            if self.region_leak > 0.0 and self.training:
                transpositions = torch.empty_like(boundary_effect).uniform_(0, 1)        # (batch_size, n_cuts)
                transpositions = transpositions < self.region_leak                        # (batch_size, n_cuts)
                boundary_effect = torch.abs(transpositions.float() - boundary_effect)     # (batch_size, n_cuts)

            not_boundary_effect = 1 - boundary_effect                                    # (batch_size, n_nodes)

            if return_entropies:
                platform_entropies = compute_entropy_safe(
                    boundary_effect, not_boundary_effect
                ) # (batch_size, n_nodes)
                entropies[:, platform:next_platform] = platform_entropies    # (batch_size, n_nodes)
                
            if hard_decisions:
                boundary_effect = torch.round(boundary_effect)                # (batch_size, n_nodes)
                not_boundary_effect = 1 - boundary_effect                    # (batch_size, n_nodes)
            
            mixture_modifier = torch.cat( # this cat-fu is to interleavingly combine the two tensors
                (not_boundary_effect.unsqueeze(-1), boundary_effect.unsqueeze(-1)),
                dim=-1
            ).flatten(start_dim=-2, end_dim=-1).unsqueeze(-1)                                                # (batch_size, n_nodes*2, 1)
            current_mixture = current_mixture.view(batch_size, 2 * n_nodes, self.n_leaves // (2 * n_nodes))    # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture.mul_(mixture_modifier)                                                            # (batch_size, 2*n_nodes, self.n_leaves // (2*n_nodes))
            current_mixture = current_mixture.flatten(start_dim=1, end_dim=2)                                # (batch_size, self.n_leaves)

            if self.usage_mode != 'none' and current_depth != self.depth.item() - 1:
                if self.usage_mode == 'soft':
                    current_node_usage = mixture_modifier.squeeze(-1).sum(dim=0)                            # (n_nodes*2,)
                elif self.usage_mode == 'hard':
                    current_node_usage = torch.round(mixture_modifier).squeeze(-1).sum(dim=0)                # (n_nodes*2,)
                self.node_usage[next_platform:next_platform+n_nodes*2] += current_node_usage.detach()        # (n_nodes*2,)

            del mixture_modifier, boundary_effect, not_boundary_effect, boundary_plane_logits, boundary_plane_coeff_scores, current_weights, current_biases

        if self.usage_mode != 'none':
            if self.usage_mode == 'hard':
                current_leaf_usage = torch.round(current_mixture).sum(dim=0)    # (n_leaves,)
            else:
                current_leaf_usage = current_mixture.sum(dim=0)                    # (n_leaves,)
            self.leaf_usage.data += current_leaf_usage.detach()

        new_logits = torch.empty((batch_size, self.n_leaves, self.output_width), dtype=torch.float, device=x.device)
        for i in range(new_logits.shape[1]):
            # convx = torch.max(torch.nn.functional.conv2d(x, self.cw1s[i], padding='same').view(batch_size, self._n_filters, -1), -1).values.view(new_logits.shape[0], -1) + self.cb1s[i]
            convx = torch.nn.functional.conv2d(x, self.cw1s[i], padding='same')
            convx = torch.nn.functional.max_pool2d(convx, 2)
            convx = torch.nn.functional.relu(convx)
            convx = torch.nn.functional.conv2d(convx, self.cw2s[i], padding='same')
            convx = torch.nn.functional.relu(convx)
            convx = torch.max(convx.view(new_logits.shape[0], self._n_filters, -1), -1).values.view(new_logits.shape[0], -1)
            hidden = torch.matmul(
                convx.squeeze(1),                    # (1, self.input_width)
                self.w1s[i]                # (self.input_width, self.leaf_width)
            )                                                 # (1, self.leaf_width)
            hidden += self.b1s[i].unsqueeze(-2)    # (1, self.leaf_width)
            new_logits[:, i] = torch.matmul(
                hidden,
                self.w2s[i]
            )
            new_logits[:, i] += self.b2s[i].unsqueeze(-2)    # (1, self.leaf_width)
        # element_logits = torch.matmul(convx, self.w1s.transpose(0, 1).flatten(1, 2))            # (batch_size, self.n_leaves * self.leaf_width)
        # element_logits = element_logits.view(batch_size, self.n_leaves, self.leaf_width)    # (batch_size, self.n_leaves, self.leaf_width)
        # element_logits += self.b1s.view(1, *self.b1s.shape)                                    # (batch_size, self.n_leaves, self.leaf_width)
        # element_activations = self.activation(element_logits)                                # (batch_size, self.n_leaves, self.leaf_width)
        # element_activations = self.leaf_dropout(element_activations)                        # (batch_size, self.n_leaves, self.leaf_width)
        # for i in range(self.n_leaves):
        #     new_logits[:, i] = torch.matmul(
        #         element_activations[:, i],
        #         self.w2s[i]
        #     ) + self.b2s[i]
        # new_logits has shape (batch_size, self.n_leaves, self.output_width)

        new_logits *= current_mixture.unsqueeze(-1)            # (batch_size, self.n_leaves, self.output_width)
        final_logits = new_logits.sum(dim=1)                # (batch_size, self.output_width)
        
        final_logits = final_logits.view(original_shape[0], self.output_width)    # (..., self.output_width)

        if not return_entropies:
            return final_logits
        else:
            return final_logits, entropies.mean(dim=0)
        
    def forward(self, x: torch.Tensor, return_entropies: bool=False, use_hard_decisions=None):
        """
        Computes the forward pass of this FFF.
        If `self.training` is True, `training_forward()` will be called, otherwise `eval_forward()` will be called.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must have shape (..., input_width).
        return_entropies : bool, optional
            Whether to return the entropies of the decisions made at each node. Defaults to False.
            If True, the mean batch entropies for each node will be returned as a tensor of shape (n_nodes,).
        use_hard_decisions : bool, optional
            Whether to use hard decisions during the forward pass. Defaults to None.
            If None and `self.training` is True, will effectively be False.
            If None and `self.training` is False, will effectively be True.
            Cannot be set to False if `self.training` is False.

        
        Returns
        -------
        torch.Tensor
            The output tensor. Will have shape (..., output_width).
        torch.Tensor, optional
            The mean batch entropies for each node. Will be returned with shape (n_nodes,) if `return_entropies` is True.
            Will not be returned if `return_entropies` is False.
        
        Raises
        ------
        ValueError
            - if `x` does not have shape (..., input_width)
            - if `return_entropies` is True and `self.training` is False
            - if `use_hard_decisions` is False and `self.training` is False

        See Also
        --------
        `training_forward()`
        `eval_forward()`
        """

        if self.training:
            return self.training_forward(x, return_entropies=return_entropies, use_hard_decisions=use_hard_decisions if use_hard_decisions is not None else False)
        else:
            if return_entropies:
                raise ValueError("Cannot return entropies during evaluation.")
            if use_hard_decisions is not None and not use_hard_decisions:
                raise ValueError("Cannot use soft decisions during evaluation.")
            return self.eval_forward(x)

    def eval_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the forward pass of this FFF during evaluation (i.e. making hard decisions at each node and traversing the FFF in logarithmic time).

        Parameters
        ----------
        x : torch.Tensor
            The input tensor. Must have shape (..., input_width).

        Returns
        -------
        torch.Tensor
            The output tensor. Will have shape (..., output_width).

        Notes
        -----
        - Dropout and region leaks are not engaged by this method.

        """
        original_shape = x.shape
        batch_size = x.shape[0]
        # x has shape (batch_size, input_width)

        current_nodes = torch.zeros((batch_size,), dtype=torch.long, device=x.device)
        for i in range(self.depth.item()):
            plane_coeffs = self.node_weights.index_select(dim=0, index=current_nodes)        # (batch_size, ch, h, w)
            plane_offsets = self.node_biases.index_select(dim=0, index=current_nodes)        # (batch_size, 1)
            # plane_coeff_score = torch.bmm(x.unsqueeze(1), plane_coeffs.unsqueeze(-1))        # (batch_size, 1, 1)
            plane_coeff_score = torch.max(torch.nn.functional.conv2d(x, plane_coeffs.squeeze(1), padding='same').view(batch_size, -1), -1).values.view(-1, 1)
            plane_score = plane_coeff_score.squeeze(-1) + plane_offsets.squeeze(-1)
            plane_choices = (plane_score.squeeze(-1) >= 0).long()                            # (batch_size,)

            platform = torch.tensor(2 ** i - 1, dtype=torch.long, device=x.device)            # (batch_size,)
            next_platform = torch.tensor(2 ** (i+1) - 1, dtype=torch.long, device=x.device)    # (batch_size,)
            current_nodes = (current_nodes - platform) * 2 + plane_choices + next_platform    # (batch_size,)

        leaves = current_nodes - next_platform                # (batch_size,)
        new_logits = torch.empty((batch_size, self.output_width), dtype=torch.float, device=x.device)
        for i in range(leaves.shape[0]):
            leaf_index = leaves[i]
            # x[i] = torch.max(torch.nn.functional.conv2d(x, self.cw1s[leaf_index], padding='same').view(leaves.shape[0], self._n_filters, -1), -1).values.view(leaves.shape[0], -1) + self.cb1s[leaf_index]
            convx = torch.nn.functional.conv2d(x[i:i+1], self.cw1s[leaf_index], padding='same')
            convx = torch.nn.functional.max_pool2d(convx, 2)
            convx = torch.nn.functional.relu(convx)
            convx = torch.nn.functional.conv2d(convx, self.cw2s[leaf_index], padding='same')
            convx = torch.nn.functional.relu(convx)
            convx = torch.max(convx.view(1, self._n_filters, -1), -1).values.view(1, -1)
            logits = torch.matmul(
                # x[i].unsqueeze(0),                    # (1, self.input_width)
                convx,
                self.w1s[leaf_index]                # (self.input_width, self.leaf_width)
            )                                                 # (1, self.leaf_width)
            logits += self.b1s[leaf_index].unsqueeze(-2)    # (1, self.leaf_width)
            activations = self.activation(logits)            # (1, self.leaf_width)
            new_logits[i] = torch.matmul(
                activations,
                self.w2s[leaf_index]
            ).squeeze(-2)                                    # (1, self.output_width)

        return new_logits.view(*original_shape[:-1], self.output_width)    # (..., self.output_width)

def compute_n_params(input_width: int, l_w: int, depth: int, output_width: int):
    fff = Net(input_width, l_w, output_width, depth, 0, 0)
    # fff = ViTFFF((3, 4, 4), 8, 4, 10, 2)
    ff = FF(input_width, l_w, output_width)
    fff = ConvFFF(
            input_width=(3, 32, 32), leaf_width=4, output_width=10, depth=depth,
            activation=torch.nn.ReLU(), dropout=0.0, train_hardened=False,
            region_leak=0.0, usage_mode= 'none', conv_size=5, n_channels=3
    )

    n_ff = 0
    n_fff = 0
    for p in ff.parameters():
        n_ff += p.numel()
    for i, p in enumerate(fff.parameters()):
        print(f"[{i}-th layer]: {p.shape}")
        n_fff += p.numel()

    print(f"FFF: {n_fff}\nFF: {n_ff}")


def main():
    typer.run(compute_n_params)


if __name__ == "__main__":
    main()
