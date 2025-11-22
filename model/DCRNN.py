"""
Some code are adapted from https://github.com/liyaguang/DCRNN
and https://github.com/xlwang233/pytorch-DCRNN, which are
licensed under the MIT License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scripts.data_utils import computeFFT
from model.cell import DCGRUCell
from torch.autograd import Variable
import EvoContext.scripts.utils2 as utils2
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


def apply_tuple(tup, fn):
    """Apply a function to a Tensor or a tuple of Tensor
    """
    if isinstance(tup, tuple):
        return tuple((fn(x) if isinstance(x, torch.Tensor) else x)
                     for x in tup)
    else:
        return fn(tup)


def concat_tuple(tups, dim=0):
    """Concat a list of Tensors or a list of tuples of Tensor
    """
    if isinstance(tups[0], tuple):
        return tuple(
            (torch.cat(
                xs,
                dim) if isinstance(
                xs[0],
                torch.Tensor) else xs[0]) for xs in zip(
                *
                tups))
    else:
        return torch.cat(tups, dim)

import torch
import torch.nn as nn
import time
import torch_geometric.nn as geo_nn
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import more_itertools as xitertools
import numpy as np
import abc
#
from typing import Dict, Callable, cast, List, Tuple
# from mamba_ssm import Mamba

GLOROTS: Dict[type, Callable[[torch.nn.Module, torch.Generator], int]]


def activatize(name: str, /) -> torch.nn.Module:
    R"""
    Get activation module.
    """
    #
    if name == "softplus":
        #
        return torch.nn.Softplus()
    elif name == "sigmoid":
        #
        return torch.nn.Sigmoid()
    elif name == "tanh":
        #
        return torch.nn.Tanh()
    elif name == "identity":
        #
        return torch.nn.Identity()
    else:
        # EXPECT:
        # It is possible to require unsupporting sequential model.
        raise RuntimeError(
            "Activation module identifier \"{:s}\" is not supported."
            .format(name),
        )

def glorot(module: torch.nn.Module, rng: torch.Generator, /) -> int:
    R"""
    Module initialization.
    """
    #
    return GLOROTS[type(module)](module, rng)

def noescape(string: str, /) -> str:
    R"""
    Remove escaping charaters.
    """
    #
    return re.sub(r"\x1b\[[0-9]+(;[0-9]+)*m", "", string)


def infotab5(title: str, lines: List[str]) -> List[str]:
    R"""
    Wrap given lines into a named tab.
    """
    # Format final generated lines according to their maximum length.
    linlen = (
        0 if len(lines) == 0 else max(len(noescape(line)) for line in lines)
    )
    barlen = max(5, (linlen - len(title)) // 2)
    return (
        [
            "\x1b[2m{:s}\x1b[0m".format("-" * barlen)
            + "\x1b[1m{:s}\x1b[0m".format(" " + title + " ")
            + "\x1b[2m{:s}\x1b[0m".format("-" * barlen),
        ]
        + lines
        + ["\x1b[2m{:s}\x1b[0m".format("-" * ((barlen + 1) * 2 + len(title)))]
    )


def auto_num_heads(embed_size: int, /) -> int:
    R"""
    Automatically get number of multi-heads.
    """
    #
    return (
        xitertools.first_true(
            range(int(np.ceil(np.sqrt(embed_size))), 0, -1),
            default=1, pred=lambda x: embed_size % x == 0 and x & (x - 1) == 0,
        )
    )

class Model(abc.ABC, torch.nn.Module):
    R"""
    Model.
    """
    #
    COSTS: Dict[str, List[float]]

    #
    COSTS = {"graph": [], "non-graph": [], "edges": []}

    # Signal to use simplest model for greatest efficency on synthetic tasks.
    SIMPLEST = False

    def __annotation__(self, /) -> None:
        R"""
        Annotate for class instance attributes.
        """
        #
        self.feat_target_size: int

    @abc.abstractmethod
    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        ...

    def initialize(self, seed: int, /) -> None:
        R"""
        Explicitly initialize the model.
        """
        #
        rng = torch.Generator("cpu")
        rng.manual_seed(seed)
        resetted = self.reset(rng)
        if resetted != sum(param.numel() for param in self.parameters()):
            # UNEXPECT:
            # All defined parameters should match exactly with initialization.
            raise NotImplementedError(
                "Defined parameters do not exactly match with initialized "
                "parameters.",
            )
        self.num_resetted_params = resetted

    def __repr__(self) -> str:
        R"""
        Get representation of the class.
        """
        # Model parameter key info has a special visible representation.
        names = []
        shapes = []
        for name, param in self.named_parameters():
            #
            names.append(name.split("."))
            shapes.append(
                "\x1b[90mx\x1b[0m".join(str(dim) for dim in param.shape),
            )
        depth = 0 if len(names) == 0 else max(len(levels) for levels in names)
        padded = [levels + [""] * (depth - len(levels)) for levels in names]

        #
        keys = (
            "\x1b[90m-\x1b[92m→\x1b[0m".join(levels).replace(
                "\x1b[92m→\x1b[0m\x1b[90m", "\x1b[90m→",
            )
            for levels in (
                [
                    [
                        "{:<{:d}s}".format(name, maxlen).replace(
                            " ", "\x1b[90m-\x1b[0m",
                        )
                        for (name, maxlen) in (
                            zip(
                                levels,
                                (
                                    max(len(name) for name in level)
                                    for level in zip(*padded)
                                ),
                            )
                        )
                    ]
                    for levels in padded
                ]
            )
        )

        # We may also care about the product besides shape.
        maxlen = (
            0
            if len(shapes) == 0 else
            max(len(noescape(shape)) for shape in shapes)
        )
        shapes = (
            [
                "{:s}{:s} ({:d})".format(
                    " " * (maxlen - len(noescape(shape))), shape,
                    int(
                        np.prod(
                            [
                                int(dim)
                                for dim in shape.split("\x1b[90mx\x1b[0m")
                            ],
                        ),
                    ),
                )
                for shape in shapes
            ]
        )

        # Generate final representation.
        return "\n".join(infotab5(
            "(Param)eter",
            [
                key + "\x1b[90m→\x1b[94m:\x1b[0m " + shape
                for (key, shape) in zip(keys, shapes)
            ],
        ))

    def moveon(self, notembedon: List[int]) -> None:
        R"""
        Set axes for moving window model.
        """
        # EXPECT:
        # By default, the model is not moving window unless it is overloaded.
        raise RuntimeError(
            "Default model is not a moving window model, and you need to "
            "explicitly overload to use moving window.",
        )

    def pretrain(self, partname: str, path: str, /) -> None:
        R"""
        Use pretrained model.
        """
        # EXPECT:
        # By default, there is no pretraining definition.
        raise RuntimeError(
            "No pretraining of \"{:s}\" is defined."
            .format(self.__class__.__name__),
        )
    
class GNNx2(Model):
    R"""
    Graph neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int, embed_inside_size: int,
        /,
        *,
        convolve: str, skip: bool, activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # TODO:
        # Given more than 2 layers, we can introduce dense connection.
        self.gnn1 = (
            self.graphicalize(
                convolve, feat_input_size_edge, feat_input_size_node,
                embed_inside_size,
                activate=activate,
            )
        )
        self.gnn2 = (
            self.graphicalize(
                convolve, feat_input_size_edge, embed_inside_size,
                feat_target_size,
                activate=activate,
            )
        )

        #
        self.edge_transform: torch.nn.Module
        self.skip: torch.nn.Module

        #
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            #
            self.edge_transform = torch.nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = torch.nn.Identity()
            self.edge_activate = activatize("identity")

        #
        if feat_input_size_node == feat_target_size:
            #
            self.skip = torch.nn.Identity()
        else:
            #
            self.skip = (
                torch.nn.Linear(feat_input_size_node, feat_target_size)
            )

        #
        self.activate = activatize(activate)

        # Use a 0-or-1 integer to mask skip connection.
        self.doskip = int(skip)

    def graphicalize(
        self,
        name: str, feat_input_size_edge: int, feat_input_size_node: int,
        feat_target_size: int,
        /,
        *,
        activate: str,
    ) -> torch.nn.Module:
        R"""
        Get unit graphical module.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        if name == "gcn":
            #
            module = (
                geo_nn.GCNConv(feat_input_size_node, feat_target_size)
            )
        elif name == "gcnub":
            #
            module = (
                geo_nn.GCNConv(
                    feat_input_size_node, feat_target_size,
                    bias=False,
                )
            )
        elif name == "gatedgcn":
            #
            module = (
                geo_nn.ResGatedGraphConv(
                    feat_input_size_node, feat_target_size,
                    edge_dim=1,
                )
            )
        elif name == "ssg":
            #
            module = (
                geo_nn.SSGConv(
                    feat_input_size_node, feat_target_size,alpha=0.05
                )
            )
        elif name == "sg":
            #
            module = (
                geo_nn.SGConv(
                    feat_input_size_node, feat_target_size,
                )
            )
        elif name == "gat":
            #
            heads = auto_num_heads(feat_target_size)
            module = (
                geo_nn.GATConv(
                    feat_input_size_node, feat_target_size // heads,
                    heads=heads, edge_dim=feat_input_size_edge,
                )
            )
        elif name == "cheb":
            #
            module = (
                geo_nn.ChebConv(feat_input_size_node, feat_target_size, 2)
            )
        elif name == "gin":
            #
            module = (
                geo_nn.GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            feat_input_size_node, feat_target_size,
                        ),
                        activatize(activate),
                        torch.nn.Linear(feat_target_size, feat_target_size),
                    ),
                    edge_dim=1,
                )
            )
        else:
            # EXPECT:
            # It is possible to require unsupporting sequential model.
            raise RuntimeError(
                "Graphical module identifier \"{:s}\" is not supported."
                .format(name),
            )
        return cast(torch.nn.Module, module)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn1, rng)
        resetted = resetted + glorot(self.gnn2, rng)
        resetted = resetted + glorot(self.edge_transform, rng)
        resetted = resetted + glorot(self.skip, rng)
        return resetted

    def convolve(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Convolve.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        node_embeds: torch.Tensor

        #
        node_embeds = (
            self.gnn1.forward(node_feats, edge_tuples, edge_feats)
        )
        node_embeds = (
            self.gnn2.forward(
                self.activate(node_embeds), edge_tuples, edge_feats,
            )
        )
        return node_embeds

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        edge_embeds: torch.Tensor
        node_embeds: torch.Tensor
        node_residuals: torch.Tensor

        #
        edge_embeds = (
            self.edge_activate(self.edge_transform.forward(edge_feats))
        )
        node_embeds = self.convolve(edge_tuples, edge_embeds, node_feats)
        node_residuals = self.skip.forward(node_feats)
        return node_embeds + self.doskip * node_residuals


class GNNx2Concat(GNNx2):
    R"""
    Graph neural network (2-layer) with input concatenation.
    """
    #
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # Super call.
        node_embeds = GNNx2.forward(self, edge_tuples, edge_feats, node_feats)
        node_embeds = torch.cat((node_embeds, node_feats), dim=1)
        return node_embeds


def graphicalize(
    name: str, 
    feat_input_size_edge, 
    feat_input_size_node: int,
    feat_target_size: int, 
    embed_inside_size: int,
    /,
    *,
    skip: bool, activate: str, concat: bool,
) -> Model:
    R"""
    Get 2-layer graphical module.
    """
    #
    if concat:
        #
        return (
            GNNx2Concat(
                feat_input_size_edge, 
                feat_input_size_node, 
                feat_target_size,
                embed_inside_size,
                convolve=name, 
                skip=skip, 
                activate=activate,
            )
        )
    else:
        #
        return (
            GNNx2(
                feat_input_size_edge, feat_input_size_node, feat_target_size,
                embed_inside_size,
                convolve=name, skip=skip, activate=activate,
            )
        )

class GNNx2Concat(GNNx2):
    R"""
    Graph neural network (2-layer) with input concatenation.
    """
    #
    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        node_embeds: torch.Tensor

        # Super call.
        node_embeds = GNNx2.forward(self, edge_tuples, edge_feats, node_feats)
        node_embeds = torch.cat((node_embeds, node_feats), dim=1)
        return node_embeds
    
class GNNx2(Model):
    R"""
    Graph neural network (2-layer).
    """
    def __init__(
        self,
        feat_input_size_edge: int, 
        feat_input_size_node: int,
        feat_target_size: int, 
        embed_inside_size: int,
        /,
        *,
        convolve: str, 
        skip: bool,
        activate: str,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # TODO:
        # Given more than 2 layers, we can introduce dense connection.
        self.gnn1 = (
            self.graphicalize(
                convolve, 
                feat_input_size_edge, 
                feat_input_size_node,
                embed_inside_size,
                activate=activate,
            )
        )
        self.gnn2 = (
            self.graphicalize(
                convolve, 
                feat_input_size_edge, 
                embed_inside_size,
                feat_target_size,
                activate=activate,
            )
        )

        #
        self.edge_transform: torch.nn.Module
        self.skip: torch.nn.Module

        #
        if feat_input_size_edge > 1 and convolve in ("gcn", "gcnub", "cheb"):
            #
            self.edge_transform = torch.nn.Linear(feat_input_size_edge, 1)
            self.edge_activate = activatize("softplus")
        else:
            self.edge_transform = torch.nn.Identity()
            self.edge_activate = activatize("identity")

        #
        if feat_input_size_node == feat_target_size:
            #
            self.skip = torch.nn.Identity()
        else:
            #
            self.skip = (
                torch.nn.Linear(feat_input_size_node, feat_target_size)
            )

        #
        self.activate = activatize(activate)

        # Use a 0-or-1 integer to mask skip connection.
        self.doskip = int(skip)

    def graphicalize(
        self,
        name: str, 
        feat_input_size_edge: int, 
        feat_input_size_node: int,
        feat_target_size: int,
        /,
        *,
        activate: str,
    ) -> torch.nn.Module:
        R"""
        Get unit graphical module.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        if name == "gcn":
            #
            module = (
                geo_nn.GCNConv(feat_input_size_node, feat_target_size)
            )
        elif name == "gcnub":
            #
            module = (
                geo_nn.GCNConv(
                    feat_input_size_node, feat_target_size,
                    bias=False,
                )
            )
        elif name == "gatedgcn":
            #
            module = (
                geo_nn.ResGatedGraphConv(
                    feat_input_size_node, feat_target_size,
                )
            )
        elif name == "gat":
            #
            heads = auto_num_heads(feat_target_size)
            module = (
                geo_nn.GATConv(
                    feat_input_size_node, feat_target_size // heads,
                    heads=heads, edge_dim=feat_input_size_edge,
                )
            )
        elif name == "cheb":
            #
            module = (
                geo_nn.ChebConv(feat_input_size_node, feat_target_size, 2)
            )
        elif name == "gin":
            #
            module = (
                geo_nn.GINEConv(
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            feat_input_size_node, feat_target_size,
                        ),
                        activatize(activate),
                        torch.nn.Linear(feat_target_size, feat_target_size),
                    ),
                    edge_dim=feat_input_size_edge,
                )
            )
        else:
            # EXPECT:
            # It is possible to require unsupporting sequential model.
            raise RuntimeError(
                "Graphical module identifier \"{:s}\" is not supported."
                .format(name),
            )
        return cast(torch.nn.Module, module)

    def reset(self, rng: torch.Generator, /) -> int:
        R"""
        Reset model parameters by given random number generator.
        """
        #
        resetted = 0
        resetted = resetted + glorot(self.gnn1, rng)
        resetted = resetted + glorot(self.gnn2, rng)
        resetted = resetted + glorot(self.edge_transform, rng)
        resetted = resetted + glorot(self.skip, rng)
        return resetted

    def convolve(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Convolve.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        node_embeds: torch.Tensor

        #
        node_embeds = (
            self.gnn1.forward(node_feats, edge_tuples, edge_feats.squeeze())
        )
        node_embeds = (
            self.gnn2.forward(
                self.activate(node_embeds), edge_tuples, edge_feats.squeeze(),
            )
        )
        return node_embeds

    def forward(
        self,
        edge_tuples: torch.Tensor, edge_feats: torch.Tensor,
        node_feats: torch.Tensor,
        /,
    ) -> torch.Tensor:
        R"""
        Forward.
        """
        # TODO:
        # Wait for Pytorch Geometric type annotation supporting.
        edge_embeds: torch.Tensor
        node_embeds: torch.Tensor
        node_residuals: torch.Tensor

        edge_embeds = edge_feats
        node_embeds = self.convolve(edge_tuples, edge_embeds, node_feats)
        node_residuals = self.skip.forward(node_feats)
        return node_embeds + self.doskip * node_residuals
    


class Linear(torch.nn.Module):
    R"""
    Linear but recurrent module.
    """
    def __init__(self, feat_input_size: int, feat_target_size: int, /) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        self.feat_input_size = feat_input_size
        self.feat_target_size = feat_target_size
        self.lin = torch.nn.Linear(self.feat_input_size, self.feat_target_size)

    def forward(
        self,
        tensor: torch.Tensor,
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        (num_times, num_samples, _) = tensor.shape
        return (
            torch.reshape(
                self.lin.forward(
                    torch.reshape(
                        tensor,
                        (num_times * num_samples, self.feat_input_size),
                    ),
                ),
                (num_times, num_samples, self.feat_target_size),
            ),
            tensor[-1],
        )


class Static(torch.nn.Module):
    R"""
    Treate static feature as dynamic.
    """
    def forward(
        self,
        tensor: torch.Tensor,
        /,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        return (torch.reshape(tensor, (1, *tensor.shape)), tensor)


class MultiheadAttention(torch.nn.Module):
    R"""
    Multi-head attention with recurrent-like forward.
    """
    def __init__(self, feat_input_size: int, feat_target_size: int, /) -> None:
        R"""
        Initialize the class.
        """
        #
        torch.nn.Module.__init__(self)

        #
        embed_size = feat_target_size
        self.num_heads = auto_num_heads(embed_size)
        self.mha = torch.nn.MultiheadAttention(embed_size, self.num_heads)

        #
        self.transform: torch.nn.Module

        #
        if feat_input_size != embed_size:
            #
            self.transform = (
                torch.nn.Linear(feat_input_size, embed_size, bias=False)
            )
        else:
            #
            self.transform = torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        R"""
        Forward.
        """
        #
        x = self.transform(x)
        (y, attn) = self.mha.forward(x, x, x)
        return (y, cast(torch.Tensor, attn))
    
def sequentialize(
    name: str, feat_input_size: int, feat_target_size: int,
    /,
) -> torch.nn.Module:
    R"""
    Get sequential module.
    """
    #
    print("snn: ", name)
    # if name == "mamba":
    #     return nn.Sequential(
    #         nn.Linear(feat_input_size, feat_target_size),
    #         Mamba(
    #             d_model=feat_target_size, # Model dimension d_model
    #             d_state=16,  # SSM state expansion factor
    #             d_conv=4,    # Local convolution width
    #             expand=2,    # Block expansion factor
    #         ),
    #     )
    if name == "linear":
        #
        return Linear(feat_input_size, feat_target_size)
    elif name == "gru":
        #
        return torch.nn.GRU(feat_input_size, feat_target_size)
    elif name == "lstm":
        #
        return torch.nn.LSTM(feat_input_size, feat_target_size)
    elif name == "gru[]":
        #
        return torch.nn.GRUCell(feat_input_size, feat_target_size)
    elif name == "lstm[]":
        #
        return torch.nn.LSTMCell(feat_input_size, feat_target_size)
    elif name == "mha":
        #
        return MultiheadAttention(feat_input_size,feat_target_size)
    elif name == "static":
        #
        return Static()
    else:
        # EXPECT:
        # It is possible to require unsupporting sequential model.
        raise RuntimeError(
            "Sequential module identifier is not supported."
        )

    
class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step,
                 hid_dim, num_nodes, num_rnn_layers,
                 dcgru_activation=None, filter_type='laplacian',
                 device=None):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device

        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(
                DCGRUCell(
                    input_dim=hid_dim,
                    num_units=hid_dim,
                    max_diffusion_step=max_diffusion_step,
                    num_nodes=num_nodes,
                    nonlinearity=dcgru_activation,
                    filter_type=filter_type))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, initial_hidden_state, supports):
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]
        # (seq_length, batch_size, num_nodes*input_dim)
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self.num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            # print(f"support shape: {supports.shape}")
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    supports[:, t, :, :], current_inputs[t, ...], hidden_state)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            current_inputs = torch.stack(output_inner, dim=0).to(
                self._device)  # (seq_len, batch_size, num_nodes * rnn_units)
        output_hidden = torch.stack(output_hidden, dim=0).to(
            self._device)  # (num_layers, batch_size, num_nodes * rnn_units)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # (num_layers, batch_size, num_nodes * rnn_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self, input_dim, max_diffusion_step, num_nodes,
                 hid_dim, output_dim, num_rnn_layers, dcgru_activation=None,
                 filter_type='laplacian', device=None, dropout=0.0):
        # input_dim -> 
        super(DCGRUDecoder, self).__init__()

        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.num_rnn_layers = num_rnn_layers
        self._device = device
        self.dropout = dropout

        cell = DCGRUCell(input_dim=hid_dim, 
                         num_units=hid_dim,
                         max_diffusion_step=max_diffusion_step,
                         num_nodes=num_nodes, 
                         nonlinearity=dcgru_activation,
                         filter_type=filter_type)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(
            DCGRUCell(
                input_dim=input_dim,
                num_units=hid_dim,
                max_diffusion_step=max_diffusion_step,
                num_nodes=num_nodes,
                nonlinearity=dcgru_activation,
                filter_type=filter_type))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            decoding_cells.append(cell)

        self.decoding_cells = nn.ModuleList(decoding_cells)
        self.projection_layer = nn.Linear(self.hid_dim, self.output_dim)
        self.dropout = nn.Dropout(p=dropout)  # dropout before projection layer

    def forward(
            self,
            inputs,
            initial_hidden_state,
            supports,
            teacher_forcing_ratio=None):
        """
        Args:
            inputs: shape (seq_len, batch_size, num_nodes, output_dim)
            initial_hidden_state: the last hidden state of the encoder, shape (num_layers, batch, num_nodes * rnn_units)
            supports: list of supports from laplacian or dual_random_walk filters
            teacher_forcing_ratio: ratio for teacher forcing
        Returns:
            outputs: shape (seq_len, batch_size, num_nodes * output_dim)
        """
        seq_length, batch_size, _, _ = inputs.shape
        inputs = torch.reshape(inputs, (seq_length, batch_size, -1))

        go_symbol = torch.zeros(
            (batch_size,
             self.num_nodes *
             self.output_dim)).to(
            self._device)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length,
            batch_size,
            self.num_nodes *
            self.output_dim).to(
            self._device)

        current_input = go_symbol  # (batch_size, num_nodes * input_dim)
        for t in range(seq_length):
            next_input_hidden_state = []
            for i_layer in range(0, self.num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    supports, current_input, hidden_state)
                current_input = output
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)

            projected = self.projection_layer(self.dropout(
                output.reshape(batch_size, self.num_nodes, -1)))
            projected = projected.reshape(
                batch_size, self.num_nodes * self.output_dim)
            outputs[t] = projected

            if teacher_forcing_ratio is not None:
                teacher_force = random.random() < teacher_forcing_ratio  # a bool value
                current_input = (inputs[t] if teacher_force else projected)
            else:
                current_input = projected

        return outputs
    
class CustomGRUCell(torch.nn.GRU):
    def __init__(self, input_size, hidden_size, num_nodes):
        super().__init__(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_nodes = num_nodes

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size*self.num_nodes, self.hidden_size)
    
class GRUGCN(nn.Module):
    """
    Sequential neural network then graph neural network (2-layer).
    """
    # feat_input_size_edge=args.input_dim, 
    #                            feat_input_size_node=args.input_dim,
    #                            feat_target_size=args.rnn_units, 
    #                            embed_inside_size=args.input_dim,
    #                            convolve=gnn, 
    #                            reduce_edge="gru",
    #                            reduce_node="gru", 
    #                            skip=True,
    #                            activate="tanh", 
    #                            concat=False,
    #                            neo_gnn=True,
    #                            num_layers=num_layers
    def __init__(
            self, 
            num_nodes, 
            feat_input_size_edge, 
            feat_input_size_node, 
            feat_target_size, 
            embed_inside_size, 
            convolve, 
            skip, 
            activate,
            concat, 
            neo_gnn,
            device, 
            num_layers=1
        ):
        super(GRUGCN, self).__init__()
        feat_input_size_edge =1

        snn_node = list()
        snn_edge = list()

        # first cell has different input dimension
        snn_node.append(torch.nn.GRU(feat_input_size_node, embed_inside_size))
        for i in range(1, num_layers):
            snn_node.append(torch.nn.GRU(embed_inside_size, embed_inside_size))
        
        snn_edge.append(torch.nn.GRU(feat_input_size_edge, embed_inside_size))
        for i in range(1, num_layers):
            snn_edge.append(torch.nn.GRU(embed_inside_size, embed_inside_size))

        gnnx2 = list()
        for i in range(num_layers):
            gnnx2.append(graphicalize(
                convolve,
                embed_inside_size,
                embed_inside_size,
                feat_target_size, 
                embed_inside_size, 
                skip=skip, 
                activate=activate, 
                concat=concat
            ))

        self.snn_node = nn.ModuleList(snn_node)
        self.snn_edge = nn.ModuleList(snn_edge)
        self.gnnx2 = nn.ModuleList(gnnx2)

        self.activate = activatize(activate)
        self.SIMPLEST = False

        self.edge_transform = torch.nn.Linear(embed_inside_size, 1)
        self.edge_activate = activatize("softplus")
        
        self.feat_target_size = feat_target_size + int(concat) * embed_inside_size

        self.neo_gnn = neo_gnn

        self.num_layers = num_layers

        self._device = device
        self.num_nodes = num_nodes
    def reset(self, rng):
        resetted = 0
        resetted = resetted + glorot(self.snn_edge, rng)
        resetted = resetted + glorot(self.snn_node, rng)
        resetted = resetted + self.gnnx2.reset(rng)
        return resetted

    
    def prepare_edge_weights(self, edge_embeds):
        """
        Prepare edge weights from edge embeddings for Laplacian computation.
        """
        # If edge embeddings are multi-dimensional, compute a scalar weight
        if edge_embeds.dim() > 1 and edge_embeds.size(-1) > 1:
            # Use the norm of the edge embedding as weight
            weights = torch.norm(edge_embeds, p=2, dim=-1)
        else:
            # If already scalar, just squeeze
            weights = edge_embeds.squeeze(-1)
        
        # Ensure positive weights
        weights = torch.abs(weights)
        
        # Normalize weights
        weights = weights / (weights.max() + 1e-6)
        
        return weights
    
    def init_hidden(self, batch_size):
            
            init_states_snn_node = []
            init_states_snn_edge = []
            for i in range(self.num_layers):
                init_states_snn_node.append(self.snn_node[i].init_hidden(batch_size))
            for i in range(self.num_layers):
                init_states_snn_edge.append(self.snn_edge[i].init_hidden(batch_size*self.num_nodes))
            # (num_layers, batch_size, num_nodes * rnn_units)
            return torch.stack(init_states_snn_node, dim=0), torch.stack(init_states_snn_edge, dim=0)
    
    def forward(self, inputs, supports):
        seq_length, b, node, dim = inputs.shape
        inputs = inputs.reshape(seq_length, b*node, dim)
        output_hidden_node = []
        for i_layer in range(self.num_layers):
            node_embeds, hn =  self.snn_node[i_layer](inputs)
            output_hidden_node.append(hn)
            inputs = node_embeds
        output_hidden_node = torch.stack(output_hidden_node, dim=0).to(
        self._device) 

        output_hidden_node = output_hidden_node.reshape(self.num_layers, b, node, -1)
        if supports.shape[2] == 1:
            supports = torch.squeeze(supports, dim=2)
        edge_tuples, edge_features = self.create_edge_tuples_and_features(supports)

        output_hidden_edge = []
        current_inputs_ssn_edge = edge_features.reshape(seq_length, -1, 1)
        for i_layer in range(self.num_layers):
            edge_out, hn = self.snn_edge[i_layer](current_inputs_ssn_edge)
            current_inputs_ssn_edge = edge_out
            output_hidden_edge.append(hn)
        output_hidden_edge = torch.stack(output_hidden_edge, dim=0).to(
        self._device) 

        output_hidden_edge = output_hidden_edge.reshape(self.num_layers, b, -1, dim)

        all_node_embeds = output_hidden_node
        all_edge_embeds = output_hidden_edge

        edge_tuples = edge_tuples.to(next(self.parameters()).device)

        outputs = []
        for l in range(self.num_layers):
            outputs_batch = []
            for i in range(b):
                current_node_embeds = all_node_embeds[l, i].to(next(self.parameters()).device)
                current_edge_embeds = all_edge_embeds[l, i].to(next(self.parameters()).device)
                
                edge_weights = self.edge_activate(self.edge_transform(current_edge_embeds))

                node_embeds = self.gnnx2[l].forward(
                    edge_tuples, 
                    edge_weights,
                    self.activate(current_node_embeds)
                )
                # print(f"node_embeds shape -: {node_embeds.shape}")
                outputs_batch.append(node_embeds)

            outputs.append(torch.stack(outputs_batch, dim=0))
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.reshape(self.num_layers, b, -1)
        # (num_layers, batch_size, num_nodes, rnn_units + input_dim)
        return outputs
    
    def create_edge_tuples_and_features(self, adj):
        adj.to(next(self.parameters()).device)
        batch_size, timesteps, num_nodes, _ = adj.shape
        node_indices = torch.arange(num_nodes)
        edge_tuples = torch.stack(torch.meshgrid(node_indices, node_indices)).reshape(2, -1).to(next(self.parameters()).device)
        edge_features = adj.reshape(timesteps, batch_size, -1).unsqueeze(-1) # (timesteps, batch_size, num_nodes num_nodes, 1)
        # non_zero_indices = torch.nonzero((edge_features.sum(dim=(0, 1)) > 0.0001) | (edge_features.sum(dim=(0, 1)) < -0.0001)).to(next(self.parameters()).device)
        # non_zero_edge_tuples = edge_tuples[:, non_zero_indices[:,0]]

        # non_zero_edge_features = edge_features[:, :, non_zero_indices[:, 0], non_zero_indices[:, 1]]
        # non_zero_edge_features = non_zero_edge_features.reshape(timesteps, batch_size, -1, 1)

        return edge_tuples, edge_features
    

class GRUGCNEncoder(nn.Module):
    """
    Sequential neural network then graph neural network (2-layer) adapted for classification.
    """
    def __init__(self, args, device=None, gnn="gcn", num_layers=1):
        super(GRUGCNEncoder, self).__init__()
        
        self.num_nodes = args.num_nodes
        self.device = device 
        self.gru_gcn = GRUGCN(
            num_nodes=self.num_nodes,
            feat_input_size_edge=args.input_dim, 
            feat_input_size_node=args.input_dim,
            feat_target_size=args.rnn_units, 
            embed_inside_size=args.input_dim,
            convolve=gnn, 
            skip=False,
            activate="tanh", 
            concat=True,
            neo_gnn=True,
            device=self.device,
            num_layers=num_layers
        )
    def forward(self, input_seq, adj):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        # (max_seq_len, batch, num_nodes, input_dim)
        final_hidden = self.gru_gcn(input_seq, adj)
        # (num_layers, batch, num_nodes*rnn_units)
        return final_hidden

    def init_hidden(self, batch):
        init_states_snn_node, init_states_snn_edge = self.gru_gcn.init_hidden(batch)
        return init_states_snn_node, init_states_snn_edge

########## Model for seizure classification/detection ##########
class STAGModel_classification(nn.Module):
    def __init__(self, args, num_classes, device=None):
        super(STAGModel_classification, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        self.enc_input_dim = args.input_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.num_classes = num_classes
        self.encoder = GRUGCNEncoder(args, device, "gcn", num_rnn_layers)
        self.fc = nn.Linear(rnn_units + args.input_dim, num_classes)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()

    def forward(self, input_seq, seq_lengths, supports):
        """
        Args:
            input_seq: input sequence, shape (batch, seq_len, num_nodes, input_dim)
            seq_lengths: actual seq lengths w/o padding, shape (batch,)
            supports: list of supports from laplacian or dual_random_walk filters
        Returns:
            pool_logits: logits from last FC layer (before sigmoid/softmax)
        """
        batch_size, max_seq_len = input_seq.shape[0], input_seq.shape[1]

        # (max_seq_len, batch, num_nodes, input_dim)
        input_seq = torch.transpose(input_seq, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        # init_hidden_state_node, init_hidden_state_edge = self.encoder.init_hidden(
        #     batch_size)
        # init_hidden_state_node = init_hidden_state_node.to(self._device)
        # init_hidden_state_edge = init_hidden_state_edge.to(self._device)

        # last hidden state of the encoder is the context
        # (num_layers, batch, rnn_units*num_nodes)
        final_hidden = self.encoder(input_seq,supports)
        # print(f"final hidden -: {final_hidden.shape}")
        # (batch_size, num_layers, rnn_units*num_nodes)
        output = torch.transpose(final_hidden, dim0=0, dim1=1)
        # print(output.shape)
        last_output = output[:, -1, :]
        # print(last_output.shape)
        last_output = last_output.reshape(batch_size,  self.num_nodes, self.rnn_units + self.enc_input_dim)
        # extract last relevant output
        # last_out = utils.last_relevant_pytorch(
        #     output, seq_lengths, batch_first=True)  # (batch_size, rnn_units*num_nodes)
        # # (batch_size, num_nodes, rnn_units)
        # last_out = last_out.view(batch_size, self.num_nodes, self.rnn_units)
        # last_out = last_out.to(self._device)

        # final FC layer
        logits = self.fc(self.relu(self.dropout(last_output)))

        # max-pooling over nodes
        pool_logits, _ = torch.max(logits, dim=1)  # (batch_size, num_classes)

        return pool_logits
########## Model for seizure classification/detection ##########


########## Model for next time prediction ##########
class STAGModel_nextTimePred(nn.Module):
    def __init__(self, args, device=None):
        super(STAGModel_nextTimePred, self).__init__()

        num_nodes = args.num_nodes
        num_rnn_layers = args.num_rnn_layers
        rnn_units = args.rnn_units
        # node feature input dimension. in paper Xi -> N*P , P -> input dimension
        enc_input_dim = args.input_dim
        # 
        dec_input_dim = args.output_dim
        output_dim = args.output_dim
        max_diffusion_step = args.max_diffusion_step

        self.num_nodes = args.num_nodes
        self.num_rnn_layers = num_rnn_layers
        self.rnn_units = rnn_units
        self._device = device
        self.output_dim = output_dim
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = bool(args.use_curriculum_learning)
        self.encoder = GRUGCNEncoder(args, device, "gcn", num_rnn_layers)
        self.decoder = DCGRUDecoder(input_dim=dec_input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=num_nodes, 
                                    hid_dim=rnn_units + enc_input_dim,
                                    output_dim=output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    dcgru_activation=args.dcgru_activation,
                                    filter_type=args.filter_type,
                                    device=device,
                                    dropout=args.dropout)

    def forward(
            self,
            encoder_inputs,
            decoder_inputs,
            input_adj_mat_seq,
            supports,
            batches_seen=None):
        """
        Args:
            encoder_inputs: encoder input sequence, shape (batch, input_seq_len, num_nodes, input_dim)
            encoder_inputs: decoder input sequence, shape (batch, output_seq_len, num_nodes, output_dim)
            supports: list of supports from laplacian or dual_random_walk filters
            batches_seen: number of examples seen so far, for teacher forcing
        Returns:
            outputs: predicted output sequence, shape (batch, output_seq_len, num_nodes, output_dim)
        """
        batch_size, output_seq_len, num_nodes, _ = decoder_inputs.shape

        # (seq_len, batch_size, num_nodes, input_dim)
        encoder_inputs = torch.transpose(encoder_inputs, dim0=0, dim1=1)
        # (seq_len, batch_size, num_nodes, output_dim)
        decoder_inputs = torch.transpose(decoder_inputs, dim0=0, dim1=1)

        # initialize the hidden state of the encoder
        # init_states_snn_node, init_states_snn_edge = self.encoder.init_hidden(batch_size)
        # init_states_snn_node = init_states_snn_node.to(self._device)
        # init_states_snn_edge = init_states_snn_edge.to(self._device)

        # encoder
        # (num_layers, batch, rnn_units*num_nodes) 64*22
        encoder_hidden_state = self.encoder(
            encoder_inputs,
            input_adj_mat_seq,
        )
        # print(f"encoder_hidden_state : {encoder_hidden_state.shape}")
        # decoder
        if self.training and self.use_curriculum_learning and (
                batches_seen is not None):
            teacher_forcing_ratio = utils2.compute_sampling_threshold(
                self.cl_decay_steps, batches_seen)
        else:
            teacher_forcing_ratio = None
        outputs = self.decoder(
            decoder_inputs,
            encoder_hidden_state,
            supports,
            teacher_forcing_ratio=teacher_forcing_ratio)  # (seq_len, batch_size, num_nodes * output_dim)
        # (seq_len, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((output_seq_len, batch_size, num_nodes, -1))
        # (batch_size, seq_len, num_nodes, output_dim)
        outputs = torch.transpose(outputs, dim0=0, dim1=1)

        return outputs
########## Model for next time prediction ##########
