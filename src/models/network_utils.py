from __future__ import annotations

from typing import Any, Callable, Union, cast

import numpy as np
import numpy.typing as npt
import tinycudann as tcnn
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeAlias

from src.constants import JOINT_FEAT_BASE_DIM, JOINT_POS_DIM, LAST_LAYER_INIT_STD, ROTMAT_FLAT_DIM, SKIP_CONNECTION_SCALE

ConfigPrimitive: TypeAlias = Union[
    "dict[str, ConfigPrimitive]", "list[ConfigPrimitive]", str, int, float, bool, None
]


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _make_periodic_embed_fn(
    p_fn: Callable[[torch.Tensor], torch.Tensor], freq: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return p_fn(x * freq)
    return fn


def _make_hannw_embed_fn(
    p_fn: Callable[[torch.Tensor], torch.Tensor], freq: torch.Tensor, w: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    def fn(x: torch.Tensor) -> torch.Tensor:
        return w * p_fn(x * freq)
    return fn


class Embedder:
    def __init__(
        self,
        include_input: bool,
        input_dims: int,
        max_freq_log2: float,
        num_freqs: int,
        log_sampling: bool,
        periodic_fns: list[Callable[[torch.Tensor], torch.Tensor]],
    ) -> None:
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.log_sampling = log_sampling
        self.periodic_fns = periodic_fns
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns: list[Callable[[torch.Tensor], torch.Tensor]] = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(_identity)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        freq_bands: torch.Tensor
        if self.log_sampling:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(_make_periodic_embed_fn(p_fn, freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires: int, input_dims: int = 3) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    if multires == 0:
        return (lambda x: x), input_dims
    assert multires > 0

    embedder_obj = Embedder(
        include_input=True,
        input_dims=input_dims,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        log_sampling=True,
        periodic_fns=[torch.sin, torch.cos],
    )

    def embed(x: torch.Tensor, eo: Embedder = embedder_obj) -> torch.Tensor:
        return eo.embed(x)

    return embed, embedder_obj.out_dim


class HannwEmbedder:
    def __init__(
        self,
        cfg: DictConfig,
        include_input: bool,
        input_dims: int,
        max_freq_log2: float,
        num_freqs: int,
        periodic_fns: list[Callable[[torch.Tensor], torch.Tensor]],
        iter_val: int,
    ) -> None:
        self.cfg = cfg
        self.include_input = include_input
        self.input_dims = input_dims
        self.max_freq_log2 = max_freq_log2
        self.num_freqs = num_freqs
        self.periodic_fns = periodic_fns
        self.iter_val = iter_val
        self.create_embedding_fn()

    def create_embedding_fn(self) -> None:
        embed_fns: list[Callable[[torch.Tensor], torch.Tensor]] = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(_identity)
            out_dim += d

        max_freq = self.max_freq_log2
        N_freqs = self.num_freqs

        freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)

        # get hann window weights
        alpha: torch.Tensor
        if self.cfg.full_band_iter <= 0 or self.cfg.kick_in_iter >= self.cfg.full_band_iter:
            alpha = torch.tensor(N_freqs, dtype=torch.float32)
        else:
            kick_in_iter = torch.tensor(self.cfg.kick_in_iter,
                                        dtype=torch.float32)
            t = torch.clamp(self.iter_val - kick_in_iter, min=0.)
            N = self.cfg.full_band_iter - kick_in_iter
            m = N_freqs
            alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            # print("freq_idx: ", freq_idx, "weight: ", w, "iteration: ", self.kwargs['iter_val'])
            for p_fn in self.periodic_fns:
                embed_fns.append(_make_hannw_embed_fn(p_fn, freq, w))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_hannw_embedder(
    cfg: DictConfig, multires: int, iter_val: int
) -> tuple[Callable[[torch.Tensor], torch.Tensor], int]:
    embedder_obj = HannwEmbedder(
        cfg,
        include_input=False,
        input_dims=3,
        max_freq_log2=multires - 1,
        num_freqs=multires,
        periodic_fns=[torch.sin, torch.cos],
        iter_val=iter_val,
    )

    def embed(x: torch.Tensor, eo: HannwEmbedder = embedder_obj) -> torch.Tensor:
        return eo.embed(x)

    return embed, embedder_obj.out_dim

class HierarchicalPoseEncoder(nn.Module):
    '''Hierarchical encoder from LEAP.'''

    def __init__(
        self,
        num_joints: int,
        ktree_parents: npt.NDArray[np.integer[Any]],
        rel_joints: bool,
        dim_per_joint: int,
        out_dim: int,
    ) -> None:
        super().__init__()

        self.num_joints = num_joints
        self.rel_joints = rel_joints
        self.ktree_parents = np.asarray(ktree_parents, dtype=np.int32)

        self.layer_0 = nn.Linear(ROTMAT_FLAT_DIM*num_joints + JOINT_POS_DIM*num_joints, dim_per_joint)
        dim_feat = JOINT_FEAT_BASE_DIM + dim_per_joint

        layers = []
        for idx in range(num_joints):
            layer = nn.Sequential(nn.Linear(dim_feat, dim_feat), nn.ReLU(), nn.Linear(dim_feat, dim_per_joint))

            layers.append(layer)

        self.layers = nn.ModuleList(layers)

        if out_dim <= 0:
            self.out_layer: nn.Module = nn.Identity()
            self.n_output_dims = num_joints * dim_per_joint
        else:
            self.out_layer = nn.Linear(num_joints * dim_per_joint, out_dim)
            self.n_output_dims = out_dim

    def forward(
        self, rots: torch.Tensor, Jtrs: torch.Tensor, skinning_weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = rots.size(0)

        if self.rel_joints:
            with torch.no_grad():
                Jtrs_rel = Jtrs.clone()
                Jtrs_rel[:, 1:, :] = Jtrs_rel[:, 1:, :] - Jtrs_rel[:, self.ktree_parents[1:], :]
                Jtrs = Jtrs_rel.clone()

        global_feat = torch.cat([rots.view(batch_size, -1), Jtrs.view(batch_size, -1)], dim=-1)
        global_feat = self.layer_0(global_feat)
        # global_feat = (self.layer_0.weight@global_feat[0]+self.layer_0.bias)[None]
        joint_feats: list[torch.Tensor] = [torch.empty(0)] * self.num_joints
        for j_idx in range(self.num_joints):
            rot = rots[:, j_idx, :]
            Jtr = Jtrs[:, j_idx, :]
            parent = self.ktree_parents[j_idx]
            if parent == -1:
                bone_l = torch.norm(Jtr, dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, global_feat], dim=-1)
                joint_feats[j_idx] = self.layers[j_idx](in_feat)
            else:
                parent_feat = joint_feats[parent]
                bone_l = torch.norm(Jtr if self.rel_joints else Jtr - Jtrs[:, parent, :], dim=-1, keepdim=True)
                in_feat = torch.cat([rot, Jtr, bone_l, parent_feat], dim=-1)
                joint_feats[j_idx] = self.layers[j_idx](in_feat)

        out = torch.cat(joint_feats, dim=-1)
        out_final: torch.Tensor = self.out_layer(out)
        return out_final

class VanillaCondMLP(nn.Module):
    def __init__(self, dim_in: int, dim_cond: int, dim_out: int, config: DictConfig, dim_coord: int = 3) -> None:
        super(VanillaCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
        if config.multires > 0:
            embed_fn, input_ch = get_embedder(config.multires, input_dims=dim_in)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.last_layer_init = config.last_layer_init

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if self.last_layer_init and l == self.num_layers - 2:
                torch.nn.init.normal_(lin.weight, mean=0., std=LAST_LAYER_INIT_STD)
                torch.nn.init.constant_(lin.bias, val=0.)


            setattr(self, "lin" + str(l), lin)

        self.activation = nn.LeakyReLU()

    def forward(self, coords: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        coords_embedded: torch.Tensor
        if self.embed_fn is not None:
            coords_embedded = self.embed_fn(coords)
        else:
            coords_embedded = coords

        x: torch.Tensor = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                assert cond is not None
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) * SKIP_CONNECTION_SCALE

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def get_skinning_mlp(n_input_dims: int, n_output_dims: int, config: DictConfig) -> VanillaCondMLP:
    if config.otype == 'VanillaMLP':
        network = VanillaCondMLP(n_input_dims, 0, n_output_dims, config)
    else:
        raise ValueError

    return network


class HannwCondMLP(nn.Module):
    def __init__(self, dim_in: int, dim_cond: int, dim_out: int, config: DictConfig, dim_coord: int = 3) -> None:
        super(HannwCondMLP, self).__init__()

        self.n_input_dims = dim_in
        self.n_output_dims = dim_out

        self.n_neurons, self.n_hidden_layers = config.n_neurons, config.n_hidden_layers

        self.config = config
        dims = [dim_in] + [self.n_neurons for _ in range(self.n_hidden_layers)] + [dim_out]

        self.embed_fn = None
        if config.multires > 0:
            _, input_ch = get_hannw_embedder(config.embedder, config.multires, 0)
            dims[0] = input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in config.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            if l in config.cond_in:
                lin = nn.Linear(dims[l] + dim_cond, out_dim)
            else:
                lin = nn.Linear(dims[l], out_dim)

            if l in config.cond_in:
                # Conditional input layer initialization
                torch.nn.init.constant_(lin.weight[:, -dim_cond:], 0.0)
            torch.nn.init.constant_(lin.bias, 0.0)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.ReLU()

    def forward(self, coords: torch.Tensor, iteration: int, cond: torch.Tensor | None = None) -> torch.Tensor:
        if cond is not None:
            cond = cond.expand(coords.shape[0], -1)

        coords_embedded: torch.Tensor
        if self.config.multires > 0:
            embed_fn, _ = get_hannw_embedder(self.config.embedder, self.config.multires, iteration)
            coords_embedded = embed_fn(coords)
        else:
            coords_embedded = coords

        x: torch.Tensor = coords_embedded
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.config.cond_in:
                assert cond is not None
                x = torch.cat([x, cond], 1)

            if l in self.config.skip_in:
                x = torch.cat([x, coords_embedded], 1) * SKIP_CONNECTION_SCALE

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x

def config_to_primitive(config: DictConfig, resolve: bool = True) -> ConfigPrimitive:
    return cast(ConfigPrimitive, OmegaConf.to_container(config, resolve=resolve))

class HashGrid(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        xL = config.max_resolution
        if xL > 0:
            L = config.n_levels
            x0 = config.base_resolution
            config.per_level_scale = float(np.exp(np.log(xL / x0) / (L - 1)))
        self.encoding = tcnn.Encoding(3, config_to_primitive(config))
        self.n_output_dims = self.encoding.n_output_dims
        self.n_input_dims = self.encoding.n_input_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1.) * 0.5 # [-1, 1] => [0, 1]

        out: torch.Tensor = self.encoding(x)
        return out
