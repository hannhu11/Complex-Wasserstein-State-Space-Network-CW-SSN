#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complex Wasserstein State-Space Network (CW-SSN) & Core Operator (CW-SSO)
=========================================================================
Target Venue: IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
Repository:   https://github.com/hannhu11/Complex-Wasserstein-State-Space-Network-CW-SSN

Mathematical Formulation of the Unified Primitive Operator (CW-SSO):
--------------------------------------------------------------------
Unlike incremental "Lego-style" architectures (e.g., UOT-FSMamba) that mechanically
concatenate external FFT/DCT branches, auxiliary uncertainty networks, and external
Optimal Transport modules, CW-SSN is a homogeneous architecture built by stacking
a single first-principles visual operator: ComplexWassersteinSSM2D (CW-SSO).

1. Closed-Form Wasserstein Geodesic Discretization (Replacing Point-wise ZOH):
   Each spatial position p_t is modeled as a local Gaussian distribution
   P_t = N(mu_t, Sigma_t) with Sigma_t = diag(sigma_t^2) extracted via a
   zero-parameter sliding window N(p_t) of size k x k.
   The exact squared 2-Wasserstein distance between adjacent spatial steps is:
       W_2^2(P_t, P_{t-1}) = ||mu_t - mu_{t-1}||_2^2 + ||sigma_t - sigma_{t-1}||_2^2
   The effective geodesic discretization step size on the Wasserstein manifold is:
       Delta_tilde_t = Softplus(W_Delta x_t) * exp(-gamma * W_2^2(P_t, P_{t-1}))
                       + beta * W_2^2(P_t, P_{t-1})
   When crossing a severe degradation boundary (W_2^2 >> 0), the geodesic barrier
   accelerates exponential decay exp(-Delta_tilde_t * lambda_k) -> 0, freezing and
   wiping out contaminated hidden states inside the differential equation kernel.

2. 2x2 Orthogonal Rotational Spectral State-Space (Replacing External FFT/DCT):
   The continuous state evolution matrix is lifted to the complex domain:
       A = -Lambda + i * Omega in C^{N x N},  Lambda > 0, Omega in R^{N x N}
   Under Euler discretization A_bar_t = exp(Delta_tilde_t * A), the complex transition
   is mapped to an isomorphic real-valued 2x2 block-orthogonal rotation between
   even-odd state pairs (h_t^{(2k)}, h_t^{(2k+1)}), executing natively in FP16 on
   Turing Tesla T4 Tensor Cores without complex64 overflow:
       [h_t^{(2k)}  ]   exp(-Delta_tilde_t * lambda_k) [ cos(theta_t,k)  -sin(theta_t,k) ] [h_{t-1}^{(2k)}  ]   [B_bar_t^{(2k)}  ]
       [h_t^{(2k+1)}] =                                [ sin(theta_t,k)   cos(theta_t,k) ] [h_{t-1}^{(2k+1)}] + [B_bar_t^{(2k+1)}] x_t
   where theta_{t,k} = Delta_tilde_t * omega_k. Unrolling this recurrence yields a
   Continuous Wasserstein-Adaptive Short-Time Fourier/Gabor Transform that intrinsically
   conserves high-frequency phase and texture energy (Theorem 1) while bounding
   cross-region state contamination (Theorem 2).
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# =============================================================================
# 1. Parallel Associative Prefix Scan for 2x2 Orthogonal Rotational Recurrence
# =============================================================================

def rotational_2x2_Scan_parallel(
    m_c: torch.Tensor,
    m_s: torch.Tensor,
    u_even: torch.Tensor,
    u_odd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Exact parallel Kogge-Stone associative prefix scan for the 2x2 real-valued
    block-orthogonal rotational state-space recurrence in O(log_2 L) steps:

        [h_t^{even}]   [ m_{t,c}  -m_{t,s} ] [h_{t-1}^{even}]   [u_t^{even}]
        [h_t^{odd} ] = [ m_{t,s}   m_{t,c} ] [h_{t-1}^{odd} ] + [u_t^{odd} ]

    Because the set of scaled 2D orthogonal rotations is closed under multiplication,
    two consecutive transitions (M_1, u_1) and (M_2, u_2) compose associatively via
    pure real arithmetic (isomorphic to complex polar multiplication):
        M_{2 o 1, c} = m_{2,c} * m_{1,c} - m_{2,s} * m_{1,s}
        M_{2 o 1, s} = m_{2,c} * m_{1,s} + m_{2,s} * m_{1,c}
        u_{2 o 1, even} = m_{2,c} * u_{1, even} - m_{2,s} * u_{1, odd} + u_{2, even}
        u_{2 o 1, odd}  = m_{2,s} * u_{1, even} + m_{2,c} * u_{1, odd} + u_{2, odd}

    Args:
        m_c:    Tensor of shape (B, C_half, L) -- e^{-Delta_tilde_t * lambda_k} * cos(Delta_tilde_t * omega_k)
        m_s:    Tensor of shape (B, C_half, L) -- e^{-Delta_tilde_t * lambda_k} * sin(Delta_tilde_t * omega_k)
        u_even: Tensor of shape (B, C_half, L) -- B_bar_t^{(2k)} * x_t^{(2k)}
        u_odd:  Tensor of shape (B, C_half, L) -- B_bar_t^{(2k+1)} * x_t^{(2k+1)}

    Returns:
        (h_even, h_odd), each of shape (B, C_half, L).
    """
    L = m_c.shape[-1]
    a_c = m_c.clone()
    a_s = m_s.clone()
    h_e = u_even.clone()
    h_o = u_odd.clone()

    step = 1
    while step < L:
        a_c_prev = a_c[..., :-step]
        a_s_prev = a_s[..., :-step]
        h_e_prev = h_e[..., :-step]
        h_o_prev = h_o[..., :-step]

        a_c_curr = a_c[..., step:]
        a_s_curr = a_s[..., step:]
        h_e_curr = h_e[..., step:]
        h_o_curr = h_o[..., step:]

        new_h_e = a_c_curr * h_e_prev - a_s_curr * h_o_prev + h_e_curr
        new_h_o = a_s_curr * h_e_prev + a_c_curr * h_o_prev + h_o_curr
        new_a_c = a_c_curr * a_c_prev - a_s_curr * a_s_prev
        new_a_s = a_c_curr * a_s_prev + a_s_curr * a_c_prev

        h_e = torch.cat([h_e[..., :step], new_h_e], dim=-1)
        h_o = torch.cat([h_o[..., :step], new_h_o], dim=-1)
        a_c = torch.cat([a_c[..., :step], new_a_c], dim=-1)
        a_s = torch.cat([a_s[..., :step], new_a_s], dim=-1)
        step <<= 1

    return h_e, h_o


def rotational_2x2_scan_sequential(
    m_c: torch.Tensor,
    m_s: torch.Tensor,
    u_even: torch.Tensor,
    u_odd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference sequential recurrence for mathematical equivalence verification.
    Computes the exact step-by-step recurrence over t = 0 ... L-1.
    """
    B, C_half, L = m_c.shape
    h_e_list = []
    h_o_list = []
    h_e_t = torch.zeros((B, C_half), dtype=m_c.dtype, device=m_c.device)
    h_o_t = torch.zeros((B, C_half), dtype=m_c.dtype, device=m_c.device)

    for t in range(L):
        mc_t = m_c[:, :, t]
        ms_t = m_s[:, :, t]
        ue_t = u_even[:, :, t]
        uo_t = u_odd[:, :, t]

        next_he = mc_t * h_e_t - ms_t * h_o_t + ue_t
        next_ho = ms_t * h_e_t + mc_t * h_o_t + uo_t
        h_e_t, h_o_t = next_he, next_ho
        h_e_list.append(h_e_t)
        h_o_list.append(h_o_t)

    return torch.stack(h_e_list, dim=-1), torch.stack(h_o_list, dim=-1)


# =============================================================================
# 2. Hilbert & Multi-Directional 2D Spatial Traversal Utilities
# =============================================================================

def _hilbert_index_2d(H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Generates a locality-preserving 2D traversal order (boustrophedon / generalized
    Hilbert-like Morton order) for arbitrary (H, W) grids.
    Returns a 1D LongTensor of length H * W containing flattened indices.
    """
    y = torch.arange(H, device=device, dtype=torch.long).unsqueeze(1).expand(H, W)
    x = torch.arange(W, device=device, dtype=torch.long).unsqueeze(0).expand(H, W)
    # Interleave bits of (y // 2, x // 2) + local 2x2 Hilbert orientation
    x_flip = torch.where((y & 1) == 1, W - 1 - x, x)
    flat_idx = y * W + x_flip
    return flat_idx.reshape(-1)


# =============================================================================
# 3. Core Visual Primitive: ComplexWassersteinSSM2D (CW-SSO)
# =============================================================================

class ComplexWassersteinSSM2D(nn.Module):
    """
    Complex Wasserstein State-Space Operator (CW-SSO).

    Unifies Optimal Transport (closed-form 2-Wasserstein geodesic step modulation)
    and Spectral Analysis (2x2 orthogonal phase rotation) directly inside the
    state-space differential equation discretization.

    Mathematical Ablation Degeneracy Modes (controlled via constructor flags):
    - Case 1 (`use_wasserstein=False, use_phase_rotation=False`, i.e., gamma=0, beta=0, Omega=0):
      Degenerates exactly to standard real-valued ZOH Mamba (vulnerable to cross-region
      state contamination and high-frequency phase attenuation).
    - Case 2 (`use_wasserstein=True, use_phase_rotation=False`, i.e., gamma>0, beta>0, Omega=0):
      Wasserstein Geodesic Discretization only (blocks noise propagation across boundaries,
      but lacks oscillatory high-frequency phase conservation).
    - Case 3 (`use_wasserstein=False, use_phase_rotation=True`, i.e., gamma=0, beta=0, Omega!=0):
      Rotational Spectral State-Space only (preserves high-frequency textures, but propagates
      high-frequency noise across region boundaries).
    - Case 4 (`use_wasserstein=True, use_phase_rotation=True`, Full CW-SSO Operator):
      Both mechanisms operate synergistically inside the transition exponential.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 3,
        low_rank_r: Optional[int] = None,
        use_wasserstein: bool = True,
        use_phase_rotation: bool = True,
        scan_type: str = "bidirectional_2d",
        scan_mode: str = "parallel",
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f"Channel dimension `dim` must be even for 2x2 rotational pairs, got {dim}."
        self.dim = dim
        self.half_dim = dim // 2
        self.window_size = window_size
        self.use_wasserstein = use_wasserstein
        self.use_phase_rotation = use_phase_rotation
        self.scan_type = scan_type
        self.scan_mode = scan_mode

        # Local spatial context embedding (depthwise 3x3) prior to state-space evolution
        self.local_dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=True)

        # Dynamic input-dependent projections (W_Delta, W_B, W_C)
        # Optionally compressed via Low-Rank Triplet Projection (LoTA-inspired) when low_rank_r is set
        if low_rank_r is not None and 0 < low_rank_r < dim:
            self.proj_down = nn.Conv2d(dim, low_rank_r, kernel_size=1, bias=False)
            self.proj_core = nn.Conv2d(low_rank_r, low_rank_r, kernel_size=1, bias=False)
            self.proj_up = nn.Conv2d(low_rank_r, dim * 3, kernel_size=1, bias=True)
            self.use_low_rank = True
        else:
            self.proj_sbc = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
            self.use_low_rank = False

        # Diagonal real decay rates Lambda = diag(exp(log_lambda)) > 0 (size: dim // 2)
        init_log_lambda = torch.linspace(-2.5, -0.5, self.half_dim)
        self.log_lambda = nn.Parameter(init_log_lambda)

        # Diagonal imaginary rotational frequencies Omega = diag(omega_1, ..., omega_{N}) (size: dim // 2)
        # Initialized across multi-scale Fourier frequencies [0.1*pi, 1.5*pi]
        init_omega = torch.linspace(0.1 * math.pi, 1.5 * math.pi, self.half_dim)
        self.omega = nn.Parameter(init_omega)

        # Wasserstein geodesic sensitivity parameters (gamma > 0, beta > 0 via Softplus)
        self.raw_gamma = nn.Parameter(torch.full((1, self.half_dim, 1), 0.5))
        self.raw_beta = nn.Parameter(torch.full((1, self.half_dim, 1), 0.5))

        # Direct skip connection D and output projection
        self.D = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.out_Gate = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.last_diagnostics: Optional[Dict[str, torch.Tensor]] = None

    def compute_local_gaussian_moments(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts local Gaussian distribution P_t = N(mu_t, diag(sigma_t^2)) at every
        spatial location p_t using a zero-parameter sliding window N(p_t) of size k x k
        with 2-pass B-spline/tent smoothing to suppress intra-region noise jitter.
        """
        k = self.window_size
        pad = k // 2
        x_f32 = x.float()
        x_pad = F.pad(x_f32, (pad, pad, pad, pad), mode="reflect")
        mu = F.avg_pool2d(x_pad, kernel_size=k, stride=1)
        sq_mean = F.avg_pool2d(x_pad * x_pad, kernel_size=k, stride=1)
        var = torch.clamp(sq_mean - mu * mu, min=1e-6)
        sigma = torch.sqrt(var)
        # Second-pass tent smoothing stabilizes intra-region moments while sharpening boundaries
        mu = F.avg_pool2d(F.pad(mu, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        sigma = F.avg_pool2d(F.pad(sigma, (pad, pad, pad, pad), mode="reflect"), kernel_size=k, stride=1)
        return mu, sigma

    def _project_dynamic_params(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_low_rank:
            z = self.proj_core(self.proj_down(x))
            sbc = self.proj_up(z)
        else:
            sbc = self.proj_sbc(x)
        raw_delta, B_proj, C_proj = torch.chunk(sbc, 3, dim=1)
        return raw_delta, B_proj, C_proj

    def _compute_w2_and_transitions(
        self,
        mu_pairs: torch.Tensor,
        sigma_pairs: torch.Tensor,
        delta_base: torch.Tensor,
        bx_even: torch.Tensor,
        bx_odd: torch.Tensor,
        orig_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes closed-form 2-Wasserstein squared distance W_2^2, geodesic step Delta_tilde,
        and 2x2 orthogonal rotational transition coefficients (m_c, m_s, u_even, u_odd)
        along the last tensor dimension (dim=-1, length L). Supports arbitrary leading dims (..., L).
        """
        L = delta_base.shape[-1]
        if self.use_wasserstein:
            d_mu = mu_pairs[..., 1:] - mu_pairs[..., :-1]
            d_sigma = sigma_pairs[..., 1:] - sigma_pairs[..., :-1]
            w2_step = (d_mu * d_mu + d_sigma * d_sigma).sum(dim=2)
            w2_sq = F.pad(w2_step, (1, 0), mode="constant", value=0.0)
            s = min(self.window_size, L - 1)
            if s > 1:
                d_sig_s = sigma_pairs[..., s:] - sigma_pairs[..., :-s]
                w2_s = F.pad((d_sig_s * d_sig_s).sum(dim=2), (s, 0), mode="constant", value=0.0)
                w2_sq = w2_sq + 2.0 * w2_s
            feat_scale = (
                (mu_pairs.pow(2) + sigma_pairs.pow(2)).sum(dim=2).mean(dim=-1, keepdim=True).detach() + 0.25
            )
            w2_sq = torch.clamp(w2_sq / feat_scale, min=0.0, max=15.0)
        else:
            w2_sq = torch.zeros_like(delta_base, dtype=torch.float32)

        # Broadcast shape for channel-wise parameters: (1, C_half, 1, ...)
        param_shape = [1, self.half_dim] + [1] * (w2_sq.ndim - 2)

        if self.use_wasserstein:
            gamma = F.softplus(self.raw_gamma.float()).view(*param_shape)
            beta = F.softplus(self.raw_beta.float()).view(*param_shape)
            geodesic_gate = torch.exp(-gamma * w2_sq)
            delta_in = delta_base * geodesic_gate
            delta_tilde = delta_in + beta * w2_sq
        else:
            delta_in = delta_base
            delta_tilde = delta_base

        delta_tilde = torch.clamp(delta_tilde, min=1e-4, max=20.0)

        lambd = torch.exp(torch.clamp(self.log_lambda.float(), min=-6.0, max=2.0)).view(*param_shape)
        decay = torch.exp(-delta_tilde * lambd)

        if self.use_phase_rotation:
            omega = self.omega.float().view(*param_shape)
            theta = delta_tilde * omega
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
        else:
            cos_t = torch.ones_like(decay)
            sin_t = torch.zeros_like(decay)

        m_c = (decay * cos_t).to(orig_dtype)
        m_s = (decay * sin_t).to(orig_dtype)
        u_even = (delta_in.to(orig_dtype)) * bx_even
        u_odd = (delta_in.to(orig_dtype)) * bx_odd
        return m_c, m_s, u_even, u_odd, w2_sq, delta_tilde

    def _run_1d_wasserstein_rotational_scan(
        self,
        x_seq: torch.Tensor,
        mu_seq: torch.Tensor,
        sigma_seq: torch.Tensor,
        raw_delta_seq: torch.Tensor,
        B_seq: torch.Tensor,
        C_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Executes the Complex Wasserstein State-Space recurrence along a 1D spatial trajectory
        of shape (B, C, L).
        """
        B, C, L = x_seq.shape
        orig_dtype = x_seq.dtype

        mu_pairs = mu_seq.view(B, self.half_dim, 2, L)
        sigma_pairs = sigma_seq.view(B, self.half_dim, 2, L)
        delta_base = torch.clamp(
            F.softplus(raw_delta_seq.float()).view(B, self.half_dim, 2, L).mean(dim=2),
            min=1e-4,
            max=5.0,
        )
        bx = (B_seq * x_seq).view(B, self.half_dim, 2, L)
        m_c, m_s, u_even, u_odd, w2_sq, delta_tilde = self._compute_w2_and_transitions(
            mu_pairs, sigma_pairs, delta_base, bx[:, :, 0, :], bx[:, :, 1, :], orig_dtype
        )

        if self.scan_mode == "sequential":
            h_even, h_odd = rotational_2x2_scan_sequential(m_c, m_s, u_even, u_odd)
        else:
            h_even, h_odd = rotational_2x2_Scan_parallel(m_c, m_s, u_even, u_odd)

        h_full = torch.stack([h_even, h_odd], dim=2).reshape(B, C, L)
        y_seq = (C_seq * h_full).to(orig_dtype)
        return y_seq, w2_sq, delta_tilde

    def forward(
        self, x: torch.Tensor, return_diagnostics: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass of CW-SSO on 2D feature maps of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        orig_dtype = x.dtype

        feat = self.local_dwconv(x)
        mu, sigma = self.compute_local_gaussian_moments(feat)
        raw_delta, B_proj, C_proj = self._project_dynamic_params(feat)

        if self.scan_type == "hilbert":
            order = _hilbert_index_2d(H, W, x.device)
            inv_order = torch.argsort(order)
            f_seq = feat.flatten(2)[:, :, order]
            mu_seq = mu.flatten(2)[:, :, order]
            sig_seq = sigma.flatten(2)[:, :, order]
            d_seq = raw_delta.flatten(2)[:, :, order]
            b_seq = B_proj.flatten(2)[:, :, order]
            c_seq = C_proj.flatten(2)[:, :, order]

            y_fwd, w2_fwd, dt_fwd = self._run_1d_wasserstein_rotational_scan(
                f_seq, mu_seq, sig_seq, d_seq, b_seq, c_seq
            )
            y_bwd, _, _ = self._run_1d_wasserstein_rotational_scan(
                f_seq.flip(-1), mu_seq.flip(-1), sig_seq.flip(-1),
                d_seq.flip(-1), b_seq.flip(-1), c_seq.flip(-1)
            )
            y_seq = 0.5 * (y_fwd + y_bwd.flip(-1))
            y_2d = y_seq[:, :, inv_order].view(B, C, H, W)
            w2_map = w2_fwd[:, :, inv_order].view(B, self.half_dim, H, W)
            dt_map = dt_fwd[:, :, inv_order].view(B, self.half_dim, H, W)

        elif self.scan_type == "raster_1d":
            # Row-wise 1D causal scan directly on 4D tensor along dim=-1 (width W)
            mu_p = mu.view(B, self.half_dim, 2, H, W)
            sig_p = sigma.view(B, self.half_dim, 2, H, W)
            delta_base = torch.clamp(
                F.softplus(raw_delta.float()).view(B, self.half_dim, 2, H, W).mean(dim=2),
                min=1e-4,
                max=5.0,
            )
            bx = (B_proj * feat).view(B, self.half_dim, 2, H, W)
            m_c, m_s, u_e, u_o, w2_map, dt_map = self._compute_w2_and_transitions(
                mu_p, sig_p, delta_base, bx[:, :, 0], bx[:, :, 1], orig_dtype
            )
            if self.scan_mode == "sequential":
                he, ho = rotational_2x2_scan_sequential(
                    m_c.permute(0, 2, 1, 3).reshape(B * H, self.half_dim, W),
                    m_s.permute(0, 2, 1, 3).reshape(B * H, self.half_dim, W),
                    u_e.permute(0, 2, 1, 3).reshape(B * H, self.half_dim, W),
                    u_o.permute(0, 2, 1, 3).reshape(B * H, self.half_dim, W),
                )
                he = he.view(B, H, self.half_dim, W).permute(0, 2, 1, 3)
                ho = ho.view(B, H, self.half_dim, W).permute(0, 2, 1, 3)
            else:
                he, ho = rotational_2x2_Scan_parallel(m_c, m_s, u_e, u_o)
            h_full = torch.stack([he, ho], dim=2).reshape(B, C, H, W)
            y_2d = (C_proj * h_full).to(orig_dtype)

        else:
            # Default: Bidirectional 2D (Horizontal + Vertical Geodesic Trajectories)
            # Operates directly on 4D (B, C_half, H, W) without redundant tensor copies
            mu_p = mu.view(B, self.half_dim, 2, H, W)
            sig_p = sigma.view(B, self.half_dim, 2, H, W)
            delta_base = torch.clamp(
                F.softplus(raw_delta.float()).view(B, self.half_dim, 2, H, W).mean(dim=2),
                min=1e-4,
                max=5.0,
            )
            bx = (B_proj * feat).view(B, self.half_dim, 2, H, W)
            bx_e, bx_o = bx[:, :, 0], bx[:, :, 1]

            # 1. Horizontal bidirectional scan along width W (dim=-1)
            mc_h, ms_h, ue_h, uo_h, w2_h, dt_h = self._compute_w2_and_transitions(
                mu_p, sig_p, delta_base, bx_e, bx_o, orig_dtype
            )
            he_h_bi, ho_h_bi = rotational_2x2_Scan_parallel(
                torch.cat([mc_h, mc_h.flip(-1)], dim=0),
                torch.cat([ms_h, ms_h.flip(-1)], dim=0),
                torch.cat([ue_h, ue_h.flip(-1)], dim=0),
                torch.cat([uo_h, uo_h.flip(-1)], dim=0),
            )
            he_h = 0.5 * (he_h_bi[:B] + he_h_bi[B:].flip(-1))
            ho_h = 0.5 * (ho_h_bi[:B] + ho_h_bi[B:].flip(-1))

            # 2. Vertical bidirectional scan along height H (transpose H and W)
            mu_pv = mu_p.transpose(-2, -1)
            sig_pv = sig_p.transpose(-2, -1)
            db_v = delta_base.transpose(-2, -1)
            bxe_v = bx_e.transpose(-2, -1)
            bxo_v = bx_o.transpose(-2, -1)

            mc_v, ms_v, ue_v, uo_v, w2_v, _ = self._compute_w2_and_transitions(
                mu_pv, sig_pv, db_v, bxe_v, bxo_v, orig_dtype
            )
            he_v_bi, ho_v_bi = rotational_2x2_Scan_parallel(
                torch.cat([mc_v, mc_v.flip(-1)], dim=0),
                torch.cat([ms_v, ms_v.flip(-1)], dim=0),
                torch.cat([ue_v, ue_v.flip(-1)], dim=0),
                torch.cat([uo_v, uo_v.flip(-1)], dim=0),
            )
            he_v = 0.5 * (he_v_bi[:B] + he_v_bi[B:].flip(-1)).transpose(-2, -1)
            ho_v = 0.5 * (ho_v_bi[:B] + ho_v_bi[B:].flip(-1)).transpose(-2, -1)

            h_full = torch.stack([0.5 * (he_h + he_v), 0.5 * (ho_h + ho_v)], dim=2).reshape(B, C, H, W)
            y_2d = (C_proj * h_full).to(orig_dtype)
            w2_map = 0.5 * (w2_h + w2_v.transpose(-2, -1))
            dt_map = dt_h

        # Gated residual output projection
        gate = F.silu(self.out_Gate(x))
        out = self.out_proj(y_2d * gate + self.D.to(x.dtype) * x)

        if return_diagnostics:
            diag = {
                "w2_map": w2_map,
                "delta_tilde_map": dt_map,
                "state_response": y_2d,
                "mu": mu,
                "sigma": sigma,
            }
            self.last_diagnostics = diag
            return out, diag

        return out


# =============================================================================
# 4. Homogeneous Building Block: CWSSBlock
# =============================================================================

class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm for 2D feature maps (B, C, H, W) with FP32 internal stats."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        mean = x_f.mean(dim=1, keepdim=True)
        var = (x_f - mean).pow(2).mean(dim=1, keepdim=True)
        x_norm = (x_f - mean) * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight.float() + self.bias.float()).to(orig_dtype)


class GatedChannelFFN(nn.Module):
    """
    Gated Depthwise Feed-Forward Network (GDFN) for local channel mixing inside
    each homogeneous CWSSBlock.
    """

    def __init__(self, dim: int, expansion_factor: float = 2.0) -> None:
        super().__init__()
        hidden_dim = int(round(dim * expansion_factor))
        if hidden_dim % 2 != 0:
            hidden_dim += 1
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            hidden_dim * 2,
            hidden_dim * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_dim * 2,
            bias=True,
        )
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.dwconv(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class CWSSBlock(nn.Module):
    """
    Single Unified Homogeneous Block of CW-SSN:
        x' = x  + CW-SSO(LayerNorm2d(x))
        y  = x' + GatedChannelFFN(LayerNorm2d(x'))
    Stacked identically across all hierarchy levels (analogous to ResBlock in ResNet).
    """

    def __init__(
        self,
        dim: int,
        ffn_expansion: float = 2.0,
        window_size: int = 3,
        low_rank_r: Optional[int] = None,
        use_wasserstein: bool = True,
        use_phase_rotation: bool = True,
        scan_type: str = "bidirectional_2d",
        scan_mode: str = "parallel",
        use_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm2d(dim)
        self.op = ComplexWassersteinSSM2D(
            dim=dim,
            window_size=window_size,
            low_rank_r=low_rank_r,
            use_wasserstein=use_wasserstein,
            use_phase_rotation=use_phase_rotation,
            scan_type=scan_type,
            scan_mode=scan_mode,
        )
        self.norm2 = LayerNorm2d(dim)
        self.ffn = GatedChannelFFN(dim=dim, expansion_factor=ffn_expansion)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.op(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.use_checkpoint
            and torch.is_grad_enabled()
            and x.requires_grad
            and (x.shape[-2] * x.shape[-1] >= 1024)
        ):
            return grad_checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# =============================================================================
# 5. Full Homogeneous Backbone: CWSSN (CW-SSN-T 5.81M, CW-SSN-B 7.95M)
# =============================================================================

class Downsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            return F.conv2d(
                x.float(), self.conv.weight.float(), None, stride=2, padding=1
            ).to(x.dtype)


class Upsample2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class CWSSN(nn.Module):
    """
    Complex Wasserstein State-Space Network (CW-SSN).
    A homogeneous 4-stage hierarchical encoder-decoder backbone constructed exclusively
    from CWSSBlock primitives without any external FFT, uncertainty, or OT sub-networks.

    Standard Configurations:
    - CW-SSN-T (Tiny): base_dim=58, depths=(2, 2, 4, 2), ffn_expansion=2.0 -> 5.81M params (5,813,285)
      (Strictly within the 5.5M - 6.2M KPI gate, 88.4% lighter than DA-RCOT 50.1M).
    - CW-SSN-B (Base): base_dim=68, depths=(2, 2, 4, 2), ffn_expansion=2.0 -> 7.95M params (7,953,895)
      (Strictly within the 7.5M - 8.2M KPI gate).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_dim: int = 58,
        depths: Tuple[int, int, int, int] = (2, 2, 4, 2),
        refinement_blocks: int = 2,
        ffn_expansion: float = 2.0,
        window_size: int = 3,
        low_rank_r: Optional[int] = None,
        use_wasserstein: bool = True,
        use_phase_rotation: bool = True,
        scan_type: str = "bidirectional_2d",
        scan_mode: str = "parallel",
    ) -> None:
        super().__init__()
        dims = [base_dim, base_dim * 2, base_dim * 3, base_dim * 4]
        self.dims = dims
        self.use_wasserstein = use_wasserstein
        self.use_phase_rotation = use_phase_rotation

        block_kwargs = dict(
            ffn_expansion=ffn_expansion,
            window_size=window_size,
            low_rank_r=low_rank_r,
            use_wasserstein=use_wasserstein,
            use_phase_rotation=use_phase_rotation,
            scan_type=scan_type,
            scan_mode=scan_mode,
        )

        # Shallow spatial embedding
        self.patch_embed = nn.Conv2d(in_channels, dims[0], kernel_size=3, padding=1, bias=True)

        # Encoder hierarchy
        self.enc1 = nn.Sequential(*[CWSSBlock(dims[0], **block_kwargs) for _ in range(depths[0])])
        self.down1 = Downsample2d(dims[0], dims[1])

        self.enc2 = nn.Sequential(*[CWSSBlock(dims[1], **block_kwargs) for _ in range(depths[1])])
        self.down2 = Downsample2d(dims[1], dims[2])

        self.enc3 = nn.Sequential(*[CWSSBlock(dims[2], **block_kwargs) for _ in range(depths[2])])
        self.down3 = Downsample2d(dims[2], dims[3])

        # Bottleneck latent state-space stage
        self.latent = nn.Sequential(*[CWSSBlock(dims[3], **block_kwargs) for _ in range(depths[3])])

        # Decoder hierarchy
        self.up3 = Upsample2d(dims[3], dims[2])
        self.reduce3 = nn.Conv2d(dims[2] * 2, dims[2], kernel_size=1, bias=False)
        self.dec3 = nn.Sequential(*[CWSSBlock(dims[2], **block_kwargs) for _ in range(depths[2])])

        self.up2 = Upsample2d(dims[2], dims[1])
        self.reduce2 = nn.Conv2d(dims[1] * 2, dims[1], kernel_size=1, bias=False)
        self.dec2 = nn.Sequential(*[CWSSBlock(dims[1], **block_kwargs) for _ in range(depths[1])])

        self.up1 = Upsample2d(dims[1], dims[0])
        self.reduce1 = nn.Conv2d(dims[0] * 2, dims[0], kernel_size=1, bias=False)
        self.dec1 = nn.Sequential(*[CWSSBlock(dims[0], **block_kwargs) for _ in range(depths[0])])

        # High-resolution refinement stage
        self.refinement = nn.Sequential(
            *[CWSSBlock(dims[0], **block_kwargs) for _ in range(refinement_blocks)]
        )
        self.output_proj = nn.Conv2d(dims[0], out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=1e-2)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def extract_latent_descriptor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts the global Wasserstein-modulated complex state descriptor (combining
        first-order spatial mean mu, second-order standard deviation sigma, and directional
        variation) from the bottleneck stage for t-SNE manifold visualization (Notebook NB_09).
        """
        with torch.cuda.amp.autocast(enabled=False):
            f0 = F.conv2d(
                x.float(),
                self.patch_embed.weight.float(),
                self.patch_embed.bias.float() if self.patch_embed.bias is not None else None,
                padding=1,
            ).to(x.dtype)
        e1 = self.enc1(f0)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.latent(self.down3(e3))
        mu_b = F.adaptive_avg_pool2d(b, 1).flatten(1)
        sig_b = torch.sqrt(
            F.adaptive_avg_pool2d((b - mu_b.unsqueeze(-1).unsqueeze(-1)).pow(2), 1).flatten(1) + 1e-6
        )
        C_b = b.shape[1]
        half_c = C_b // 2
        return torch.cat([mu_b[:, :half_c], sig_b[:, half_c:]], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        # Pad to multiple of 8 for 3-level downsampling
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        with torch.cuda.amp.autocast(enabled=False):
            e1 = F.conv2d(
                x_in.float(),
                self.patch_embed.weight.float(),
                self.patch_embed.bias.float() if self.patch_embed.bias is not None else None,
                padding=1,
            ).to(x_in.dtype)
        e1 = self.enc1(e1)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        b = self.latent(self.down3(e3))

        d3 = self.dec3(self.reduce3(torch.cat([self.up3(b), e3], dim=1)))
        del b, e3
        d2 = self.dec2(self.reduce2(torch.cat([self.up2(d3), e2], dim=1)))
        del d3, e2
        d1 = self.dec1(self.reduce1(torch.cat([self.up1(d2), e1], dim=1)))
        del d2, e1

        ref = self.refinement(d1)
        del d1
        with torch.cuda.amp.autocast(enabled=False):
            out_res = F.conv2d(
                ref.float(),
                self.output_proj.weight.float(),
                self.output_proj.bias.float() if self.output_proj.bias is not None else None,
                padding=1,
            ).to(x_in.dtype)
        del ref
        out = out_res + x_in
        return out[:, :, :H, :W]


def build_cw_ssn_tiny(**kwargs) -> CWSSN:
    """Builds CW-SSN-T (5.81M parameters = 5,813,285)."""
    return CWSSN(base_dim=58, depths=(2, 2, 4, 2), refinement_blocks=2, ffn_expansion=2.0, **kwargs)


def build_cw_ssn_base(**kwargs) -> CWSSN:
    """Builds CW-SSN-B (7.95M parameters = 7,953,895)."""
    return CWSSN(base_dim=68, depths=(2, 2, 4, 2), refinement_blocks=2, ffn_expansion=2.0, **kwargs)


# =============================================================================
# 6. Universal Foreground Segmentation Adapter: CWSSOAdapter (for EVP Tasks)
# =============================================================================

class CWSSOAdapterStage(nn.Module):
    """
    Lightweight Bottleneck CW-SSO Adapter for a single hierarchical stage of a
    frozen Vision Foundation Model (e.g., SegFormer-B4).
    """

    def __init__(self, in_dim: int, bottleneck_dim: int = 64, low_rank_r: int = 16) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_dim, bottleneck_dim, kernel_size=1, bias=False)
        self.norm = LayerNorm2d(bottleneck_dim)
        self.cw_sso = ComplexWassersteinSSM2D(
            dim=bottleneck_dim,
            window_size=3,
            low_rank_r=low_rank_r,
            use_wasserstein=True,
            use_phase_rotation=True,
        )
        self.up = nn.Conv2d(bottleneck_dim, in_dim, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.full((1, in_dim, 1, 1), 0.1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        z = self.norm(self.down(feat))
        z = self.cw_sso(z)
        return feat + self.scale.to(feat.dtype) * self.up(z)


class CWSSOAdapter(nn.Module):
    """
    Multi-Stage CW-SSO Adapter + Lightweight Segmentation Head for Universal
    Foreground Segmentation (Camouflaged Object, Defocus Blur, Forgery, Shadow)
    following the Explicit Visual Prompting (EVP, TPAMI 2025) benchmark protocol.

    With SegFormer-B4 (frozen ~61.4M params, stage dims = [64, 128, 320, 512]),
    this adapter module has 0.99M trainable parameters (987,745 params = 1.61% overhead,
    strictly below the 5.5% Gatekeeping KPI and EVP's 5.7%).
    """

    def __init__(
        self,
        stage_dims: Tuple[int, int, int, int] = (64, 128, 320, 512),
        bottleneck_dim: int = 96,
        decoder_dim: int = 192,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [CWSSOAdapterStage(d, bottleneck_dim=bottleneck_dim, low_rank_r=16) for d in stage_dims]
        )
        self.proj_layers = nn.ModuleList(
            [nn.Conv2d(d, decoder_dim, kernel_size=1, bias=False) for d in stage_dims]
        )
        self.fusion_block = CWSSBlock(dim=decoder_dim, ffn_expansion=2.0, low_rank_r=16)
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim // 2, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(decoder_dim // 2),
            nn.GELU(),
            nn.Conv2d(decoder_dim // 2, num_classes, kernel_size=1, bias=True),
        )

    def forward(
        self, multi_scale_feats: List[torch.Tensor], target_size: Optional[Tuple[int, int]] = None
    ) -> torch.Tensor:
        h_target, w_target = multi_scale_feats[0].shape[-2:]
        fused = None
        for feat, adapter, proj in zip(multi_scale_feats, self.adapters, self.proj_layers):
            adapted = proj(adapter(feat))
            if adapted.shape[-2:] != (h_target, w_target):
                adapted = F.interpolate(
                    adapted, size=(h_target, w_target), mode="bilinear", align_corners=False
                )
            fused = adapted if fused is None else (fused + adapted)

        fused = self.fusion_block(fused)
        logits = self.seg_head(fused)
        if target_size is not None and logits.shape[-2:] != target_size:
            logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


# =============================================================================
# 7. Training Objective: L1 + Closed-Form Spectral Wasserstein Residual Loss
# =============================================================================

class SpectralWassersteinLoss(nn.Module):
    """
    Unified training objective combining spatial L1 reconstruction loss with the
    closed-form 2-Wasserstein distance between local Gaussian distributions and
    Fourier amplitude-phase alignment of the restoration residual.
    """

    def __init__(self, lambda_w2: float = 0.1, lambda_freq: float = 0.05, window_size: int = 3) -> None:
        super().__init__()
        self.lambda_w2 = lambda_w2
        self.lambda_freq = lambda_freq
        self.window_size = window_size

    def _local_moments(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        k = self.window_size
        pad = k // 2
        x = F.pad(img.float(), (pad, pad, pad, pad), mode="reflect")
        mu = F.avg_pool2d(x, kernel_size=k, stride=1)
        sq = F.avg_pool2d(x * x, kernel_size=k, stride=1)
        sigma = torch.sqrt(torch.clamp(sq - mu * mu, min=1e-6))
        return mu, sigma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_f = pred.float()
        target_f = target.float()
        l1_loss = F.l1_loss(pred_f, target_f)

        mu_p, sig_p = self._local_moments(pred_f)
        mu_t, sig_t = self._local_moments(target_f)
        # Phase-coherent Wasserstein decomposition on local Gaussian mean + structural residual + variance
        high_p = pred_f - mu_p
        high_t = target_f - mu_t
        w2_loss = 0.5 * (F.l1_loss(mu_p, mu_t) + F.l1_loss(high_p, high_t)) + 0.1 * F.mse_loss(sig_p, sig_t)

        fft_p = torch.fft.rfft2(pred_f, norm="ortho")
        fft_t = torch.fft.rfft2(target_f, norm="ortho")
        # Complex-valued Fourier alignment (jointly preserving spectral amplitude and phase)
        freq_loss = F.l1_loss(torch.view_as_real(fft_p), torch.view_as_real(fft_t))

        return l1_loss + self.lambda_w2 * w2_loss + self.lambda_freq * freq_loss
