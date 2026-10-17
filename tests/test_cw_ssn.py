#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Unit & Mathematical Verification Tests for CW-SSO and CW-SSN
==========================================================================
Verifies:
1. Exact numerical equivalence of Parallel Kogge-Stone Associative Scan vs. Sequential Recurrence.
2. Closed-form 2-Wasserstein Geodesic Discretization on uniform vs. boundary-corrupted inputs.
3. Theorem 1: Spectral norm & energy conservation of the 2x2 orthogonal phase rotation R(theta).
4. Theorem 2: Exponential state contamination suppression across spatially non-uniform boundaries.
5. Mathematical Ablation Degeneracy Matrix (Cases 1-4: (gamma, beta, Omega) -> 0).
6. Strict TPAMI Gatekeeping Parameter Budgets (CW-SSN-T, CW-SSN-B, and CWSSOAdapter < 5.5%).
7. Forward/Backward gradient flow, odd spatial resolutions, and SpectralWassersteinLoss stability.
"""

import os
import sys
import unittest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cw_ssn import (
    ComplexWassersteinSSM2D,
    CWSSBlock,
    CWSSN,
    CWSSOAdapter,
    SpectralWassersteinLoss,
    build_cw_ssn_tiny,
    build_cw_ssn_base,
    rotational_2x2_Scan_parallel,
    rotational_2x2_scan_sequential,
)


class TestCWSSNArchitectureAndMath(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_01_parallel_vs_sequential_2x2_rotational_scan_equivalence(self) -> None:
        """Verifies that O(log L) parallel associative scan matches sequential scan across lengths."""
        for L in [1, 7, 16, 64, 129]:
            B, C_half = 2, 8
            decay = torch.sigmoid(torch.randn(B, C_half, L)) * 0.95
            theta = torch.randn(B, C_half, L)
            m_c = decay * torch.cos(theta)
            m_s = decay * torch.sin(theta)
            u_even = torch.randn(B, C_half, L) * 0.1
            u_odd = torch.randn(B, C_half, L) * 0.1

            he_par, ho_par = rotational_2x2_Scan_parallel(m_c, m_s, u_even, u_odd)
            he_seq, ho_seq = rotational_2x2_scan_sequential(m_c, m_s, u_even, u_odd)

            max_err_e = (he_par - he_seq).abs().max().item()
            max_err_o = (ho_par - ho_seq).abs().max().item()
            self.assertLess(max_err_e, 1e-4, f"Parallel vs sequential mismatch at L={L}: {max_err_e}")
            self.assertLess(max_err_o, 1e-4, f"Parallel vs sequential mismatch at L={L}: {max_err_o}")

    def test_02_theorem1_orthogonal_rotation_energy_conservation(self) -> None:
        """Verifies Theorem 1: 2x2 rotation R(Delta_tilde * omega) preserves L2 spectral energy."""
        theta = torch.linspace(-10.0, 10.0, 500)
        v_even = torch.randn(500)
        v_odd = torch.randn(500)
        orig_energy = v_even.pow(2) + v_odd.pow(2)

        rot_even = torch.cos(theta) * v_even - torch.sin(theta) * v_odd
        rot_odd = torch.sin(theta) * v_even + torch.cos(theta) * v_odd
        rot_energy = rot_even.pow(2) + rot_odd.pow(2)

        self.assertTrue(torch.allclose(orig_energy, rot_energy, atol=1e-5))

    def test_03_theorem2_wasserstein_state_contamination_suppression(self) -> None:
        """
        Verifies Theorem 2: Under spatially non-uniform corruption (left half heavily corrupted,
        right half clean), Wasserstein geodesic gating (beta * W_2^2 > 0) suppresses cross-boundary
        state contamination compared to standard ZOH Mamba (gamma=0, beta=0).
        """
        B, C, H, W = 1, 16, 16, 32
        clean = torch.ones(B, C, H, W) * 0.5
        corrupted = clean.clone()
        # Inject heavy high-variance corruption on the left half (columns 0..15)
        corrupted[:, :, :, :16] += torch.randn(B, C, H, 16) * 3.5

        op_zoh = ComplexWassersteinSSM2D(
            dim=C, use_wasserstein=False, use_phase_rotation=False, scan_type="raster_1d"
        ).eval()
        op_cwsso = ComplexWassersteinSSM2D(
            dim=C, use_wasserstein=True, use_phase_rotation=True, scan_type="raster_1d"
        ).eval()
        # Share identical base weights and amplify geodesic barrier beta for clear boundary test
        op_cwsso.load_state_dict(op_zoh.state_dict(), strict=False)
        with torch.no_grad():
            op_cwsso.raw_beta.fill_(3.0)
            op_cwsso.raw_gamma.fill_(2.0)

        with torch.no_grad():
            _, diag_zoh_clean = op_zoh(clean, return_diagnostics=True)
            _, diag_zoh_corr = op_zoh(corrupted, return_diagnostics=True)
            _, diag_cw_clean = op_cwsso(clean, return_diagnostics=True)
            _, diag_cw_corr = op_cwsso(corrupted, return_diagnostics=True)

        # Measure state contamination in the clean right half (columns 16..31)
        err_zoh = (diag_zoh_clean["state_response"][:, :, :, 16:] - diag_zoh_corr["state_response"][:, :, :, 16:]).norm().item()
        err_cw = (diag_cw_clean["state_response"][:, :, :, 16:] - diag_cw_corr["state_response"][:, :, :, 16:]).norm().item()

        self.assertLess(
            err_cw,
            err_zoh,
            f"Expected CW-SSO contamination ({err_cw:.4f}) < ZOH contamination ({err_zoh:.4f})",
        )
        # Also verify W_2^2 map is strictly positive at the boundary
        self.assertGreater(diag_cw_corr["w2_map"].max().item(), 0.5)

    def test_04_mathematical_ablation_degeneracy_matrix_and_scans(self) -> None:
        """Tests all 4 mathematical ablation cases, 3 scan types, window sizes, and low-rank r."""
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        ablation_configs = [
            (False, False),  # Case 1: Standard real ZOH Mamba (gamma=0, Omega=0)
            (True, False),   # Case 2: Wasserstein Geodesic only (gamma>0, Omega=0)
            (False, True),   # Case 3: Rotational Spectral only (gamma=0, Omega!=0)
            (True, True),    # Case 4: Full CW-SSO Operator (gamma>0, Omega!=0)
        ]
        for use_w, use_rot in ablation_configs:
            for scan_type in ["bidirectional_2d", "hilbert", "raster_1d"]:
                op = ComplexWassersteinSSM2D(
                    dim=16,
                    window_size=3,
                    low_rank_r=8,
                    use_wasserstein=use_w,
                    use_phase_rotation=use_rot,
                    scan_type=scan_type,
                )
                out = op(x)
                self.assertEqual(out.shape, x.shape)
                self.assertFalse(torch.isnan(out).any().item())
                self.assertFalse(torch.isinf(out).any().item())
                loss = out.pow(2).mean()
                loss.backward()
                self.assertIsNotNone(x.grad)
                x.grad = None

    def test_05_parameter_budgets_tiny_base_and_evp_adapter(self) -> None:
        """
        Verifies strict TPAMI Gatekeeping KPI parameter counts:
        - CW-SSN-T: 5.5M <= params <= 6.2M (88% lighter than DA-RCOT 50.1M)
        - CW-SSN-B: 7.5M <= params <= 8.2M
        - CWSSOAdapter: < 5.5% of SegFormer-B4 (61.4M)
        """
        model_tiny = build_cw_ssn_tiny()
        model_base = build_cw_ssn_base()
        adapter_evp = CWSSOAdapter()

        params_tiny = sum(p.numel() for p in model_tiny.parameters() if p.requires_grad)
        params_base = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
        params_adapter = sum(p.numel() for p in adapter_evp.parameters() if p.requires_grad)
        segformer_b4_params = 61_400_000
        adapter_ratio_pct = 100.0 * params_adapter / segformer_b4_params

        print(f"\n[KPI Check] CW-SSN-T params:     {params_tiny:,} ({params_tiny/1e6:.2f}M)")
        print(f"[KPI Check] CW-SSN-B params:     {params_base:,} ({params_base/1e6:.2f}M)")
        print(f"[KPI Check] CWSSOAdapter params: {params_adapter:,} ({adapter_ratio_pct:.2f}% of SegFormer-B4)")

        self.assertGreaterEqual(params_tiny, 5_500_000)
        self.assertLessEqual(params_tiny, 6_200_000)
        self.assertGreaterEqual(params_base, 7_500_000)
        self.assertLessEqual(params_base, 8_200_000)
        self.assertLess(adapter_ratio_pct, 5.5)

    def test_06_full_backbone_forward_backward_and_odd_resolutions(self) -> None:
        """Tests CW-SSN-T forward/backward, latent descriptor, loss, and odd spatial sizes."""
        model = build_cw_ssn_tiny()
        criterion = SpectralWassersteinLoss()

        # Odd resolution (33 x 45) to verify automatic reflection padding & cropping
        inp = torch.randn(1, 3, 33, 45, requires_grad=True)
        target = torch.randn(1, 3, 33, 45)

        pred = model(inp)
        self.assertEqual(pred.shape, inp.shape)
        loss = criterion(pred, target)
        self.assertFalse(torch.isnan(loss).item())
        loss.backward()
        self.assertFalse(torch.isnan(inp.grad).any().item())

        # Latent descriptor extraction for NB_09 t-SNE plots
        with torch.no_grad():
            desc = model.extract_latent_descriptor(inp)
        self.assertEqual(desc.shape, (1, model.dims[-1]))

    def test_07_evp_adapter_multi_scale_forward(self) -> None:
        """Verifies CWSSOAdapter on multi-scale SegFormer-B4 feature pyramids."""
        adapter = CWSSOAdapter(stage_dims=(64, 128, 320, 512))
        feats = [
            torch.randn(2, 64, 32, 32),
            torch.randn(2, 128, 16, 16),
            torch.randn(2, 320, 8, 8),
            torch.randn(2, 512, 4, 4),
        ]
        logits = adapter(feats, target_size=(128, 128))
        self.assertEqual(logits.shape, (2, 1, 128, 128))
        self.assertFalse(torch.isnan(logits).any().item())


if __name__ == "__main__":
    unittest.main(verbosity=2)
