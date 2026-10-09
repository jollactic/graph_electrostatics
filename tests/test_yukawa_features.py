"""Independent checks of screened kernel normalization and self projection."""
import math
import unittest
import torch
torch.serialization.add_safe_globals([slice])
from graph_longrange.features import GTOElectrostaticFeatures, compute_yukawa_factor
from graph_longrange.gto_utils import GTOYukawaSelfInteractionBlock, GTOSelfInteractionBlock
from graph_longrange.utils import FIELD_CONSTANT


class YukawaFeaturesTests(unittest.TestCase):
    def setUp(self):
        self.old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        torch.set_default_dtype(self.old_dtype)

    def test_zero_mode_and_charge_preservation(self):
        descriptor = GTOElectrostaticFeatures(1, 1., 1, [0.8, 1.4], True, 3., pbc_handling="pbc")
        descriptor.configure_yukawa(0.5)
        cache = descriptor.precompute_geometry(
            torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, dtype=torch.long),
            torch.ones(1), torch.tensor([[0., 0., 0.], [1., 2., 3.]]),
            torch.zeros(2, dtype=torch.long), torch.tensor([100.]), torch.ones(1, 3, dtype=torch.bool),
        )
        alpha = torch.tensor([[1., 2., 3., 4.], [2., 0., 0., 0.]])
        beta = torch.tensor([[0.5, 0., 0., 0.], [0.5, 1., 2., 3.]])
        charge = descriptor.forward_dynamic(cache, alpha + beta)
        spin = descriptor.forward_dynamic_yukawa(cache, alpha-beta, compute_yukawa_factor(cache["k_norm2"], .5))
        expected = torch.zeros_like(spin)
        expected[:, :2] = math.sqrt(4*math.pi)*FIELD_CONSTANT*2/100/.5**2
        torch.testing.assert_close(spin, expected)
        torch.testing.assert_close(charge, torch.zeros_like(charge))
        torch.testing.assert_close(.5*(charge+spin)+.5*(charge-spin), charge)
        torch.testing.assert_close(descriptor.forward_dynamic(cache, alpha), torch.zeros_like(charge))

    def test_nonzero_kernel_and_validation(self):
        torch.testing.assert_close(compute_yukawa_factor(torch.tensor([0., 4.]), 2.), torch.tensor([.25, .125]))
        for kappa in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                compute_yukawa_factor(torch.zeros(1), kappa)

    def test_self_integral_coulomb_limit(self):
        args = dict(l_source=1, sigma_source=1., l_receive=1,
                    sigmas_receive=[.8, 1.4], normalize_source="multipoles", normalize_receive="receiver")
        coulomb = GTOSelfInteractionBlock(**args)
        screened = GTOYukawaSelfInteractionBlock(**args, kappa=1e-8)
        torch.testing.assert_close(screened.overlap_constants, coulomb.overlap_constants, rtol=2e-7, atol=1e-7)
        screened = GTOYukawaSelfInteractionBlock(**args, kappa=.5)
        self.assertTrue(bool(torch.all(screened.overlap_constants < coulomb.overlap_constants)))

    def test_realspace_kernel_fourier_integral_and_gradients(self):
        import scipy.integrate
        import numpy as np
        from graph_longrange.realspace_electrostatics import gaussian_yukawa_kernel
        for r in [0., .2, 2., 10.]:
            value, _ = scipy.integrate.quad(
                lambda k: (2/math.pi)*k*k*np.exp(-.8**2*k*k)
                * np.sinc(k*r/math.pi)/(k*k+.5**2), 0, np.inf,
                epsabs=1e-12,
            )
            actual = gaussian_yukawa_kernel(torch.tensor(r), torch.tensor(.8), .5)
            self.assertAlmostEqual(actual.item(), value, places=11)
        distance = torch.tensor([.2, 2., 10.], requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda r: gaussian_yukawa_kernel(r, torch.tensor(.8), .5), (distance,)
        ))
        self.assertTrue(torch.isfinite(gaussian_yukawa_kernel(torch.tensor([0., 1., 100.]), torch.tensor(1.), 100.)).all())

    def test_realspace_periodic_convergence(self):
        from graph_longrange.kspace import compute_k_vectors_flat
        positions = torch.tensor([[0., 0., 0.], [2., .3, .1]], requires_grad=True)
        source = torch.tensor([[1.], [-.7]])
        batch = torch.zeros(2, dtype=torch.long)
        descriptor = GTOElectrostaticFeatures(0, .8, 0, [1.], False, 5., pbc_handling="auto")
        descriptor.configure_yukawa(.5)
        real_cache = {"mode": "realspace", "node_positions": positions, "batch": batch}
        isolated = descriptor.forward_dynamic_yukawa(real_cache, source, None)
        cell = torch.eye(3).unsqueeze(0)*30
        k, k2, kb, mask = compute_k_vectors_flat(5., cell, 2*math.pi*torch.linalg.inv(cell).transpose(-1, -2))
        cache = descriptor.precompute_geometry(k, k2, kb, mask, positions, batch,
                                              torch.tensor([30.**3]), torch.ones(1, 3, dtype=torch.bool))
        periodic = descriptor.forward_dynamic_yukawa(cache, source, compute_yukawa_factor(k2, .5))
        torch.testing.assert_close(isolated, periodic, rtol=2e-5, atol=2e-5)
        gi = torch.autograd.grad(isolated.sum(), positions, retain_graph=True)[0]
        gp = torch.autograd.grad(periodic.sum(), positions)[0]
        torch.testing.assert_close(gi, gp, rtol=2e-5, atol=2e-5)

    def test_isolated_dipole_gradients_and_self_terms(self):
        descriptor = GTOElectrostaticFeatures(1, .8, 1, [1.], False, 5., pbc_handling="realspace")
        descriptor.configure_yukawa(.5)
        positions = torch.tensor([[0., 0., 0.], [2., .3, .1]], requires_grad=True)
        source = torch.tensor([[1., .1, .2, .3], [-.7, .3, -.2, .1]])
        batch = torch.zeros(2, dtype=torch.long)
        def evaluate(pos):
            return descriptor.forward_dynamic_yukawa(
                {"mode": "realspace", "node_positions": pos, "batch": batch}, source,
            )
        self.assertTrue(torch.autograd.gradcheck(evaluate, (positions,), atol=1e-5))
        without = evaluate(positions)
        descriptor.realspace_yukawa_features.include_self_interaction = True
        with_self = evaluate(positions)
        torch.testing.assert_close(with_self-without, descriptor.yukawa_self_interaction_terms(source))


if __name__ == "__main__":
    unittest.main()
