"""AsdSteel1D constitutive law as described in: 
Kashani, M. M., Lowes, L. N., Crewe, A. J., & Alexander, N. A. (2015). 
“Phenomenological hysteretic model for corroded reinforcing bars including 
inelastic buckling and low-cycle fatigue degradation” 
Computers & Structures, 156, 58-71"""

from __future__ import annotations  # To have clean hints of ArrayLike in docs

import typing as t
import warnings

import numpy as np
from numpy.typing import ArrayLike

from ...core.base import ConstitutiveLaw


class AsdSteel1D(ConstitutiveLaw):
    """Class for AsdSteel1D Constitutive Law."""

    __materials__: t.Tuple[str] = (
        'steel',
        'rebars',
    )

    def __init__(
        self,
        E: float,
        fy: float,
        Delta: float,
        Eh: t.Optional[float] = None,
        eps_su: t.Optional[float] = None,
        ftk: t.Optional[float] = None,
        name: t.Optional[str] = None,
        Stress_Slip: t.Optional[ArrayLike] = None,
        Strain_Slip: t.Optional[ArrayLike] = None,
        L_OVer_D_ratio: t.Optional[float] = None,
    ) -> None:
        """Initialize an AsdSteel1D Material.

        Arguments:
            E (float): The elastic modulus.
            fy (float): The yield strength.
            Delta (float): Shape parameter.

        Keyword Arguments:
            Eh (float, optional): The hardening modulus. If omitted, it is
                calculated from ``ftk`` and ``eps_su``.
            eps_su (float, optional): The ultimate strain.
            ftk (float, optional): The ultimate stress used to calculate
                ``Eh`` when it is omitted.
            name (str, optional): A descriptive name for the constitutive law.
            Stress_Slip (ArrayLike, optional): Stress values for the
                stress-slip relationship.
            Strain_Slip (ArrayLike, optional): Strain values for the
                stress-slip relationship.
            L_OVer_D_ratio (float, optional): Length-to-diameter ratio used
                to model reinforcement buckling in compression.
        """
        super().__init__(name=name, base_name='AsdSteel1D')
        if E > 0:
            self._E = E
        else:
            raise ValueError('Elastic modulus E must be greater than zero')
        if Eh is None:
            if eps_su is None or ftk is None:
                raise ValueError(
                    'Eh can only be omitted when both eps_su and ftk are '
                    'provided.'
                )
            eps_hardening = eps_su - fy / E
            if eps_hardening == 0:
                raise ValueError(
                    'eps_su must differ from the yield strain to calculate Eh.'
                )
            Eh = (ftk - fy) / eps_hardening

        self._fy = fy
        self._Eh = Eh
        self._eps_su = eps_su
        self._eps_sy = fy / E
        self._Delta=Delta
        if (Stress_Slip is None) != (Strain_Slip is None):
            raise ValueError(
                'Stress_Slip and Strain_Slip must be provided together.'
            )
        if L_OVer_D_ratio is None:
            self._LDRatio=None
            self._LambdaP=None
            self._Rho1=None
            self._Rho2=None
            self._SigmaStar=None
        else:
            if not np.isfinite(L_OVer_D_ratio) or L_OVer_D_ratio <= 0:
                raise ValueError(
                    'L_OVer_D_ratio must be a finite positive value.'
                )
            self._LDRatio = L_OVer_D_ratio
            self._LambdaP = np.sqrt(self._fy / 100) * self._LDRatio
            self._Rho1 = 4.572 * self._LambdaP - 74.43
            self._Rho2 = 318.4 * np.exp(-0.071 * self._LambdaP)
            self._SigmaStar = 3.75 * self._fy / self._LDRatio
            if self._LDRatio > 30:
                warnings.warn(
                    'AsdSteel1D is calibrated for compression up to '
                    'L_OVer_D_ratio <= 30.',
                    UserWarning,
                    stacklevel=2,
                )
        self._bonded_strain = None
        self._bonded_stress = None
        self._bonded_stress_right = None
        self._bonded_compression_strain = None
        self._bonded_compression_stress = None
        self._bonded_compression_stress_right = None
        if Stress_Slip is not None and Strain_Slip is not None:
            stress_slip = np.asarray(Stress_Slip, dtype=float)
            strain_slip = np.asarray(Strain_Slip, dtype=float)
            if (
                stress_slip.ndim != 1
                or strain_slip.ndim != 1
                or len(stress_slip) != len(strain_slip)
                or len(stress_slip) < 2
                or not np.all(np.isfinite(stress_slip))
                or not np.all(np.isfinite(strain_slip))
                or np.any(stress_slip < 0)
                or np.any(strain_slip < 0)
                or np.any(np.diff(strain_slip) <= 0)
                or np.max(stress_slip) <= 0
            ):
                raise ValueError(
                    'Stress_Slip and Strain_Slip must be finite one-'
                    'dimensional arrays of equal length, with non-negative '
                    'stress and strain values, with strictly increasing '
                    'strain values and at least one positive stress.'
                )
            self._set_bonded_curve(stress_slip, strain_slip)
            compression_strains, _ = self._get_tension_curve(
                float(np.max(stress_slip))
            )
            compression_stresses = np.abs(
                self._get_base_stress(-compression_strains)
            )
            (
                self._bonded_compression_strain,
                self._bonded_compression_stress,
                self._bonded_compression_stress_right,
            ) = self._combine_bonded_curve(
                compression_strains,
                compression_stresses,
                stress_slip,
                strain_slip,
                use_steel_curve=True,
            )

    def _get_base_stress(
        self, eps: t.Union[float, ArrayLike]
    ) -> t.Union[float, ArrayLike]:
        """Return the AsdSteel1D stress without a slip contribution."""
        eps = np.asarray(eps, dtype=float)
        mu = self._Eh / self._E
        abs_eps = np.abs(eps)
        tension_stress = (np.sign(eps)* self._fy* (1 - mu)/ 2* (1+ (1 + mu) * abs_eps / (1 - mu) / self._eps_sy- np.sqrt((abs_eps / self._eps_sy - 1) ** 2 + self._Delta)))
        if self._LDRatio is None or self._LDRatio < 6:
            sig = tension_stress
        elif self._LDRatio <= 8:
            sig = np.where(
                eps >= 0,
                tension_stress,
                np.maximum(self._E * eps, -self._fy),
            )
        else:
            compressive_plastic_strain = np.maximum(
                -eps - self._eps_sy, 0
            )
            compressive_stress = -(
                self._SigmaStar
                + (self._fy - self._SigmaStar)
                * np.exp(
                    -(
                        self._Rho1
                        + self._Rho2
                        * np.sqrt(compressive_plastic_strain)
                    )
                    * compressive_plastic_strain
                )
            )
            sig = np.where(
                eps >= 0,
                tension_stress,
                np.where(
                    eps >= -self._eps_sy,
                    self._E * eps,
                    compressive_stress,
                ),
            )
        if self._eps_su is not None:
            sig = np.where(np.abs(eps) > self._eps_su, 0.0, sig)
        if sig.ndim == 0:
            return float(sig)
        return sig

    def _set_bonded_curve(
        self, stress_slip: np.ndarray, strain_slip: np.ndarray
    ) -> None:
        """Combine the steel and slip curves using strain compatibility."""
        steel_strains, steel_stresses = self._get_tension_curve(
            float(np.max(stress_slip))
        )
        (
            self._bonded_strain,
            self._bonded_stress,
            self._bonded_stress_right,
        ) = self._combine_bonded_curve(
            steel_strains,
            steel_stresses,
            stress_slip,
            strain_slip,
        )

    def _get_tension_curve(
        self, max_slip_stress: float
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """Return sampled tensile strains and stresses."""
        if self._eps_su is not None:
            steel_strains = np.linspace(0, self._eps_su, 1000)
            steel_stresses = self._get_base_stress(steel_strains)
        else:
            strain_limit = max(self._eps_sy, max_slip_stress / self._E)
            for _ in range(32):
                steel_strains = np.linspace(0, strain_limit, 1000)
                steel_stresses = self._get_base_stress(steel_strains)
                if steel_stresses[-1] >= max_slip_stress:
                    break
                strain_limit *= 2
            else:
                raise ValueError(
                    'Could not determine a steel strain range for '
                    'Stress_Slip.'
                )
        return steel_strains, steel_stresses

    def _combine_bonded_curve(
        self,
        steel_strains: np.ndarray,
        steel_stresses: np.ndarray,
        stress_slip: np.ndarray,
        strain_slip: np.ndarray,
        use_steel_curve: bool = False,
    ) -> t.Tuple[np.ndarray, np.ndarray, t.Optional[float]]:
        """Combine sampled steel and slip curves using strain compatibility."""
        peak_index = int(np.argmax(stress_slip))
        if peak_index == 0:
            raise ValueError(
                'Stress_Slip must increase from its first value to its '
                'maximum.'
            )
        ascending_stress = stress_slip[: peak_index + 1]
        ascending_strain = strain_slip[: peak_index + 1]
        if np.any(np.diff(ascending_stress) < 0):
            raise ValueError(
                'Stress_Slip must be non-decreasing up to its maximum.'
            )

        steel_peak_index = int(np.argmax(steel_stresses))
        ascending_steel_stress = steel_stresses[: steel_peak_index + 1]
        ascending_steel_strain = steel_strains[: steel_peak_index + 1]
        if np.any(np.diff(ascending_steel_stress) < 0):
            raise ValueError(
                'The steel stress curve must be non-decreasing up to its '
                'maximum when Stress_Slip and Strain_Slip are used.'
            )

        steel_governs = use_steel_curve or (
            np.max(steel_stresses) <= np.max(stress_slip)
        )
        if steel_governs:
            slip_strains = np.interp(
                steel_stresses, ascending_stress, ascending_strain
            )
            if use_steel_curve and steel_peak_index < len(steel_stresses) - 1:
                descending_stress = stress_slip[peak_index:][::-1]
                descending_strain = strain_slip[peak_index:][::-1]
                if np.any(np.diff(descending_stress) < 0):
                    raise ValueError(
                        'Stress_Slip must be non-increasing after its '
                        'maximum to combine the compressive curve.'
                    )
                keep = np.r_[
                    descending_stress[1:] != descending_stress[:-1],
                    True,
                ]
                slip_strains[steel_peak_index + 1 :] = np.interp(
                    steel_stresses[steel_peak_index + 1 :],
                    descending_stress[keep],
                    descending_strain[keep],
                )
            bonded_strain = steel_strains + slip_strains
            bonded_stress = steel_stresses
            bonded_stress_right = (
                0.0 if self._eps_su is not None else None
            )
        else:
            bonded_strain = strain_slip + np.interp(
                stress_slip, ascending_steel_stress, ascending_steel_strain
            )
            bonded_stress = stress_slip

        if np.any(np.diff(bonded_strain) <= 0):
            raise ValueError(
                'The combined strain values from Stress_Slip and '
                'Strain_Slip must be strictly increasing.'
            )
        return bonded_strain, bonded_stress, (
            bonded_stress_right if steel_governs else None
        )

    def get_stress(self, eps: t.Union[float, ArrayLike]) -> t.Union[float, ArrayLike]:
        """Return the stress given strain."""
        eps = eps if np.isscalar(eps) else np.atleast_1d(eps)
        # Preprocess eps array in order
        eps = self.preprocess_strains_with_limits(eps=eps)
        if self._bonded_strain is None or self._bonded_stress is None:
            return self._get_base_stress(eps)
        eps_array = np.asarray(eps, dtype=float)
        abs_eps = np.abs(eps_array)
        tensile_stress = np.interp(abs_eps,self._bonded_strain,self._bonded_stress,right=self._bonded_stress_right,)
        if (self._bonded_compression_strain is None or self._bonded_compression_stress is None):
            return np.sign(eps) * tensile_stress
        compressive_stress = np.interp( abs_eps,self._bonded_compression_strain,self._bonded_compression_stress,right=self._bonded_compression_stress_right,)
        stress = np.where(eps_array < 0, -compressive_stress, tensile_stress)
        if stress.ndim == 0:
            return float(stress)
        return stress

    # le funzioni sottostanti sono ancora da aggiornare

    def get_tangent(
        self, eps: t.Union[float, ArrayLike]
    ) -> t.Union[float, ArrayLike]:
        """Return the tangent for the given strain."""
        eps = eps if np.isscalar(eps) else np.atleast_1d(eps)
        eps = self.preprocess_strains_with_limits(eps=eps)
        mu = self._Eh / self._E
        factor = self._fy * (1 - mu) / 2

        if np.isscalar(eps):
            if eps <= 0 or (
                self._eps_su is not None and eps > self._eps_su
            ):
                return 0.0
            normalized_eps = eps / self._eps_sy
            root = np.sqrt((normalized_eps - 1) ** 2 + self._Delta)
            return factor * (
                (1 + mu) / ((1 - mu) * self._eps_sy)
                - (normalized_eps - 1) / (self._eps_sy * root)
            )

        tangent = np.zeros_like(eps, dtype=float)
        mask = eps > 0
        if self._eps_su is not None:
            mask &= eps <= self._eps_su
        normalized_eps = eps[mask] / self._eps_sy
        root = np.sqrt((normalized_eps - 1) ** 2 + self._Delta)
        tangent[mask] = factor * (
            (1 + mu) / ((1 - mu) * self._eps_sy)
            - (normalized_eps - 1) / (self._eps_sy * root)
        )
        return tangent

    def get_ultimate_strain(self) -> t.Tuple[float, float]:
        """Return the negative and positive ultimate strains."""
        if self._eps_su is None:
            return (-2 * self._eps_sy, 2 * self._eps_sy)
        return (-self._eps_su, self._eps_su)

    # def get_tangent(
    #     self, eps: t.Union[float, ArrayLike]
    # ) -> t.Union[float, ArrayLike]:
    #     """Return the tangent for given strain."""
    #     if np.isscalar(eps):
    #         tangent = (
    #             self._E if -self._eps_sy <= eps <= self._eps_sy else self._Eh
    #         )
    #         if (self._eps_su is not None) and (
    #             eps > self._eps_su or eps < -self._eps_su
    #         ):
    #             tangent = 0
    #         return tangent

    #     eps = np.atleast_1d(eps)
    #     tangent = np.ones_like(eps) * self._E
    #     tangent[eps > self._eps_sy] = self._Eh
    #     tangent[eps < -self._eps_sy] = self._Eh
    #     if self._eps_su is not None:
    #         tangent[eps > self._eps_su] = 0
    #         tangent[eps < -self._eps_su] = 0  # pylint: disable=E1130

    #     return tangent

    # def __marin__(
    #     self, strain: t.Tuple[float, float]
    # ) -> t.Tuple[t.List[t.Tuple], t.List[t.Tuple]]:
    #     """Returns coefficients and strain limits for Marin integration in a
    #     simply formatted way.

    #     Arguments:
    #         strain (float, float): Tuple defining the strain profile: eps =
    #             strain[0] + strain[1]*y.

    #     Example:
    #         [(0, -0.002), (-0.002, -0.003)]
    #         [(a0, a1, a2), (a0)]
    #     """
    #     strains = []
    #     coeff = []
    #     eps_sy_n, eps_sy_p = self.get_ultimate_strain(yielding=True)
    #     eps_su_n, eps_su_p = self.get_ultimate_strain()
    #     if strain[1] == 0:
    #         # Uniform strain equal to strain[0]
    #         # Understand in which branch are we
    #         strain[0] = self.preprocess_strains_with_limits(strain[0])
    #         if strain[0] > eps_sy_p and strain[0] <= eps_su_p:
    #             # We are in the Hardening part positive
    #             strains = None
    #             a0 = self._Eh * strain[0] + self._fy * (1 - self._Eh / self._E)
    #             a1 = self._Eh * strain[1]
    #             coeff.append((a0, a1))
    #         elif strain[0] < eps_sy_n and strain[0] >= eps_su_n:
    #             # We are in the Hardening part negative
    #             strains = None
    #             a0 = self._Eh * strain[0] - self._fy * (1 - self._Eh / self._E)
    #             a1 = self._Eh * strain[1]
    #             coeff.append((a0, a1))
    #         elif abs(strain[0]) <= self._eps_sy:
    #             # We are in the elastic part
    #             strains = None
    #             a0 = self._E * strain[0]
    #             a1 = self._E * strain[1]
    #             coeff.append((a0, a1))
    #         else:
    #             strains = None
    #             coeff.append((0.0,))
    #     else:
    #         # Hardening part negative
    #         strains.append((eps_su_n, eps_sy_n))
    #         a0 = self._Eh * strain[0] - self._fy * (1 - self._Eh / self._E)
    #         a1 = self._Eh * strain[1]
    #         coeff.append((a0, a1))
    #         # Elastic part
    #         strains.append((eps_sy_n, eps_sy_p))
    #         a0 = self._E * strain[0]
    #         a1 = self._E * strain[1]
    #         coeff.append((a0, a1))
    #         # Hardening part positive
    #         strains.append((eps_sy_p, eps_su_p))
    #         a0 = self._Eh * strain[0] + self._fy * (1 - self._Eh / self._E)
    #         a1 = self._Eh * strain[1]
    #         coeff.append((a0, a1))
    #     return strains, coeff

    # def __marin_tangent__(
    #     self, strain: t.Tuple[float, float]
    # ) -> t.Tuple[t.List[t.Tuple], t.List[t.Tuple]]:
    #     """Returns coefficients and strain limits for Marin integration of
    #     tangent in a simply formatted way.

    #     Arguments:
    #         strain (float, float): Tuple defining the strain profile: eps =
    #             strain[0] + strain[1]*y.

    #     Example:
    #         [(0, -0.002), (-0.002, -0.003)]
    #         [(a0, a1, a2), (a0)]
    #     """
    #     strains = []
    #     coeff = []
    #     eps_sy_n, eps_sy_p = self.get_ultimate_strain(yielding=True)
    #     eps_su_n, eps_su_p = self.get_ultimate_strain()
    #     if strain[1] == 0:
    #         # Uniform strain equal to strain[0]
    #         # Understand in which branch are we
    #         strain[0] = self.preprocess_strains_with_limits(strain[0])
    #         if strain[0] > eps_sy_p and strain[0] <= eps_su_p:
    #             # We are in the Hardening part positive
    #             strains = None
    #             a0 = self._Eh
    #             coeff.append((a0,))
    #         elif strain[0] < eps_sy_n and strain[0] >= eps_su_n:
    #             # We are in the Hardening part negative
    #             strains = None
    #             a0 = self._Eh
    #             coeff.append((a0,))
    #         elif abs(strain[0]) <= self._eps_sy:
    #             # We are in the elastic part
    #             strains = None
    #             a0 = self._E
    #             coeff.append((a0,))
    #         else:
    #             strains = None
    #             coeff.append((0.0,))
    #     else:
    #         # Hardening part negative
    #         strains.append((eps_su_n, eps_sy_n))
    #         a0 = self._Eh
    #         coeff.append((a0,))
    #         # Elastic part
    #         strains.append((eps_sy_n, eps_sy_p))
    #         a0 = self._E
    #         coeff.append((a0,))
    #         # Hardening part positive
    #         strains.append((eps_sy_p, eps_su_p))
    #         a0 = self._Eh
    #         coeff.append((a0,))
    #     return strains, coeff

    # def get_ultimate_strain(
    #     self, yielding: bool = False
    # ) -> t.Tuple[float, float]:
    #     """Return the ultimate strain (negative and positive)."""
    #     if yielding:
    #         return (-self._eps_sy, self._eps_sy)
    #     # If not specified eps
    #     if self._eps_su is None:
    #         return (-self._eps_sy * 2, self._eps_sy * 2)
    #     return (-self._eps_su, self._eps_su)
