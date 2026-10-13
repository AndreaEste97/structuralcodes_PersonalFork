"""AsdSteel1D constitutive law as described in: 
Kashani, M. M., Lowes, L. N., Crewe, A. J., & Alexander, N. A. (2015). 
“Phenomenological hysteretic model for corroded reinforcing bars including 
inelastic buckling and low-cycle fatigue degradation” 
Computers & Structures, 156, 58-71"""

from __future__ import annotations  # To have clean hints of ArrayLike in docs

import typing as t

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

    def get_stress(
        self, eps: t.Union[float, ArrayLike]
    ) -> t.Union[float, ArrayLike]:
        """Return the stress given strain."""
        eps = eps if np.isscalar(eps) else np.atleast_1d(eps)
        # Preprocess eps array in order
        eps = self.preprocess_strains_with_limits(eps=eps)
        mu=self._Eh/self._E
        # Compute stress
        sig = self._E * eps
        if np.isscalar(sig):
            sig=np.sign(eps)*self._fy*(1-mu)/2*(1+(1+mu)*abs(eps)/(1-mu)/self._eps_sy-np.sqrt((abs(eps)/self._eps_sy-1)**2+self._Delta))
            if (self._eps_su is not None) and (abs(eps) > self._eps_su):
                sig = 0
            return sig
        sig=np.sign(eps)*self._fy*(1-mu)/2*(1+(1+mu)*abs(eps)/(1-mu)/self._eps_sy-np.sqrt((abs(eps)/self._eps_sy-1)**2+self._Delta))
        if self._eps_su is not None:
            sig[eps > self._eps_su] = 0
            sig[eps < -self._eps_su] = 0
        return sig

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
