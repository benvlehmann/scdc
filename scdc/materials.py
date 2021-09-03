"""This module defines a set of materials with all the needed parameters."""

from .material import Material


SILICON = Material(
    symbol='Si',
    T_c=None,
    Delta=(1.17/2.),
    m_star_ratio=0.98,
    c_s=5880.,
    beta=1.96e-5,
    gamma=1.28e-2
)

CARBON = Material(
    symbol='C',
    T_c=None,
    Delta=(5.5/2.),
    m_star_ratio=1.4,
    c_s=13360.,
    beta=4.45e-5,
    gamma=1.60e-2
)

TUNGSTEN = Material(
    symbol='W',
    T_c=0.015,
    E_F=9.2,
    Delta=(4.52e-6/2.),
    m_star_ratio=1.,
    c_s=5000.,
    beta=1.67e-5,
    gamma=5.60e0
)

IRIDIUM = Material(
    symbol='Ir',
    T_c=0.14,
    Delta=(4.21e-5/2.),
    m_star_ratio=1.,
    c_s=5000.,
    beta=1.67e-5,
    gamma=1.83e0
)

HAFNIUM = Material(
    symbol='Hf',
    T_c=0.165,
    Delta=(4.97e-5/2.),
    m_star_ratio=1.,
    c_s=3000.,
    beta=5.10e-5,
    gamma=1.01e0
)

ALUMINUM = Material(
    symbol='Al',
    T_c=1.2,
    E_F=11.7,
    Delta=(3.61e-4/2.),
    m_star_ratio=1.,
    c_s=6320.,
    beta=2.26e-4,
    gamma=7.92e-1
)

ZINC = Material(
    symbol='Zn',
    T_c=0.855,
    E_F=9.47,
    Delta=(2.57e-4/2.),
    m_star_ratio=1.,
    c_s=4000.,
    beta=9.07e-5,
    gamma=5.94e-1
)

INDIUM = Material(
    symbol='In',
    T_c=3.4,
    E_F=8.63,
    Delta=(1.02e-3/2.),
    m_star_ratio=1.,
    c_s=1215.,
    beta=8.37e-6,
    gamma=9.04e-2
)

TIN = Material(
    symbol='Sn',
    T_c=3.72,
    E_F=10.2,
    Delta=(1.12e-3/2.),
    m_star_ratio=1.,
    c_s=3000.,
    beta=5.10e-5,
    gamma=2.13e-1
)

TANTALUM = Material(
    symbol='Ta',
    T_c=4.48,
    Delta=(1.35e-3/2.),
    m_star_ratio=1.,
    c_s=3300.,
    beta=6.17e-5,
    gamma=2.14e-1
)

LEAD = Material(
    symbol='Pb',
    T_c=7.19,
    E_F=9.47,
    Delta=(2.16e-3/2.),
    m_star_ratio=1.,
    c_s=2000.,
    beta=2.27e-5,
    gamma=1.02e-1
)

NIOBIUM = Material(
    symbol='Nb',
    T_c=9.26,
    E_F=5.32,
    Delta=(2.79e-3/2.),
    m_star_ratio=1.,
    c_s=3480.,
    beta=6.86e-5,
    gamma=1.57e-1
)

NIOBIUM_NITRIDE = Material(
    symbol='NbN',
    T_c=16.,
    Delta=(4.82e-3/2.),
    m_star_ratio=1.,
    c_s=5000.,
    beta=1.42e-4,
    gamma=1.72e-1
)

BENCHMARK_MATERIALS = (
    SILICON, CARBON, TUNGSTEN, IRIDIUM, HAFNIUM, ALUMINUM, ZINC, INDIUM, TIN,
    TANTALUM, LEAD, NIOBIUM, NIOBIUM_NITRIDE
)

HEAVYMEDONIUM = Material(
    symbol='FAKE_HM',
    T_c=None,
    Delta=(5.5/2.),
    m_star_ratio=1.4,
    c_s=13360.,
    beta=4.45e-5,
    gamma=10.
)

UNOBTAINIUM = Material(gamma=3.)

HIGHGAMMANIUM = Material(
    symbol='HiGa',
    T_c=1.2,
    Delta=(3.61e-4/2.),
    m_star_ratio=1.,
    c_s=6320.,
    beta=2.26e-4,
    gamma=7.92e0
)

HIGHEFERMIUM = Material(
    symbol='HiEF',
    T_c=1.2,
    Delta=(3.61e-4/2.),
    m_star_ratio=1.,
    c_s=6320.,
    beta=2.26e-4,
    gamma=7.92e-1,
    E_F=1e3
)
