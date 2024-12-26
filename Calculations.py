# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 15:28:01 2024

@author: Gaurav Bharatha
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from numpy import sin, cos, tan

'''THERMALLY PERFECT GAS ASSUMPTION'''
# constants
R = 287  # J/kgK


### Isentropic Flow Relations
def isen(**kwargs):
    gam = kwargs.get('gamma')
    if gam is None:
        raise ValueError("You must provide the ratio of specific heats 'gamma'.")

    nongamma_keys = [key for key in kwargs if key != 'gamma']

    nongamma = nongamma_keys[0]
    value = kwargs[nongamma]

    def calc_mach_from_p0p(p0_p, gam):
        def f(x):
            return ((1 + ((gam - 1) / 2) * x ** 2) ** (gam / (gam - 1))) - p0_p

        def df(x):
            return 2 * gam * x * (gam / 2 - 1 / 2) * (x ** 2 * (gam / 2 - 1 / 2) + 1) ** (gam / (gam - 1)) / (
                    (gam - 1) * (x ** 2 * (gam / 2 - 1 / 2) + 1))

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_rho0rho(rho0_rho, gam):
        def f(x):
            return ((1 + ((gam - 1) / 2) * x ** 2) ** (1 / (gam - 1))) - rho0_rho

        def df(x):
            return 2 * x * (gam / 2 - 1 / 2) * (x ** 2 * (gam / 2 - 1 / 2) + 1) ** (1 / (gam - 1)) / (
                    (gam - 1) * (x ** 2 * (gam / 2 - 1 / 2) + 1))

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_t0t(T0_T, gam):
        def f(x):
            return (1 + ((gam - 1) / 2) * x ** 2) - T0_T

        def df(x):
            return 2 * x * (gam / 2 - 1 / 2)

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_prandtl_meyer(angle, gam):
        rad_ang = angle * np.pi / 180

        def f(x):
            return (np.sqrt((gam + 1) / (gam - 1))) * np.arctan(
                np.sqrt((x ** 2 - 1) * ((gam - 1) / (gam + 1)))) - np.arctan(np.sqrt(x ** 2 - 1)) - rad_ang

        def df(x):
            return x * np.sqrt((gam + 1) / (gam - 1)) * np.sqrt((gam - 1) * (x ** 2 - 1) / (gam + 1)) / (
                    (x ** 2 - 1) * ((gam - 1) * (x ** 2 - 1) / (gam + 1) + 1)) - 1 / (x * np.sqrt(x ** 2 - 1))

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_aa(aa, gam, supersonic=True):
        def f(x):
            return (1 / x) * ((2 / (gam + 1)) * (1 + ((gam - 1) / 2) * x ** 2)) ** ((gam + 1) / (2 * (gam - 1))) - aa

        def df(x):
            return 2 * (2 * (x ** 2 * (gam / 2 - 1 / 2) + 1) / (gam + 1)) ** ((gam + 1) / (2 * gam - 2)) * (
                    gam / 2 - 1 / 2) * (gam + 1) / ((2 * gam - 2) * (x ** 2 * (gam / 2 - 1 / 2) + 1)) - (
                    2 * (x ** 2 * (gam / 2 - 1 / 2) + 1) / (gam + 1)) ** ((gam + 1) / (2 * gam - 2)) / x ** 2

        if supersonic:
            x0 = 1.5
            M = sc.optimize.newton(f, x0, df)

        else:
            x0 = 0.5
            M = sc.optimize.newton(f, x0, df)

        return M

    def M_func(value, gam):
        return value

    def calc_mach_from_mach_wave_angle(mu, gam):
        mu_rad = mu * np.pi / 180
        M = 1 / np.sin(mu_rad)
        return M

    mach_calculators = {
        'M': M_func,
        'p0_p': calc_mach_from_p0p,
        'rho0_rho': calc_mach_from_rho0rho,
        'T0_T': calc_mach_from_t0t,
        'Prandtl_Meyer_Angle_(deg)': calc_mach_from_prandtl_meyer,
        'Mach_Wave_Angel_(deg)': calc_mach_from_mach_wave_angle,
        'A_As_sub': lambda v, gam: calc_mach_from_aa(v, gam, supersonic=False),
        'A_As_sup': lambda v, gam: calc_mach_from_aa(v, gam, supersonic=True),
    }

    M = mach_calculators[nongamma](value, gam)

    results = {
        'M': M,
        'p0_p': (1 + ((gam - 1) / 2) * M ** 2) ** (gam / (gam - 1)),
        'rho0 rho': (1 + ((gam - 1) / 2) * M ** 2) ** (1 / (gam - 1)),
        'T0_T': 1 + ((gam - 1) / 2) * M ** 2,
        'Prandtl Meyer Angle (deg)': '' if M < 1 else (((np.sqrt((gam + 1) / (gam - 1))) * np.arctan(
            np.sqrt((M ** 2 - 1) * ((gam - 1) / (gam + 1)))) - np.arctan(np.sqrt(M ** 2 - 1))) * 180 / np.pi),
        'Mach Wave Angle (deg)': '' if M < 1 else np.arcsin(1 / M) * 180 / np.pi,
        'A/A*': (1 / M) * ((2 / (gam + 1)) * (1 + ((gam - 1) / 2) * M ** 2)) ** ((gam + 1) / (2 * (gam - 1))),
        'p*/p': (2 / (gam + 1)) ** (gam / (gam - 1)),
        'rho*/rho': (2 / (gam + 1)) ** (1 / (gam - 1)),
        'T*/T': (2 / (gam + 1))
    }

    return results


### Normal Shock Relations
def norm_shock(**kwargs):
    gam = kwargs.get('gamma')
    if gam is None:
        raise ValueError("You must provide the ratio of specific heats 'gamma'.")
    nongamma_keys = [key for key in kwargs if key != 'gamma']

    nongamma = nongamma_keys[0]
    value = kwargs[nongamma]

    def calc_M1_from_M2(M2, gam):
        def f(x):
            return ((2 + (gam - 1) * x ** 2) / (2 * gam * x ** 2 - (gam - 1))) - M2 ** 2

        def df(x):
            return -4 * gam * x * (x ** 2 * (gam - 1) + 2) / (2 * gam * x ** 2 - gam + 1) ** 2 + 2 * x * (gam - 1) / (
                        2 * gam * x ** 2 - gam + 1)

        x0 = 0.5
        M1 = sc.optimize.newton(f, x0, df)
        return M1

    def calc_M1_from_rho2rho1(rho2_rho1, gam):
        def f(x):
            return (((gam + 1) * x ** 2) / (2 + (gam - 1) * x ** 2)) - rho2_rho1

        def df(x):
            return -2 * x ** 3 * (gam - 1) * (gam + 1) / (x ** 2 * (gam - 1) + 2) ** 2 + 2 * x * (gam + 1) / (
                        x ** 2 * (gam - 1) + 2)

        x0 = 1
        M1 = sc.optimize.newton(f, x0, df)
        return M1

    def calc_M1_from_p2p1(p2_p1, gam):
        def f(x):
            return 1 + (2 * gam / (gam + 1)) * (x ** 2 - 1) - p2_p1

        def df(x):
            return 4 * gam * x / (gam + 1)

        x0 = 1
        M1 = sc.optimize.newton(f, x0, df)
        return M1

    def calc_M1_from_T2T1(T2_T1, gam):
        def f(x):
            return 1 + ((2 * (gam - 1) * (x ** 2 - 1) * (gam * x ** 2 + 1)) / ((gam + 1) ** 2) * x ** 2) - T2_T1

        def df(x):
            return 2 * gam * x ** 3 * (2 * gam - 2) * (x ** 2 - 1) / (gam + 1) ** 2 + 2 * x ** 3 * (2 * gam - 2) * (
                        gam * x ** 2 + 1) / (gam + 1) ** 2 + 2 * x * (2 * gam - 2) * (x ** 2 - 1) * (
                        gam * x ** 2 + 1) / (gam + 1) ** 2

        x0 = 1
        M1 = sc.optimize.newton(f, x0, df)
        return M1

    def calc_M1_from_p02p01(p02_p01, gam):
        def f(x):
            return ((((gam + 1) * x ** 2) / (2 + (gam - 1) * x ** 2)) ** (gam / (gam - 1))) * (
                        ((gam + 1) / (2 * gam * x ** 2 - (gam - 1))) ** (1 / (gam - 1))) - p02_p01

        def df(x):
            return -4 * gam * x * ((gam + 1) / (2 * gam * x ** 2 - gam + 1)) ** (1 / (gam - 1)) * (
                        x ** 2 * (gam + 1) / (x ** 2 * (gam - 1) + 2)) ** (gam / (gam - 1)) / (
                        (gam - 1) * (2 * gam * x ** 2 - gam + 1)) + gam * (
                        (gam + 1) / (2 * gam * x ** 2 - gam + 1)) ** (1 / (gam - 1)) * (
                        x ** 2 * (gam + 1) / (x ** 2 * (gam - 1) + 2)) ** (gam / (gam - 1)) * (
                        x ** 2 * (gam - 1) + 2) * (
                        -2 * x ** 3 * (gam - 1) * (gam + 1) / (x ** 2 * (gam - 1) + 2) ** 2 + 2 * x * (gam + 1) / (
                            x ** 2 * (gam - 1) + 2)) / (x ** 2 * (gam - 1) * (gam + 1))

        x0 = 1
        M1 = sc.optimize.newton(f, x0, df)
        return M1

    def calc_M1_from_p02p1(p02_p1, gam):
        def f(x):
            return ((1 - gam + 2 * gam * x ** 2) / (gam + 1)) * (
                        ((((gam + 1) ** 2) * x ** 2) / (4 * gam * x ** 2 - 2 * (gam - 1))) ** (
                            gam / (gam - 1))) - p02_p1

        def df(x):
            return 4 * gam * x * (x ** 2 * (gam + 1) ** 2 / (4 * gam * x ** 2 - 2 * gam + 2)) ** (gam / (gam - 1)) / (
                        gam + 1) + gam * (x ** 2 * (gam + 1) ** 2 / (4 * gam * x ** 2 - 2 * gam + 2)) ** (
                        gam / (gam - 1)) * (
                        -8 * gam * x ** 3 * (gam + 1) ** 2 / (4 * gam * x ** 2 - 2 * gam + 2) ** 2 + 2 * x * (
                            gam + 1) ** 2 / (4 * gam * x ** 2 - 2 * gam + 2)) * (2 * gam * x ** 2 - gam + 1) * (
                        4 * gam * x ** 2 - 2 * gam + 2) / (x ** 2 * (gam - 1) * (gam + 1) ** 3)

        x0 = 1
        M1 = sc.optimize.newton(f, x0, df)
        return M1

    def M1_func(value, gam):
        return value

    if nongamma == 'M1' and value < 1.0:
        raise ValueError("M1 cannot be less than 1")

    reln_calculate = {
        'M1': M1_func,
        'M2': calc_M1_from_M2,
        'rho2_rho1': calc_M1_from_rho2rho1,
        'p2_p1': calc_M1_from_p2p1,
        'T2_T1': calc_M1_from_T2T1,
        'p02_p01': calc_M1_from_p02p01,
        'p02_p1': calc_M1_from_p02p1
    }

    M1 = reln_calculate[nongamma](value, gam)

    results = {
        'M1': M1,
        'M2': np.sqrt((2 + (gam - 1) * M1 ** 2) / (2 * gam * M1 ** 2 - (gam - 1))),
        'rho2/rho1': (((gam + 1) * M1 ** 2) / (2 + (gam - 1) * M1 ** 2)),
        'p2/p1': 1 + (2 * gam / (gam + 1)) * (M1 ** 2 - 1),
        'T2/T1': 1 + ((2 * (gam - 1) * (M1 ** 2 - 1) * (gam * M1 ** 2 + 1)) / (((gam + 1) * M1) ** 2)),
        'p02/p01': ((((gam + 1) * M1 ** 2) / (2 + (gam - 1) * M1 ** 2)) ** (gam / (gam - 1))) * (
                    ((gam + 1) / (2 * gam * M1 ** 2 - (gam - 1))) ** (1 / (gam - 1))),
        'p02/p1': ((1 - gam + 2 * gam * M1 ** 2) / (gam + 1)) * (
                    ((((gam + 1) ** 2) * M1 ** 2) / (4 * gam * M1 ** 2 - 2 * (gam - 1))) ** (gam / (gam - 1)))
    }

    return results


### Oblique Shock Relations
def oblique_shock(**kwargs):
    gam = kwargs.get('gamma')
    if gam is None:
        raise ValueError("You must provide the ratio of specific heats 'gamma'.")

    gamma = kwargs['gamma']

    nongamma_keys = [key for key in kwargs if key != 'gamma']
    nongamma_keys = [i for i in nongamma_keys if i != 'Strong_Solution']
    nongamma = nongamma_keys

    reqd_keys = ['M1', 'M1n', 'del_deg', 'wave_ang_deg']
    optional = ['Strong_Solution']

    if nongamma[0] == nongamma[1]:
        raise TypeError("You cannot use the same parameter twice!")

    for i in nongamma:
        if i not in reqd_keys:
            if i not in optional:
                raise TypeError("How")

        if i == 'M1' and kwargs[i] <= 1.0:
            raise ValueError("M1 cannot be less than or equal to 1")

    values = [kwargs[i] for i in nongamma]

    def delta_from_beta_M1(M1, beta):
        return np.arctan(((((np.sin(beta) ** 2) * (M1 ** 2) - 1)) / ((2 + (gamma + np.cos(2 * beta)) * M1 ** 2))) * (
                    2 / np.tan(beta)))

    def find_Beta_delta_max(M1):
        def f(x):
            return 4 * M1 ** 2 * sin(x) * cos(x) / ((M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x)) + 4 * M1 ** 2 * (
                        M1 ** 2 * sin(x) ** 2 - 1) * sin(2 * x) / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) ** 2 * tan(x)) + 2 * (M1 ** 2 * sin(x) ** 2 - 1) * (
                        -tan(x) ** 2 - 1) / ((M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x) ** 2)

        def df(x):
            return 16 * M1 ** 4 * sin(x) * sin(2 * x) * cos(x) / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) ** 2 * tan(x)) + 16 * M1 ** 4 * (
                        M1 ** 2 * sin(x) ** 2 - 1) * sin(2 * x) ** 2 / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) ** 3 * tan(x)) + 8 * M1 ** 2 * (-tan(x) ** 2 - 1) * sin(
                x) * cos(x) / ((M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x) ** 2) - 4 * M1 ** 2 * sin(x) ** 2 / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x)) + 4 * M1 ** 2 * cos(x) ** 2 / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x)) + 8 * M1 ** 2 * (M1 ** 2 * sin(x) ** 2 - 1) * (
                        -tan(x) ** 2 - 1) * sin(2 * x) / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) ** 2 * tan(x) ** 2) + 8 * M1 ** 2 * (
                        M1 ** 2 * sin(x) ** 2 - 1) * cos(2 * x) / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) ** 2 * tan(x)) + 2 * (M1 ** 2 * sin(x) ** 2 - 1) * (
                        -2 * tan(x) ** 2 - 2) * (-tan(x) ** 2 - 1) / (
                        (M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x) ** 3) - 2 * (M1 ** 2 * sin(x) ** 2 - 1) * (
                        2 * tan(x) ** 2 + 2) / ((M1 ** 2 * (gamma + cos(2 * x)) + 2) * tan(x))

        x0 = 0.2
        max_beta = sc.optimize.newton(f, x0, df)
        max_delta = delta_from_beta_M1(M1, max_beta)
        return [max_beta, max_delta]

    def find_Beta_delta_max2(M1n):
        def f(x):
            return 2 * (M1n ** 2 - 1) * (-tan(x) ** 2 - 1) / (
                        (M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) * tan(x) ** 2) + 2 * (M1n ** 2 - 1) * (
                        2 * M1n ** 2 * (gamma + cos(2 * x)) * cos(x) / sin(x) ** 3 + 2 * M1n ** 2 * sin(2 * x) / sin(
                    x) ** 2) / ((M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) ** 2 * tan(x))

        def df(x):
            return 2 * (M1n ** 2 - 1) * (-2 * tan(x) ** 2 - 2) * (-tan(x) ** 2 - 1) / (
                        (M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) * tan(x) ** 3) - 2 * (M1n ** 2 - 1) * (
                        2 * tan(x) ** 2 + 2) / ((M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) * tan(x)) + 4 * (
                        M1n ** 2 - 1) * (
                        2 * M1n ** 2 * (gamma + cos(2 * x)) * cos(x) / sin(x) ** 3 + 2 * M1n ** 2 * sin(2 * x) / sin(
                    x) ** 2) * (-tan(x) ** 2 - 1) / (
                        (M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) ** 2 * tan(x) ** 2) + 2 * (M1n ** 2 - 1) * (
                        -2 * M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 - 6 * M1n ** 2 * (gamma + cos(2 * x)) * cos(
                    x) ** 2 / sin(x) ** 4 + 4 * M1n ** 2 * cos(2 * x) / sin(x) ** 2 - 8 * M1n ** 2 * sin(2 * x) * cos(
                    x) / sin(x) ** 3) / ((M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) ** 2 * tan(x)) + 2 * (
                        M1n ** 2 - 1) * (
                        2 * M1n ** 2 * (gamma + cos(2 * x)) * cos(x) / sin(x) ** 3 + 2 * M1n ** 2 * sin(2 * x) / sin(
                    x) ** 2) * (
                        4 * M1n ** 2 * (gamma + cos(2 * x)) * cos(x) / sin(x) ** 3 + 4 * M1n ** 2 * sin(2 * x) / sin(
                    x) ** 2) / ((M1n ** 2 * (gamma + cos(2 * x)) / sin(x) ** 2 + 2) ** 3 * tan(x))

        x0 = 0.2
        max_beta = sc.optimize.newton(f, x0, df)
        max_delta = delta_from_beta_M1(M1n, max_beta)
        return [max_beta, max_delta]

    def calc_beta_delta(M1, M1n):
        beta = np.arcsin(M1n / M1)
        delta = delta_from_beta_M1(M1, beta)
        tmp = {
            'M1': M1,
            'M1n': M1n,
            'del_deg': (180 / np.pi) * delta,
            'wave_angle_deg': (180 / np.pi) * beta
        }
        return tmp

    def calc_M1n_delta(M1, wave_ang_deg):
        beta = wave_ang_deg * np.pi / 180
        M1n = M1 * np.sin(beta)
        delta = delta_from_beta_M1(M1, beta)
        tmp = {
            'M1': M1,
            'M1n': M1n,
            'del_deg': (180 / np.pi) * delta,
            'wave_angle_deg': (180 / np.pi) * beta
        }
        return tmp

    def calc_M1n_beta(M1, del_deg, strong):
        delta = del_deg * np.pi / 180
        gamma = 1.4

        def f(x):
            return ((((np.sin(x) ** 2) * (M1 ** 2) - 1)) / ((2 + (gamma + np.cos(2 * x)) * M1 ** 2))) * (
                        2 / np.tan(x)) - np.tan(delta)

        def df(x):
            return 4 * M1 ** 2 * np.sin(x) * np.cos(x) / (
                        (M1 ** 2 * (gamma + np.cos(2 * x)) + 2) * np.tan(x)) + 4 * M1 ** 2 * (
                        M1 ** 2 * np.sin(x) ** 2 - 1) * np.sin(2 * x) / (
                        (M1 ** 2 * (gamma + np.cos(2 * x)) + 2) ** 2 * np.tan(x)) + 2 * (
                        M1 ** 2 * np.sin(x) ** 2 - 1) * (-np.tan(x) ** 2 - 1) / (
                        (M1 ** 2 * (gamma + np.cos(2 * x)) + 2) * np.tan(x) ** 2)

        x0 = find_Beta_delta_max(M1)
        if strong:
            beta = sc.optimize.newton(f, x0[0] + 0.1, df)
            M1n = M1 * np.sin(beta)
            tmp = {
                'M1': M1,
                'M1n': M1n,
                'del_deg': (180 / np.pi) * delta,
                'wave_angle_deg': (180 / np.pi) * beta
            }
        else:
            beta = sc.optimize.newton(f, x0[0] - 0.1, df)
            M1n = M1 * np.sin(beta)
            tmp = {
                'M1': M1,
                'M1n': M1n,
                'del_deg': (180 / np.pi) * delta,
                'wave_angle_deg': (180 / np.pi) * beta
            }
        return tmp

    def calc_M1_delta(M1n, wave_angle_deg):
        beta = wave_angle_deg * np.pi / 180
        M1 = M1n / np.sin(beta)
        delta = delta_from_beta_M1(M1, beta)
        tmp = {
            'M1': M1,
            'M1n': M1n,
            'del_deg': (180 / np.pi) * delta,
            'wave_angle_deg': (180 / np.pi) * beta
        }
        return tmp

    def calc_M1_beta(M1n, del_deg, strong):
        delta = del_deg * np.pi / 180

        def f(x):
            return ((2 * (M1n ** 2 - 1)) / (
                        np.tan(x) * (2 + (gamma + np.cos(2 * x)) * (M1n / np.sin(x)) ** 2))) - np.tan(delta)

        def df(x):
            return (2 * M1n ** 2 - 1) * (-np.tan(x) ** 2 - 1) / (
                        (M1n ** 2 * (gamma + np.cos(2 * x)) / np.sin(x) ** 2 + 2) * np.tan(x) ** 2) + (
                        2 * M1n ** 2 - 1) * (
                        2 * M1n ** 2 * (gamma + np.cos(2 * x)) * np.cos(x) / np.sin(x) ** 3 + 2 * M1n ** 2 * np.sin(
                    2 * x) / np.sin(x) ** 2) / (
                        (M1n ** 2 * (gamma + np.cos(2 * x)) / np.sin(x) ** 2 + 2) ** 2 * np.tan(x))

        x0 = find_Beta_delta_max2(M1n)
        if x0[1] < delta:
            raise ValueError("Detached Shock")
            return None

        if strong:
            beta = sc.optimize.newton(f, x0[0] + 0.1, df)
            M1 = M1n / np.sin(beta)
            tmp = {
                'M1': M1,
                'M1n': M1n,
                'del_deg': (180 / np.pi) * delta,
                'wave_angle_deg': (180 / np.pi) * beta
            }
            return tmp
        else:
            beta = sc.optimize.newton(f, x0[0] - 0.1, df)
            M1 = M1n / np.sin(beta)
            tmp = {
                'M1': M1,
                'M1n': M1n,
                'del_deg': (180 / np.pi) * delta,
                'wave_angle_deg': (180 / np.pi) * beta
            }
            return tmp

    def calc_M1_M1n(wave_angle_deg, del_deg):
        beta = wave_angle_deg * np.pi / 180
        delta = del_deg * np.pi / 180
        gamma = 1.4

        def f(x):
            return (2 * ((np.sin(beta) ** 2) * (x ** 2) - 1)) / (
                        np.tan(beta) * (2 + (gamma + np.cos(2 * beta)) * x ** 2)) - np.tan(delta)

        def df(x):
            return -2 * x * (gamma + cos(2 * beta)) * (2 * x ** 2 * sin(beta) ** 2 - 2) / (
                        (x ** 2 * (gamma + cos(2 * beta)) + 2) ** 2 * tan(beta)) + 4 * x * sin(beta) ** 2 / (
                        (x ** 2 * (gamma + cos(2 * beta)) + 2) * tan(beta))

        x0 = 1.5
        M1 = sc.optimize.newton(f, x0, df)
        M1n = M1 * np.sin(beta)
        tmp = {
            'M1': M1,
            'M1n': M1n,
            'del_deg': (180 / np.pi) * delta,
            'wave_angle_deg': (180 / np.pi) * beta
        }
        return tmp

    def calc_unknown(p1, v1, p2, v2):
        param_values = {p1: v1, p2: v2}
        combos = {
            frozenset({'M1', 'M1n'}): calc_beta_delta,
            frozenset({'M1', 'wave_ang_deg'}): calc_M1n_delta,
            frozenset({'M1', 'del_deg'}): calc_M1n_beta,
            frozenset({'M1n', 'wave_angle_deg'}): calc_M1_delta,
            frozenset({'M1n', 'del_deg'}): calc_M1_beta,
            frozenset({'wave_angle_deg', 'del_deg'}): calc_M1_M1n
        }

        input_combo = frozenset(param_values.keys())

        handle = combos.get(input_combo)
        func_params = handle.__code__.co_varnames[:handle.__code__.co_argcount]
        func_params = [param for param in func_params if param != 'strong']
        args = [param_values[param] for param in func_params]

        if 'wave_angle_deg' not in nongamma:
            if not ({'M1', 'M1n'}.issubset(nongamma)):
                strong = kwargs['Strong_Solution']
                args.append(strong)

        print(args)
        return handle(*args)

    Output = calc_unknown(nongamma[1], values[1], nongamma[0], values[0])

    properties = norm_shock(M1=Output['M1n'], gamma=kwargs['gamma'])

    results = {
        'M1n': Output['M1n'],
        'M1': Output['M1'],
        'Wave angle (deg)': Output['wave_angle_deg'],
        'Delta (deg)': Output['del_deg'],
        'M2n': properties['M2'],
        'M2': properties['M2'] / np.sin((Output['wave_angle_deg'] - Output['del_deg']) * np.pi / 180),
        'rho2/rho1': properties['rho2/rho1'],
        'p2/p1': properties['p2/p1'],
        'T2/T1': properties['T2/T1'],
        'p02/p01': properties['p02/p01']
    }
    return results


# print(calc_M1_M1n(53.42294052722865, 20.0))
print(oblique_shock(M1=2.0, del_deg=20, gamma=1.4, Strong_Solution=True))