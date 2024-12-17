import numpy as np
import matplotlib.pyplot as plt
import scipy as sc


'''THERMALLY PERFECT GAS ASSUMPTION'''
#constants
R = 287 #J/kgK



### Isentropic Flow Relations
def isen(**kwargs):
    # Define the keys for isentropic flow quantities
    keys = ['M', 'p0/p', 'rho0/rho', 'T0/T',
            'Prandtl-Meyer Angle (deg)', 'Mach Wave Angle (deg)',
            'A/A* (sub)', 'A/A* (sup)']


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
            return 2*gam*x*(gam/2 - 1/2)*(x**2*(gam/2 - 1/2) + 1)**(gam/(gam - 1))/((gam - 1)*(x**2*(gam/2 - 1/2) + 1))

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_rho0rho(rho0_rho, gam):
        def f(x):
            return ((1 + ((gam - 1) / 2) * x ** 2) ** (1 / (gam - 1))) - rho0_rho

        def df(x):
            return 2 * x * (gam / 2 - 1 / 2) * (x ** 2 * (gam / 2 - 1 / 2) + 1) ** (1 / (gam - 1)) / ((gam - 1) * (x ** 2 * (gam / 2 - 1 / 2) + 1))

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_t0t(T0_T, gam):
        def f(x):
            return (1 + ((gam - 1) / 2) * x ** 2)  - T0_T

        def df(x):
            return 2*x*(gam/2 - 1/2)

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M


    def calc_mach_from_prandtl_meyer(angle, gam):
        rad_ang = angle * np.pi/180
        def f(x):
            return (np.sqrt((gam + 1) / (gam - 1))) * np.arctan(
                np.sqrt((x ** 2 - 1) * ((gam + 1) / (gam - 1)))) - np.arctan(np.sqrt(x ** 2 - 1)) - rad_ang

        def df(x):
            return x * np.sqrt((gam + 1) / (gam - 1)) * np.sqrt((gam + 1) * (x ** 2 - 1) / (gam - 1)) / (
                        (1 + (gam + 1) * (x ** 2 - 1) / (gam - 1)) * (x ** 2 - 1)) - 1 / (x * np.sqrt(x ** 2 - 1))

        x0 = 1.0
        M = sc.optimize.newton(f, x0, df)
        return M

    def calc_mach_from_aa(aa, gam, supersonic=True):
        def f(x):
            return (1/x) * ((2/(gam+1)) * (1 + ((gam-1)/2)*x**2))**((gam+1)/(2*(gam-1))) - aa

        def df(x):
            return 2*(2*(x**2*(gam/2 - 1/2) + 1)/(gam + 1))**((gam + 1)/(2*gam - 2))*(gam/2 - 1/2)*(gam + 1)/((2*gam - 2)*(x**2*(gam/2 - 1/2) + 1)) - (2*(x**2*(gam/2 - 1/2) + 1)/(gam + 1))**((gam + 1)/(2*gam - 2))/x**2

        if supersonic:
            x0 = 1.5
            M = sc.optimize.newton(f, x0, df)

        else:
            x0 = 0.5
            M = sc.optimize.newton(f, x0, df)

        return M


    def calc_mach_from_mach_wave_angle(mu,gam):
        mu_rad = mu * np.pi/180
        M = 1 / np.sin(mu_rad)
        return M

    mach_calculators = {
        'M': value,
        'p0/p': calc_mach_from_p0p,
        'rho0/rho': calc_mach_from_rho0rho,
        'T0/T': calc_mach_from_t0t,
        'Prandtl-Meyer Angle (deg)': calc_mach_from_prandtl_meyer,
        'Mach Wave Angel (deg)': calc_mach_from_mach_wave_angle,
        'A/A* (sub)': lambda v, gam: calc_mach_from_aa(v, gam, supersonic=False),
        'A/A* (sup)': lambda v, gam: calc_mach_from_aa(v, gam, supersonic=True),
    }
    M = mach_calculators[nongamma](value, gam)

    results = {
        'M': M,
        'p0/p': (1 + ((gam - 1) / 2) * M ** 2) ** (gam / (gam - 1)),
        'rho0/rho': (1 + ((gam - 1) / 2) * M ** 2) ** (1 / (gam - 1)),
        'T0/T': 1 + ((gam - 1) / 2) * M ** 2,
        'Prandtl-Meyer Angle (deg)': ((np.sqrt((gam + 1) / (gam - 1))) * np.arctan(
                np.sqrt((M ** 2 - 1) * ((gam + 1) / (gam - 1)))) - np.arctan(np.sqrt(M ** 2 - 1))) * 180/np.pi,
        'Mach Wave Angle (deg)': np.arcsin(1/M) * 180/np.pi,
        'A/A* ': (1/M) * ((2/(gam+1)) * (1 + ((gam-1)/2)*M**2))**((gam+1)/(2*(gam-1))),
    }

    return results


### Normal Shock Relations





### Oblique Shock Relations

### Conical Shock Relations