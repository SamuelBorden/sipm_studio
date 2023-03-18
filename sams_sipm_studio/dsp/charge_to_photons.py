from uncertainties import ufloat, unumpy
import numpy as np

e_charge = 1.6e-19

def sipm_photons(peak_locs, peak_err, gain):
    peak_loc_u = ufloat(peak_locs[0],peak_err[0])
    return peak_loc_u/(gain*e_charge)


h= 6.64e-34 
c=3e8



def apd_photons(charge, charge_err, photosensitivity =24, l = 560):
    charge = ufloat(charge[0], charge_err[0])
    photosensitivity = ufloat(photosensitivity,1)
    l = ufloat(l,10)
    # this conversion factor is determined by the integration over the LED spectrum 
    conversion_factor = 1.1880412517513259e+17 # in units of gamma/coulomb
    return ((charge / photosensitivity) * (l * 1e-9)) / (h * c)
#     return charge*conversion_factor

def apd_charge_photons(light_charge, light_charge_err, dark_charge, dark_charge_err):
    light = ufloat(light_charge[0], light_charge_err[0])
    dark = ufloat(dark_charge[0], dark_charge_err[0])
    M = ufloat(50 ,1)
    QE = ufloat(0.7, 0.1)
    return (light-dark)/(QE*M*e_charge)
