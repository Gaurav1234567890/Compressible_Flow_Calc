import tkinter as tk
from tkinter import ttk
import Calculations as calc

print("Compressible Flows Calculator by: Gaurav B\n")
print("ver 1.0")
print("---------------\n")


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False



root = tk.Tk()
root.title("Compressible Flow Calculator")

isen_dropdown = ['M','p0_p','rho0_rho','T0_T','Prandtl_Meyer_Angle_(deg)','Mach_Wave_Angel_(deg)','A_As_sub','A_As_sup']
norm_shock_dropdown = ['M1','M2','rho2_rho1','p2_p1','T2_T1','p02_p01','p02_p1']
oblique_shock_dropdown = ['M1n','M1','wave_angle_deg','M2n','M2','rho2_rho1','p2_p1','T2_T1','p02_p01','p02_p1']


##isen
isen_select = tk.StringVar(root)
isen_select.set('M')

dropdown_1 = tk.OptionMenu(root, isen_select, *isen_dropdown)
dropdown_1_label = tk.Label(root,text='Isentropic Flow Relations')
dropdown_1.pack(pady=20)

input_1entry = tk.Entry(root)
input_1entry.pack(pady=5)
button = tk.Button(root, text="calculate", command=calc)