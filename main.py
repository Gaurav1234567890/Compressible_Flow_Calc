import tkinter as tk
from tkinter import ttk
import sympy as sp
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


def isen_calc():
    key = isen_dropdown_var.get()
    value = isen_input_entry.get()
    gam = gamma_entry.get()

    if is_float(value):
        value = float(value)
    else:
        raise ValueError("Input must be a number!")

    if is_float(gam):
        gam = float(gam)
    else:
        raise ValueError("Input must be a number!")

    kwargs = {key:value,'gamma':gam}
    isenDict = calc.isen(**kwargs)
    out = []
    for k,v in isenDict.items():
        out.append(f"{k}:{v:.4f}")
    isen_output_label.config(text="\n".join(out))




root = tk.Tk()
root.title("Compressible Flow Calculator")

isen_dropdown = ['M','p0_p','rho0_rho','T0_T','Prandtl_Meyer_Angle_(deg)','Mach_Wave_Angel_(deg)','A_As_sub','A_As_sup']
norm_shock_dropdown = ['M1','M2','rho2_rho1','p2_p1','T2_T1','p02_p01','p02_p1']
oblique_shock_dropdown = ['M1n','M1','wave_angle_deg','M2n','M2','rho2_rho1','p2_p1','T2_T1','p02_p01','p02_p1']


##isen
isen_frame = tk.Frame(root,bd=2,relief=tk.GROOVE,padx=10,pady=5)
isen_frame.pack(pady=5,fill='x')

isen_dropdown_var = tk.StringVar(value='M')
isen_dropdown_menu = ttk.Combobox(isen_frame,textvariable=isen_dropdown_var,values=isen_dropdown)
isen_dropdown_menu.pack(side='left',padx=5)

isen_input_entry = tk.Entry(isen_frame, width=10)
isen_input_entry.pack(side="left", padx=5)

gamma_label = tk.Label(isen_frame, text='Gamma: ')
gamma_label.pack(side='left',padx=5)
gamma_entry = tk.Entry(isen_frame,width=10)
gamma_entry.pack(side='left',padx=5)


isen_output_label = tk.Label(root, text=f"Outputs will appear here.", wraplength=400, justify="left", bg="lightgray")
isen_output_label.pack(pady=5, fill="x")

isen_button = tk.Button(root, text="calculate", command=isen_calc)
isen_button.pack(pady=10)

root.mainloop()
