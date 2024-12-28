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


def isen_calc():
    key = isen_dropdown_var.get()
    value = isen_input_entry.get()
    gam = gamma_entry_i.get()

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
        out.append(f"{k}: {v:.4f}")
    isen_output_label.config(text="\n".join(out))


def norm_calc():
    key = norm_dropdown_var.get()
    value = norm_input_entry.get()
    gam = gamma_entry_n.get()

    if is_float(value):
        value = float(value)
    else:
        raise ValueError("Input must be a number!")

    if is_float(gam):
        gam = float(gam)
    else:
        raise ValueError("Input must be a number!")

    kwargs = {key: value, 'gamma': gam}
    normDict = calc.norm_shock(**kwargs)
    out = []
    for k, v in normDict.items():
        out.append(f"{k}: {v:.4f}")
    norm_output_label.config(text="\n".join(out))

def obli_calc():
    key_1 = obli_dropdown_var_1.get()
    value_1 = obli_input_entry_1.get()

    key_2 = obli_dropdown_var_2.get()
    value_2 = obli_input_entry_2.get()

    gam = gamma_entry_o.get()
    strong_or_weak = strong_toggle.get()

    keys = [key_1,key_2]
    for i, key in enumerate(keys):
        if key == 'turn_angle_deg':
            keys[i] = 'del_deg'

    if strong_or_weak == 1:
        Strong = True
    else:
        Strong = False

    value = [value_1,value_2]
    for i in range(len(value)):
        if is_float(value[i]):
            value[i] = float(value[i])
        else:
            raise ValueError("Input must be a number!")

    if is_float(gam):
        gam = float(gam)
    else:
        raise ValueError("Input must be a number!")

    kwargs = {keys[0]:value[0], keys[1]:value[1], 'gamma':gam, "Strong_Solution":Strong}
    print(kwargs)
    obliDict = calc.oblique_shock(**kwargs)
    out = []
    for k, v in obliDict.items():
        out.append(f"{k}: {v:.4f}")
    obli_output_label.config(text="\n".join(out))


root = tk.Tk()
root.title("Compressible Flow Calculator by Gaurav B")

isen_dropdown = ['M','p0_p','rho0_rho','T0_T','Prandtl_Meyer_Angle_(deg)','Mach_Wave_Angle_(deg)','A_As_sub','A_As_sup']
norm_shock_dropdown = ['M1','M2','rho2_rho1','p2_p1','T2_T1','p02_p01','p02_p1']
oblique_shock_dropdown = ['M1n','M1','wave_angle_deg','turn_angle_deg']  #'M2n','M2','rho2_rho1','p2_p1','T2_T1','p02_p01','p02_p1'


##isen
isen_frame = tk.Frame(root,bd=2,relief=tk.GROOVE,padx=10,pady=5)
isen_frame.pack(pady=5,fill='x')

isen_dropdown_label = tk.Label(isen_frame, text=f"Isentropic Flow Relations")
isen_dropdown_label.pack(pady=5)

isen_dropdown_var = tk.StringVar(value='M')
isen_dropdown_menu = ttk.Combobox(isen_frame,textvariable=isen_dropdown_var,values=isen_dropdown)
isen_dropdown_menu.pack(side='left',padx=5)


isen_input_entry = tk.Entry(isen_frame, width=10)
isen_input_entry.pack(side="left", padx=5)

gamma_label_i = tk.Label(isen_frame, text='Gamma: ')
gamma_label_i.pack(side='left',padx=5)
gamma_entry_i = tk.Entry(isen_frame,width=10)
gamma_entry_i.pack(side='left',padx=5)

isen_output_label = tk.Label(root, text=f"Outputs will appear here.", wraplength=400, justify="left", bg="lightgray")
isen_output_label.pack(pady=5, fill="x")

isen_button = tk.Button(root, text="calculate", command=isen_calc)
isen_button.pack(pady=10)


##normal shock
norm_frame = tk.Frame(root,bd=2,relief=tk.GROOVE,padx=10,pady=5)
norm_frame.pack(pady=5,fill='x')

norm_dropdown_label = tk.Label(norm_frame, text=f"Normal Shock Relations")
norm_dropdown_label.pack(pady=5)

norm_dropdown_var = tk.StringVar(value='M1')
norm_dropdown_menu = ttk.Combobox(norm_frame,textvariable=norm_dropdown_var,values=norm_shock_dropdown)
norm_dropdown_menu.pack(side='left',padx=5)

norm_input_entry = tk.Entry(norm_frame, width=10)
norm_input_entry.pack(side="left", padx=5)

gamma_label_n = tk.Label(norm_frame, text='Gamma: ')
gamma_label_n.pack(side='left',padx=5)
gamma_entry_n = tk.Entry(norm_frame,width=10)
gamma_entry_n.pack(side='left',padx=5)

norm_output_label = tk.Label(root, text="Outputs will appear here.", wraplength=400, justify="left", bg="lightgray")
norm_output_label.pack(pady=5, fill="x")

norm_button = tk.Button(root, text="calculate", command=norm_calc)
norm_button.pack(pady=10)


##Oblique Shocks
obli_frame = tk.Frame(root,bd=2,relief=tk.GROOVE,padx=10,pady=5)
obli_frame.pack(pady=5,fill='x')

obli_dropdown_label = tk.Label(obli_frame, text=f"Oblique Shock Relations")
obli_dropdown_label.pack(pady=5)

obli_dropdown_var_1 = tk.StringVar(value='M1')
obli_dropdown_menu_1 = ttk.Combobox(obli_frame,textvariable=obli_dropdown_var_1,values=oblique_shock_dropdown)
obli_dropdown_menu_1.pack(side='left',padx=5)

obli_input_entry_1 = tk.Entry(obli_frame, width=10)
obli_input_entry_1.pack(side="left", padx=5)

obli_dropdown_var_2 = tk.StringVar(value='M1n')
obli_dropdown_menu_2 = ttk.Combobox(obli_frame,textvariable=obli_dropdown_var_2,values=oblique_shock_dropdown)
obli_dropdown_menu_2.pack(side='left',padx=5)

obli_input_entry_2 = tk.Entry(obli_frame, width=10)
obli_input_entry_2.pack(side="left", padx=5)

gamma_label_o = tk.Label(obli_frame, text='Gamma: ')
gamma_label_o.pack(side='left',padx=5)
gamma_entry_o = tk.Entry(obli_frame,width=10)
gamma_entry_o.pack(side='left',padx=5)

obli_output_label = tk.Label(root, text="Outputs will appear here.", wraplength=400, justify="left", bg="lightgray")
obli_output_label.pack(pady=5, fill="x")

strong_toggle = tk.BooleanVar(value=False)
strong_weak = tk.Checkbutton(obli_frame, text="Strong Solution ",variable=strong_toggle)
strong_weak.pack(side='left',padx=5)

obli_button = tk.Button(root, text="calculate", command=obli_calc)
obli_button.pack(pady=10)
root.mainloop()
