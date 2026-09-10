# analisis.py -- Practica 1: densidad de solidos y analisis de incertidumbres
# Laboratorio de Fenomenos Colectivos 2027-1, grupo 8161, equipo E6.
# Reproduce todos los calculos y las graficas del reporte.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "figure.dpi": 160})


# ------------------------------------------------------------------ utilidades
def estad(x):
    """Media, desviacion estandar muestral s y desviacion estandar de la media."""
    x = np.asarray(x, float)
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return x.mean(), s, s / np.sqrt(len(x))


def combina(*u):
    """Raiz de la suma de cuadrados de las contribuciones independientes."""
    return np.sqrt(sum(np.asarray(v, float) ** 2 for v in u))


def densidad_cilindro(m, um, d, ud, L, uL):
    """rho = 4 m / (pi d^2 L) con propagacion de incertidumbres."""
    V = np.pi * d ** 2 * L / 4
    uV = V * combina(2 * ud / d, uL / L)
    rho = m / V
    urho = rho * combina(um / m, uV / V)
    return V, uV, rho, urho


def minimos_cuadrados(x, y, ux, uy):
    """Ajuste y = m x + b por minimos cuadrados (Apendice D del curso).
    u_ef se toma como el mayor entre la incertidumbre instrumental propagada,
    sqrt(uy^2 + m^2 ux^2), y la desviacion estandar de los residuos."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    Sx, Sy, Sxx, Sxy = x.sum(), y.sum(), (x * x).sum(), (x * y).sum()
    D = n * Sxx - Sx ** 2
    m = (n * Sxy - Sx * Sy) / D
    b = (Sxx * Sy - Sxy * Sx) / D
    res = y - (m * x + b)
    s_res = np.sqrt((res ** 2).sum() / (n - 2)) if n > 2 else 0.0
    u_inst = np.sqrt(np.mean(np.asarray(uy, float) ** 2 + m ** 2 * np.asarray(ux, float) ** 2))
    u_ef = max(u_inst, s_res)
    return dict(n=n, Sx=Sx, Sy=Sy, Sxx=Sxx, Sxy=Sxy, D=D, m=m, b=b,
                u_inst=u_inst, s_res=s_res, u_ef=u_ef,
                um=u_ef * np.sqrt(n / D), ub=u_ef * np.sqrt(Sxx / D))


def reporta_ajuste(nombre, a):
    print(f"  {nombre}: n={a['n']}  Sx={a['Sx']:.4f}  Sy={a['Sy']:.4f}  "
          f"Sxx={a['Sxx']:.4f}  Sxy={a['Sxy']:.4f}  Delta={a['D']:.4f}")
    print(f"     m = {a['m']:.4f} +/- {a['um']:.4f}   b = {a['b']:.4f} +/- {a['ub']:.4f}"
          f"   (u_inst={a['u_inst']:.4f}, s_res={a['s_res']:.4f}, u_ef={a['u_ef']:.4f})")


# ================================================================ 1.1 ALAMBRE
print("=" * 70 + "\nPRACTICA 1.1  Densidad del alambre (equipo E6)")
# Primera parte: regla (division minima 1 cm), vernier (0.05 mm), granataria (0.1 g)
V1, uV1, rho1, urho1 = densidad_cilindro(0.9, 0.1, 0.066, 0.005, 33.0, 0.5)
print(f"Parte 1: V = {V1:.4f} +/- {uV1:.4f} cm3   rho = {rho1:.2f} +/- {urho1:.2f} g/cm3")
print(f"   contribuciones relativas: m {0.1/0.9:.3f}, d {2*0.005/0.066:.3f}, L {0.5/33:.3f}")

# Tercera parte: micrometro digital (0.001 mm), balanza analitica (0.1 mg), regla
d = np.array([0.0643, 0.0647, 0.0673, 0.0643, 0.0643, 0.0644, 0.0649, 0.0641,
              0.0645, 0.0642, 0.0651, 0.0644, 0.0641, 0.0638, 0.0637])   # cm
m = np.array([0.9403, 0.9405, 0.9404, 0.9404, 0.9405, 0.9405, 0.9404, 0.9405,
              0.9405, 0.9404, 0.9405, 0.9405, 0.9404, 0.9405, 0.9406])   # g
L, uL = 33.0, 0.5                                                          # cm
b_d, b_m = 0.0001, 0.00022     # resolucion del micrometro (cm) y especificacion de la balanza (g)
dm, sd, sdm = estad(d)
mm, sm, smm = estad(m)
ud = combina(sdm, b_d)
um = combina(smm, b_m)
print(f"d: media={dm:.5f} cm  s={sd:.5f}  s_media={sdm:.5f}  u_d={ud:.5f}")
print(f"m: media={mm:.5f} g   s={sm:.5f}  s_media={smm:.5f}  u_m={um:.5f}")
V3, uV3, rho3, urho3 = densidad_cilindro(mm, um, dm, ud, L, uL)
print(f"Parte 3: V = {V3:.5f} +/- {uV3:.5f} cm3   rho = {rho3:.3f} +/- {urho3:.3f} g/cm3")
cm_, cd_, cL_ = (um / mm) ** 2, (2 * ud / dm) ** 2, (uL / L) ** 2
tot = cm_ + cd_ + cL_
print(f"   fraccion de la varianza: m {cm_/tot:.3f}  d {cd_/tot:.3f}  L {cL_/tot:.3f}"
      f"   (u_rel = {np.sqrt(tot):.4f})")
print(f"   con cinta de 1 mm (u_L = 0.05 cm): u_rho = "
      f"{rho3*combina(um/mm, 2*ud/dm, 0.05/L):.3f} g/cm3")
print(f"   intervalo 1u: [{rho3-urho3:.2f}, {rho3+urho3:.2f}]   2u: [{rho3-2*urho3:.2f}, {rho3+2*urho3:.2f}]")

# Tablas grupales (seis equipos)
rho_g1 = np.array([5.89, 7.05, 21.70, 5.89, 5.10, 7.97])
rho_g3 = np.array([8.72, 8.78, 8.29, 8.65, 8.60, 8.60])
print("Grupo, parte 1: media=%.2f s=%.2f ; sin E3: media=%.2f s=%.2f"
      % (rho_g1.mean(), rho_g1.std(ddof=1),
         np.delete(rho_g1, 2).mean(), np.delete(rho_g1, 2).std(ddof=1)))
print("Grupo, parte 3: media=%.2f s=%.2f s_media=%.2f"
      % (rho_g3.mean(), rho_g3.std(ddof=1), rho_g3.std(ddof=1) / np.sqrt(6)))

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.errorbar(np.arange(1, 16), d * 10, yerr=b_d * 10, fmt="o", ms=4, capsize=2, label="lecturas")
ax.axhline(dm * 10, color="C3", label="media")
ax.axhspan((dm - sd) * 10, (dm + sd) * 10, color="C3", alpha=0.15, label="media $\\pm s$")
ax.set_xlabel("Número de lectura")
ax.set_ylabel("Diámetro $d$ (mm)")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
fig.savefig("figuras/alambre_diametro.png")

# ========================================================= 1.2 ONCE CILINDROS
print("=" * 70 + "\nPRACTICA 1.2  Once cilindros (dos repeticiones por cilindro)")
V12 = np.array([[1.460, 1.450], [1.550, 1.964], [2.316, 2.312], [2.474, 2.479],
                [2.828, 2.824], [2.122, 2.117], [3.228, 3.341], [3.630, 3.855],
                [3.957, 3.957], [4.405, 4.483], [5.444, 5.458]])            # cm3
M12 = np.array([[12.35, 12.36], [17.37, 17.38], [19.62, 19.62], [22.13, 22.12],
                [24.05, 24.05], [27.89, 27.89], [28.24, 28.24], [34.27, 34.28],
                [33.67, 33.67], [39.13, 39.13], [38.65, 38.65]])            # g
u_bal = 0.01                                    # resolucion de la balanza (g)
Vb = V12.mean(axis=1)
uV = combina(np.abs(V12[:, 0] - V12[:, 1]) / 2, 0.01 * Vb)   # repeticion + 1 % instrumental
Mb = M12.mean(axis=1)
uM = combina(np.abs(M12[:, 0] - M12[:, 1]) / 2, u_bal)
rho12 = Mb / Vb
urho12 = rho12 * combina(uM / Mb, uV / Vb)
for i in range(11):
    print(f"  cil {i+1:2d}: V = {Vb[i]:.3f} +/- {uV[i]:.3f}  m = {Mb[i]:.3f} +/- {uM[i]:.3f}"
          f"  rho = {rho12[i]:.2f} +/- {urho12[i]:.2f}")
aj_glob = minimos_cuadrados(Vb, Mb, uV, uM)
reporta_ajuste("ajuste global (11 puntos)", aj_glob)
grupos12 = {"A (latón)": [1, 3, 5, 7, 9], "B (cobre)": [4, 8, 10]}
ajustes12 = {}
for nombre, idx in grupos12.items():
    k = np.array(idx) - 1
    ajustes12[nombre] = minimos_cuadrados(Vb[k], Mb[k], uV[k], uM[k])
    reporta_ajuste(nombre, ajustes12[nombre])
    print(f"     media de densidades individuales: {rho12[k].mean():.2f}")

fig, ax = plt.subplots(figsize=(6.0, 3.8))
xx = np.linspace(0, 6, 2)
ax.plot(xx, aj_glob["m"] * xx + aj_glob["b"], "k--", lw=1,
        label=f"ajuste global: $p$ = {aj_glob['m']:.2f} g/cm$^3$")
colores = {"A (latón)": "C0", "B (cobre)": "C1"}
for nombre, idx in grupos12.items():
    k = np.array(idx) - 1
    a = ajustes12[nombre]
    ax.errorbar(Vb[k], Mb[k], xerr=uV[k], yerr=uM[k], fmt="o", ms=4, capsize=2,
                color=colores[nombre], label=f"grupo {nombre}")
    ax.plot(xx, a["m"] * xx + a["b"], color=colores[nombre], lw=1,
            label=f"ajuste {nombre[0]}: $p$ = {a['m']:.2f} g/cm$^3$")
k = np.array([2, 6, 11]) - 1
ax.errorbar(Vb[k], Mb[k], xerr=uV[k], yerr=uM[k], fmt="s", ms=4, capsize=2,
            color="C3", label="sin clasificar (2, 6, 11)")
for i in range(11):
    ax.annotate(str(i + 1), (Vb[i], Mb[i]), textcoords="offset points", xytext=(4, -9), fontsize=7)
ax.set_xlim(0, 6); ax.set_ylim(0, 45)
ax.set_xlabel("Volumen $V$ (cm$^3$)"); ax.set_ylabel("Masa $m$ (g)")
ax.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig("figuras/p12_ajuste.png")

# ==================================================== 1.3 CUATRO MATERIALES
print("=" * 70 + "\nPRACTICA 1.3  Cuatro grupos de cilindros (d y h en mm, m en g)")
datos13 = {   # (d1, h1, m1), (d2, h2, m2) por cilindro, seis cilindros por grupo
 "blancos":   [((9.989, 2.080, 0.19), (9.978, 2.084, 0.18)), ((9.797, 3.927, 0.34), (9.790, 3.944, 0.34)),
               ((9.956, 6.098, 0.54), (9.962, 6.108, 0.54)), ((9.885, 8.009, 0.72), (9.890, 8.023, 0.72)),
               ((9.982, 10.000, 0.92), (9.999, 10.093, 0.91)), ((9.943, 12.078, 1.08), (9.942, 12.090, 1.09))],
 "amarillos": [((9.690, 2.073, 1.31), (9.958, 2.072, 1.31)), ((9.960, 3.973, 2.62), (9.961, 3.977, 2.62)),
               ((9.942, 5.991, 3.93), (9.945, 6.003, 3.93)), ((9.956, 8.012, 5.28), (9.953, 8.015, 5.28)),
               ((9.992, 10.064, 6.68), (9.993, 10.061, 6.69)), ((9.992, 11.928, 7.93), (9.995, 11.940, 7.92))],
 "plateados": [((10.002, 2.054, 0.42), (9.987, 2.078, 0.42)), ((9.953, 3.996, 0.83), (9.934, 4.046, 0.83)),
               ((9.933, 6.039, 1.25), (9.939, 6.036, 1.26)), ((9.956, 8.069, 1.68), (9.948, 8.073, 1.69)),
               ((10.003, 10.022, 2.12), (9.997, 10.025, 2.12)), ((9.940, 12.018, 2.51), (9.949, 12.014, 2.51))],
 "cafés":     [((10.060, 2.080, 1.41), (10.111, 2.064, 1.41)), ((10.025, 4.004, 2.80), (10.023, 4.008, 2.80)),
               ((9.982, 6.036, 4.19), (9.991, 6.038, 4.19)), ((9.959, 7.965, 5.52), (9.954, 7.964, 5.52)),
               ((9.907, 9.891, 6.77), (9.895, 9.892, 6.77)), ((9.872, 12.084, 8.21), (9.859, 12.083, 8.21))],
}
b_mic = 0.001                                   # resolucion del micrometro (mm)
res13, aj13 = {}, {}
for grupo, cil in datos13.items():
    filas = []
    for (d1, h1, m1), (d2, h2, m2) in cil:
        dd, hh, mmm = (d1 + d2) / 2, (h1 + h2) / 2, (m1 + m2) / 2
        ud_ = combina(abs(d1 - d2) / 2, b_mic)
        uh_ = combina(abs(h1 - h2) / 2, b_mic)
        um_ = combina(abs(m1 - m2) / 2, u_bal)
        V_, uV_, r_, ur_ = densidad_cilindro(mmm, um_, dd / 10, ud_ / 10, hh / 10, uh_ / 10)
        filas.append((dd, ud_, hh, uh_, mmm, um_, V_, uV_, r_, ur_))
    t = np.array(filas)
    res13[grupo] = t
    print(f"-- {grupo}")
    for i, f in enumerate(t):
        print(f"  cil {i+1}: d={f[0]:.3f}+/-{f[1]:.3f}  h={f[2]:.3f}+/-{f[3]:.3f}  m={f[4]:.3f}+/-{f[5]:.3f}"
              f"  V={f[6]:.4f}+/-{f[7]:.4f}  rho={f[8]:.3f}+/-{f[9]:.3f}")
    aj13[grupo] = minimos_cuadrados(t[:, 6], t[:, 4], t[:, 7], t[:, 5])
    reporta_ajuste("ajuste " + grupo, aj13[grupo])
    print(f"     media de densidades individuales: {t[:, 8].mean():.3f} (s = {t[:, 8].std(ddof=1):.3f})")

fig, ax = plt.subplots(figsize=(6.0, 3.8))
xx = np.linspace(0, 1.0, 2)
for i, (grupo, t) in enumerate(res13.items()):
    a = aj13[grupo]
    ax.errorbar(t[:, 6], t[:, 4], xerr=t[:, 7], yerr=t[:, 5], fmt="o", ms=4, capsize=2, color=f"C{i}",
                label=f"{grupo}: $p$ = ({a['m']:.2f} $\\pm$ {a['um']:.2f}) g/cm$^3$")
    ax.plot(xx, a["m"] * xx + a["b"], color=f"C{i}", lw=1)
ax.set_xlim(0, 1.0); ax.set_ylim(0, 9)
ax.set_xlabel("Volumen $V$ (cm$^3$)"); ax.set_ylabel("Masa $m$ (g)")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig("figuras/p13_ajuste.png")

# ============================================================ 1.4 CACAHUATES
print("=" * 70 + "\nPRACTICA 1.4  Cuenta de cacahuates")
mc = np.array([1.42, 1.41, 1.08, 1.35, 1.02, 1.18, 1.17, 1.27, 0.47, 0.98])   # g
Mb_, uMb_ = 850.0, 15.0     # contenido neto declarado en la etiqueta (g) y tolerancia estimada
for etiqueta, x in (("los diez", mc), ("sin el cacahuate 9", np.delete(mc, 8))):
    mu, s, smu = estad(x)
    u_mu = combina(smu, u_bal)
    N = Mb_ / mu
    uN = N * combina(u_mu / mu, uMb_ / Mb_)
    print(f"  {etiqueta}: media={mu:.3f} g  s={s:.3f}  s_media={smu:.3f}  u={u_mu:.3f}"
          f"  N = {N:.1f} +/- {uN:.1f}  (2u: {2*uN:.0f})")

fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.bar(np.arange(1, 11), mc, color="C0", alpha=0.8)
ax.axhline(mc.mean(), color="C3", label=f"media = {mc.mean():.3f} g")
ax.axhspan(mc.mean() - mc.std(ddof=1), mc.mean() + mc.std(ddof=1), color="C3", alpha=0.15, label="media $\\pm s$")
ax.set_xlabel("Cacahuate"); ax.set_ylabel("Masa (g)"); ax.set_xticks(range(1, 11))
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figuras/cacahuates.png")
print("Figuras guardadas en figuras/")
