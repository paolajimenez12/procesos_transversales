
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Procesos Transversales", layout="wide")
st.title("Procesos Transversales: Producción → Inventarios → Almacenes → Transporte")
st.caption("Simula cómo se encadenan los procesos y dónde aparecen cuellos de botella.")

# ---------- Parámetros ----------
st.sidebar.header("Parámetros de simulación")

DAYS = st.sidebar.slider("Horizonte (días)", 14, 90, 30, step=1)

base_demand = st.sidebar.number_input("Demanda diaria media", 100, 5000, 800, step=50)
demand_cv    = st.sidebar.slider("Variabilidad de la demanda (CV %)", 0, 80, 20, step=5)

prod_cap     = st.sidebar.number_input("Capacidad diaria de producción (unid)", 50, 10000, 900, step=50)
prod_eff     = st.sidebar.slider("Eficiencia de producción (%)", 40, 100, 85, step=5)

ss           = st.sidebar.number_input("Stock de seguridad (unid)", 0, 10000, 1500, step=50)
rop_days     = st.sidebar.slider("Punto de pedido (días de cobertura)", 1, 30, 10, step=1)

wh_capacity  = st.sidebar.number_input("Capacidad de almacén (unid)", 100, 100000, 10000, step=500)
pick_rate    = st.sidebar.number_input("Capacidad de preparación/día (picking)", 50, 10000, 1000, step=50)

fleet_cap    = st.sidebar.number_input("Capacidad de transporte/día (unid)", 50, 10000, 900, step=50)
lead_time    = st.sidebar.slider("Lead time transporte (días)", 0, 10, 2, step=1)

np.random.seed(7)

# ---------- Generar demanda ----------
mu = base_demand
sigma = base_demand * (demand_cv/100)
demand = np.maximum(0, np.random.normal(mu, sigma, DAYS).astype(int))

# ---------- Simulación ----------
days = np.arange(1, DAYS+1)
prod_capacity_effective = int(prod_cap * (prod_eff/100))

inv = np.zeros(DAYS+1, dtype=int)  # inventario disponible al inicio del día
inv[0] = ss  # iniciamos con SS
backlog = np.zeros(DAYS+1, dtype=int)
to_ship_queue = np.zeros(DAYS+1, dtype=int)  # cola en almacén pendiente de preparar/enviar
in_transit = np.zeros(DAYS+lead_time+5, dtype=int)  # cola de transporte

produced = np.zeros(DAYS, dtype=int)
picked   = np.zeros(DAYS, dtype=int)
shipped  = np.zeros(DAYS, dtype=int)
received_client = np.zeros(DAYS, dtype=int)

orders_release = np.zeros(DAYS, dtype=int)  # reposiciones internas cuando se alcanza ROP
ROP = rop_days * mu  # punto de pedido como días de cobertura estimada

for t in range(DAYS):
    # reposición interna (simulada como producción priorizada al inventario si por debajo del ROP)
    if inv[t] + to_ship_queue[t] < ROP:
        orders_release[t] = min(prod_capacity_effective, wh_capacity - (inv[t] + to_ship_queue[t]))
    else:
        orders_release[t] = 0

    # producción del día (limitada por capacidad)
    produced[t] = min(prod_capacity_effective, wh_capacity - (inv[t] + to_ship_queue[t]))

    # actualizar inventario con producción disponible
    inv[t] += produced[t]

    # atender demanda (picking) limitado por pick_rate y stock disponible
    can_pick = min(inv[t], pick_rate, demand[t] + backlog[t])
    picked[t] = can_pick
    inv[t] -= picked[t]

    # backlog si no atendemos toda la demanda
    demand_today = demand[t] + backlog[t]
    backlog[t+1] = max(0, demand_today - picked[t])

    # preparar para envío: lo pickeado entra en cola de envío
    to_ship_queue[t] += picked[t]

    # enviar limitado por capacidad de flota
    ship_today = min(to_ship_queue[t], fleet_cap)
    shipped[t] = ship_today
    to_ship_queue[t] -= ship_today

    # entra en tránsito y llegará en t+lead_time
    if lead_time == 0:
        received_client[t] += ship_today
    else:
        in_transit[t+lead_time] += ship_today

    # llegada al cliente desde tránsito
    received_client[t] += in_transit[t]

    # pasar estado al siguiente día
    inv[t+1] = inv[t]  # lo que queda
    to_ship_queue[t+1] += to_ship_queue[t]  # cola restante

# KPIs
service_level = (received_client.sum() / max(demand.sum(),1)) * 100
util_prod = (produced.sum() / (prod_capacity_effective * DAYS)) * 100
util_pick = (picked.sum() / (pick_rate * DAYS)) * 100
util_fleet = (shipped.sum() / (fleet_cap * DAYS)) * 100

k1,k2,k3,k4 = st.columns(4)
k1.metric("Nivel de servicio", f"{service_level:.1f}%")
k2.metric("Utilización producción", f"{util_prod:.1f}%")
k3.metric("Utilización picking", f"{util_pick:.1f}%")
k4.metric("Utilización transporte", f"{util_fleet:.1f}%")

# Dataframes para gráficos
df = pd.DataFrame({
    "día": days,
    "demanda": demand,
    "producido": produced,
    "pickeado": picked,
    "enviado": shipped,
    "entregado": received_client[:DAYS],
    "inventario": inv[1:DAYS+1],
    "backlog": backlog[1:DAYS+1],
    "cola_envío": to_ship_queue[:DAYS],
})

# Gráficos lado a lado
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(px.line(df, x="día", y=["producido","demanda"], title="Producción vs Demanda"), use_container_width=True)
with c2:
    st.plotly_chart(px.line(df, x="día", y=["enviado","entregado"], title="Enviado vs Entregado"), use_container_width=True)

t1, t2 = st.columns(2)
with t1:
    st.plotly_chart(px.line(df, x="día", y=["inventario","backlog"], title="Inventario y Backlog"), use_container_width=True)
with t2:
    st.plotly_chart(px.line(df, x="día", y=["pickeado","cola_envío"], title="Picking y Cola de Envío"), use_container_width=True)

# Heatmap de utilización por proceso
util_df = pd.DataFrame({
    "Proceso": ["Producción","Picking","Transporte"],
    "Utilización %": [util_prod, util_pick, util_fleet]
})
st.subheader("Utilización por proceso")
st.plotly_chart(px.bar(util_df, x="Proceso", y="Utilización %", text="Utilización %", range_y=[0,100]), use_container_width=True)

st.markdown("""
**Cómo interpretar**  
- Si *Producción* < *Demanda* → aparecerá **backlog**.  
- Si *Picking* es menor que lo que la demanda requiere → cola en almacén.  
- Si *Transporte* es el límite → habrá **enviado** < **pickeado** y retrasos en **entregado**.  
- Ajusta el **ROP** y **Stock de seguridad** para amortiguar la variabilidad.
""")
