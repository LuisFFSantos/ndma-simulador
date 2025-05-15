import streamlit as st
import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from io import BytesIO
import matplotlib.pyplot as plt

# ---- APP STREAMLIT ----
st.set_page_config(page_title="Formação de NDMA", layout="wide")
st.title("💊 Simulação da Formação de N-Nitrosaminas (NDMA)")

# Estado da sessão para controle
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

st.sidebar.header("🔧 Parâmetros da Simulação")
R2NH = st.sidebar.number_input("Concentração Inicial de Dimetilamina [M]", value=1e-3, format="%e")
NO2 = st.sidebar.number_input("Concentração Inicial de Nitrito [M]", value=6.5e-5, format="%e")
Cl = st.sidebar.number_input("Concentração de Cloreto [M]", value=1.0)
pH = st.sidebar.slider("pH do Meio", min_value=1.0, max_value=10.0, value=3.15, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Constantes Cinéticas")
KNA = st.sidebar.number_input("Ka do HNO2", value=7.079e-4, format="%e")
KECLNO = st.sidebar.number_input("Const. assoc. ClNO [M⁻²]", value=1.1e-3, format="%e")
KRCLNO = st.sidebar.number_input("Vel. ClNO [M⁻¹s⁻¹]", value=3.1e7, format="%e")
KEN2O3 = st.sidebar.number_input("Const. assoc. N2O3 [M⁻¹]", value=3e-3, format="%e")
KRN2O3 = st.sidebar.number_input("Vel. N2O3 [M⁻¹s⁻¹]", value=1.2e8, format="%e")
KRNO = st.sidebar.number_input("Vel. H2NO2⁺ [M⁻²s⁻¹]", value=7000.0)
PKA = st.sidebar.number_input("pKa da amina secundária", value=10.91)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Parâmetros de Tempo")
start_time = st.sidebar.number_input("Tempo Inicial (s)", value=0)
stop_time = st.sidebar.number_input("Tempo Final (s)", value=10000000)
dtout = st.sidebar.number_input("Passo de Integração (DT) (s)", value=100)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Unidade de Saída do NDMA")
threshold_ppb = st.sidebar.number_input("Limite Regulatório [µg/L] (EMA/ICH M7)", value=96.0)
unit = st.sidebar.selectbox("Escolher unidade de concentração final", ["mol/L", "µg/L (ppb)"])

# ---- FUNÇÃO DO MODELO CINÉTICO ----
def ndma_formation(t, y, Cl, pH, KNA, KECLNO, KRCLNO, KEN2O3, KRN2O3, KRNO, PKA):
    KA = 10**(-PKA)
    H = 10**(-pH)
    R2NH, NO2, NDMA = y
    fH = H / (H + KNA)
    fN = KA / (H + KA)
    RXN1 = KRCLNO * KECLNO * H * R2NH * NO2 * Cl * fH * fN
    RXN2 = KRN2O3 * KEN2O3 * R2NH * NO2**2 * fH**2 * fN
    RXN3 = KRNO * H * R2NH * NO2 * fH * fN
    dR2NH_dt = - (RXN1 + RXN2 + RXN3)
    dNO2_dt = - (RXN1 + RXN2 + RXN3)
    dNDMA_dt = RXN1 + RXN2 + RXN3
    return [dR2NH_dt, dNO2_dt, dNDMA_dt]

# ---- SIMULAÇÃO ----
def simulate_ndma_formation(R2NH_init, NO2_init, Cl, pH, start_time, stop_time, dt, 
                             KNA, KECLNO, KRCLNO, KEN2O3, KRN2O3, KRNO, PKA):
    y0 = [R2NH_init, NO2_init, 0.0]
    t_eval = np.arange(start_time, stop_time + dt, dt)
    t_span = (start_time, stop_time)
    sol = solve_ivp(ndma_formation, t_span, y0, args=(Cl, pH, KNA, KECLNO, KRCLNO, KEN2O3, KRN2O3, KRNO, PKA),
                    t_eval=t_eval, method='RK45')
    df = pd.DataFrame({
        "Tempo (h)": sol.t / 3600,
        "[DMA] (M)": sol.y[0],
        "[NO2-] (M)": sol.y[1],
        "[NDMA] (M)": sol.y[2]
    })
    KA = 10**(-PKA)
    H = 10**(-pH)
    fH = H / (H + KNA)
    fN = KA / (H + KA)
    RXN1 = KRCLNO * KECLNO * H * R2NH_init * NO2_init * Cl * fH * fN
    RXN2 = KRN2O3 * KEN2O3 * R2NH_init * NO2_init**2 * fH**2 * fN
    RXN3 = KRNO * H * R2NH_init * NO2_init * fH * fN
    init_rate = RXN1 + RXN2 + RXN3
    ndma_24h = init_rate * 24 * 3600
    percent_conv = (ndma_24h / NO2_init) * 100
    ndma_24h_ppb = ndma_24h * 74.08 * 1e6  # g/mol * ug/g
    # threshold_ppb agora é configurável no sidebar
    compliance = "🟢 Aprovado" if ndma_24h_ppb <= threshold_ppb else "🔴 Acima do limite"
    extra_data = {
        "Rxn ClNO [M/s]": RXN1,
        "Rxn N2O3 [M/s]": RXN2,
        "Rxn H2NO2⁺ [M/s]": RXN3,
        "Taxa Inicial Total [M/s]": init_rate,
        "[NDMA] após 24h [M]": ndma_24h,
        "[NDMA] após 24h [µg/L]": ndma_24h_ppb,
        "% Conversão NO2⁻": percent_conv,
        "Limite Regulatório (EMA/ICH M7)": f"{threshold_ppb} µg/L",
        "Status de Conformidade": compliance
    }
    return df, extra_data

if st.sidebar.button("▶️ Executar Simulação"):
    df_result, extra = simulate_ndma_formation(
        R2NH, NO2, Cl, pH,
        start_time, stop_time, dtout,
        KNA, KECLNO, KRCLNO, KEN2O3, KRN2O3, KRNO, PKA
    )
    st.session_state.df_result = df_result
    st.session_state.extra = extra
    df_result, extra = simulate_ndma_formation(
        R2NH, NO2, Cl, pH,
        start_time, stop_time, dtout,
        KNA, KECLNO, KRCLNO, KEN2O3, KRN2O3, KRNO, PKA
    )
    st.session_state.show_results = True

if st.session_state.show_results and 'df_result' in st.session_state and 'extra' in st.session_state:
    df_result = st.session_state.df_result
    extra = st.session_state.extra
    st.subheader("📈 Concentração de NDMA ao Longo do Tempo")
    fig, ax = plt.subplots()
    ax.plot(df_result["Tempo (h)"], df_result["[NDMA] (M)"], label="NDMA", color="blue")
    ax.set_xlabel("Tempo (h)")
    ax.set_ylabel("[NDMA] (M)")
    ax.set_title("Formação de NDMA")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Dados Completos da Simulação")
    st.dataframe(df_result)

    st.subheader("🧪 Cálculos Cinéticos Auxiliares")
    for key, val in extra.items():
        if unit == "µg/L (ppb)" and "[NDMA] após 24h" in key and "µg/L" in key:
            st.markdown(f"**{key}:** {val:.2f} µg/L")
        elif "[NDMA] após 24h [M]" in key and unit == "mol/L":
            st.markdown(f"**{key}:** {val:.3e} mol/L")
        elif "%" in key:
            st.markdown(f"**{key}:** {val:.5f}%")
        else:
            st.markdown(f"**{key}:** {val}")

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_result.to_excel(writer, index=False, sheet_name="NDMA Simulation")
        pd.DataFrame.from_dict(extra, orient='index', columns=['Valor']).to_excel(writer, sheet_name="Resumo Cinético")
    output.seek(0)
    st.download_button(
        label="⬇️ Baixar Resultados em Excel",
        data=output,
        file_name="ndma_simulacao.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    if st.button("🧹 Limpar Resultados"):
        st.session_state.show_results = False
