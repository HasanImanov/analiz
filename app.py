import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import accuracy_score, log_loss

st.set_page_config(page_title="Əmsal Analizi", page_icon="⚽", layout="wide")

NETICE_ADI = {
    "H": "1 – Ev sahibinin qələbəsi (H)",
    "D": "X – Heç-heçə (D)",
    "A": "2 – Qonağın qələbəsi (A)",
}
QOLLAR = np.arange(11)   # Poisson üçün 0-10 qol


# =====================================================
# DATA
# =====================================================
@st.cache_data
def data_yukle():
    df = pd.read_csv("butun_liqalar_temiz.csv", parse_dates=["tarix"])
    df["liqa"] = df["Country"] + " - " + df["League"]
    df = df.dropna(subset=["H", "D", "A"])
    df["hesab"] = df["FTHG"].astype(int).astype(str) + "-" + df["FTAG"].astype(int).astype(str)

    marja = 1 / df["H"] + 1 / df["D"] + 1 / df["A"]
    df["pH"], df["pD"], df["pA"] = (1 / df["H"]) / marja, (1 / df["D"]) / marja, (1 / df["A"]) / marja

    # İkinci hissə = oyunun sonu − birinci hissə
    df["2HHG"] = df["FTHG"] - df["1HHG"]
    df["2HAG"] = df["FTAG"] - df["1HAG"]
    df["2HR"] = np.select([df["2HHG"] > df["2HAG"], df["2HHG"] < df["2HAG"]], ["H", "A"], "D")
    df.loc[df["1HHG"].isna(), "2HR"] = np.nan
    df["1H_qol"] = df["1HHG"] + df["1HAG"]
    df["2H_qol"] = df["2HHG"] + df["2HAG"]
    return df.sort_values("tarix").reset_index(drop=True)


@st.cache_data
def favorit_stat(df):
    ev_fav = df["H"] <= df["A"]
    fav = pd.DataFrame({
        "emsal": np.where(ev_fav, df["H"], df["A"]).round(2),
        "geldi": np.where(ev_fav, df["FTR"] == "H", df["FTR"] == "A").astype(int),
    })
    fav["qazanc"] = np.where(fav["geldi"] == 1, fav["emsal"] - 1, -1)
    return fav


# =====================================================
# MODEL 1: ƏMSALA GÖRƏ (oxşar əmsallı oyunlar)
# =====================================================
def oxsar_tap(data, e1, ex, e2, tolerans, min_oyun):
    tol = tolerans
    while True:
        oxsar = data[(abs(data["H"] - e1) / e1 <= tol)
                     & (abs(data["D"] - ex) / ex <= tol)
                     & (abs(data["A"] - e2) / e2 <= tol)]
        if len(oxsar) >= min_oyun or tol > 0.3:
            return oxsar, tol
        tol += 0.01


# =====================================================
# MODEL 2: KOMANDALARA GÖRƏ (Poisson)
# =====================================================
def poisson_matris(lam_ev, lam_qonaq):
    """Hər hesabın ehtimalı: sətir = ev qolu, sütun = qonaq qolu."""
    return np.outer(poisson.pmf(QOLLAR, lam_ev), poisson.pmf(QOLLAR, lam_qonaq))


def matris_hda(M):
    cem = M.sum()
    return {"H": np.tril(M, -1).sum() / cem, "D": np.trace(M) / cem, "A": np.triu(M, 1).sum() / cem}


EV_Q, QONAQ_Q = np.meshgrid(QOLLAR, QOLLAR, indexing="ij")   # hər xananın ev və qonaq qolu


def matris_normal(M):
    return M / M.sum()


def matris_qollar(M):
    """Matrisdən: hər iki komanda vurur (BTTS) və 2.5-dən çox qol ehtimalı."""
    M = matris_normal(M)
    return {"btts": M[(EV_Q > 0) & (QONAQ_Q > 0)].sum(), "o25": M[(EV_Q + QONAQ_Q) > 2.5].sum()}


def netice_daxilinde_hesablar(M, netice, n=3, btts=None, ust=None, p_btts=0.5, p_ust=0.5):
    """
    Proqnoz edilən nəticəyə (H/D/A) UYĞUN gələn ən ehtimal olunan hesablar.
    btts / ust verilsə, hesab həm də 'hər iki komanda vurur' və '2.5 üst/alt' proqnozuna uyğun olur.
    Şərtlər birlikdə mümkün deyilsə (məs. 'ev udur + Bəli + Alt'), etibarı AŞAĞI olan şərt buraxılır.
    """
    M = matris_normal(M)
    maska = {"H": EV_Q > QONAQ_Q, "D": EV_Q == QONAQ_Q, "A": EV_Q < QONAQ_Q}[netice]
    sertler = []                              # (maska, etibar)
    if btts is not None:
        sertler.append((((EV_Q > 0) & (QONAQ_Q > 0)) == btts, max(p_btts, 1 - p_btts)))
    if ust is not None:
        sertler.append((((EV_Q + QONAQ_Q) > 2.5) == ust, max(p_ust, 1 - p_ust)))

    sertler.sort(key=lambda s: s[1], reverse=True)   # ən etibarlı şərt əvvəldə
    while sertler:
        tam = maska.copy()
        for s, _ in sertler:
            tam &= s
        if tam.any():
            maska = tam
            break
        sertler.pop()                          # ən az etibarlı şərti burax

    yalniz = np.where(maska, M, 0).ravel()
    sira = [k for k in yalniz.argsort()[::-1][:n] if yalniz[k] > 0.001]   # 0% hesabları göstərmə
    return [(f"{k // len(QOLLAR)}-{k % len(QOLLAR)}", yalniz[k]) for k in sira]


def etibar(p):
    if p >= 0.65:
        return "🟢 Yüksək"
    if p >= 0.55:
        return "🟡 Orta"
    return "🔴 Aşağı"


def qarisdir(a, b, w):
    """İki ehtimal lüğətini w : (1-w) nisbətində qarışdırır."""
    return {k: w * a[k] + (1 - w) * b[k] for k in a}


def komanda_lambda(df, ev, qonaq, tarix, il, ev_col, qonaq_col):
    """
    Gözlənilən qol:
      ev sahibi = (ev komandasının evdə VURDUĞU + qonaq komandanın səfərdə BURAXDIĞI) / 2
      qonaq     = (qonaq komandanın səfərdə VURDUĞU + ev komandasının evdə BURAXDIĞI) / 2
    """
    son = df[(df["tarix"] < tarix) & (df["tarix"] >= tarix - pd.Timedelta(days=365 * il))]
    e = son[son["homeTeam"] == ev].dropna(subset=[ev_col, qonaq_col])
    q = son[son["awayTeam"] == qonaq].dropna(subset=[ev_col, qonaq_col])
    if len(e) < 5 or len(q) < 5:
        return None
    lam_ev = (e[ev_col].mean() + q[ev_col].mean()) / 2
    lam_q = (q[qonaq_col].mean() + e[qonaq_col].mean()) / 2
    return lam_ev, lam_q, len(e), len(q)


# =====================================================
# GÖSTƏRİŞ FUNKSİYALARI
# =====================================================
def yekun_goster(ehtimallar, basliq):
    sirali = sorted(ehtimallar.items(), key=lambda x: x[1], reverse=True)
    ferq = sirali[0][1] - sirali[1][1]
    metn = (f"**{basliq}: {NETICE_ADI[sirali[0][0]]}**, {sirali[0][1]:.1%}\n\n"
            f"Sonra: {NETICE_ADI[sirali[1][0]]} ({sirali[1][1]:.1%}), "
            f"{NETICE_ADI[sirali[2][0]]} ({sirali[2][1]:.1%})")
    if ferq < 0.05:
        st.warning(metn + "\n\n⚖️ İlk iki nəticə arasındakı fərq 5%-dən azdır: çox bərabərdir.")
    elif sirali[0][1] >= 0.6:
        st.success(metn + "\n\n💪 Aydın favorit var.")
    else:
        st.info(metn)


def hda_metrikalar(ehtimallar, bukmeker=None):
    cols = st.columns(3)
    for col, k in zip(cols, "HDA"):
        delta = None if bukmeker is None else f"{(ehtimallar[k] - bukmeker[k]) * 100:+.1f} faiz bəndi vs bukmeker"
        col.metric(NETICE_ADI[k], f"{ehtimallar[k]:.1%}", delta=delta)


def hisse_qol_cedveli(d, vurdu_1, buraxdi_1, vurdu_2, buraxdi_2):
    return pd.DataFrame({
        "1-ci hissə": [d[vurdu_1].mean(), d[buraxdi_1].mean(), (d[vurdu_1] > 0).mean() * 100],
        "2-ci hissə": [d[vurdu_2].mean(), d[buraxdi_2].mean(), (d[vurdu_2] > 0).mean() * 100],
    }, index=["Orta vurduğu qol", "Orta buraxdığı qol", "Qol vurduğu oyun %"]).round(2)


df = data_yukle()
fav = favorit_stat(df)

# =====================================================
# SOL PANEL
# =====================================================
st.sidebar.header("👥 Komandalar")
liqalar = ["Bütün liqalar"] + sorted(df["liqa"].unique())
liqa = st.sidebar.selectbox("Liqa", liqalar)
komanda_data = df if liqa == "Bütün liqalar" else df[df["liqa"] == liqa]
komandalar = sorted(set(komanda_data["homeTeam"]) | set(komanda_data["awayTeam"]))

ev_komanda = st.sidebar.selectbox("🏠 Ev sahibi", komandalar, index=None, placeholder="Adını yaz və ya seç...")
qonaq_komanda = st.sidebar.selectbox("✈️ Qonaq", komandalar, index=None, placeholder="Adını yaz və ya seç...")
st.sidebar.caption("💡 Siyahıda adı yazmağa başla, axtarış avtomatik işləyir (məs. 'barc').")

st.sidebar.header("💰 Əmsallar (istəyə bağlı)")
e1 = st.sidebar.number_input("1 (ev sahibi)", min_value=1.01, value=None, step=0.01, format="%.2f", placeholder="məs. 1.60")
ex = st.sidebar.number_input("X (heç-heçə)", min_value=1.01, value=None, step=0.01, format="%.2f", placeholder="məs. 4.00")
e2 = st.sidebar.number_input("2 (qonaq)", min_value=1.01, value=None, step=0.01, format="%.2f", placeholder="məs. 5.50")

with st.sidebar.expander("➕ Qol əmsalları (istəyə bağlı)"):
    st.caption("Yazsan, 'hər iki komanda vurur' və '2.5 üst/alt' proqnozu daha dəqiq olur.")
    btts_y = st.number_input("Hər iki komanda vurur: Bəli", min_value=1.01, value=None, step=0.01, format="%.2f")
    btts_n = st.number_input("Hər iki komanda vurur: Xeyr", min_value=1.01, value=None, step=0.01, format="%.2f")
    ust25 = st.number_input("2.5 Üst", min_value=1.01, value=None, step=0.01, format="%.2f")
    alt25 = st.number_input("2.5 Alt", min_value=1.01, value=None, step=0.01, format="%.2f")

with st.sidebar.expander("⚙️ Ayarlar"):
    tolerans = st.slider("Əmsal oxşarlığı (±%)", 1, 20, 4) / 100
    min_oyun = st.slider("Minimum oxşar oyun", 10, 200, 40)
    il = st.slider("Komanda modeli: son neçə il", 1, 5, 2)

emsal_var = None not in (e1, ex, e2)
komanda_var = ev_komanda is not None and qonaq_komanda is not None

# =====================================================
# BAŞLIQ
# =====================================================
st.title("⚽ Əmsal və Komanda Analizi")
basliq = f"{ev_komanda or '?'} vs {qonaq_komanda or '?'}"
st.caption(f"Data: {len(df):,} oyun, {df['liqa'].nunique()} liqa")

if not emsal_var and not komanda_var:
    st.info("👈 Sol paneldə **iki komanda seç** və/və ya **üç əmsalı (1, X, 2) yaz**. "
            "Hər birini ayrıca da istifadə edə bilərsən.")
    if ev_komanda or qonaq_komanda:
        st.caption("İki komandanın hər ikisini seçməlisən.")
    if any(v is not None for v in (e1, ex, e2)) and not emsal_var:
        st.caption("Əmsalların üçünü də yazmalısan.")
if komanda_var or emsal_var:
    st.subheader(f"🆚 {basliq}" if komanda_var else "💰 Yalnız əmsala görə analiz")

# ----- Model 1 hesablamaları -----
if emsal_var:
    data = df if liqa == "Bütün liqalar" else df[df["liqa"] == liqa]
    oxsar, tol = oxsar_tap(data, e1, ex, e2, tolerans, min_oyun)
    marja = 1 / e1 + 1 / ex + 1 / e2
    bukmeker = {"H": (1 / e1) / marja, "D": (1 / ex) / marja, "A": (1 / e2) / marja}
    if len(oxsar) == 0:
        st.warning("Bu əmsallarla keçmişdə oyun tapılmadı. Oxşarlığı artır və ya başqa liqa seç.")
        emsal_var = False
else:
    bukmeker = None

# ----- Model 2 hesablamaları -----
if komanda_var:
    indi = df["tarix"].max() + pd.Timedelta(days=1)
    k_son = komanda_lambda(df, ev_komanda, qonaq_komanda, indi, il, "FTHG", "FTAG")
    k_1h = komanda_lambda(df, ev_komanda, qonaq_komanda, indi, il, "1HHG", "1HAG")
    k_2h = komanda_lambda(df, ev_komanda, qonaq_komanda, indi, il, "2HHG", "2HAG")
    if k_son is None:
        st.warning(f"Son {il} ildə komandaların hər birinin ən azı 5 oyunu olmalıdır "
                   f"({ev_komanda} evdə, {qonaq_komanda} səfərdə). Ayarlardan ili artır.")

# =====================================================
# 📌 YEKUN PROQNOZ (bütün üsulları birləşdirir)
# =====================================================
# Test nəticəsinə görə: əmsal + komanda olanda 80% əmsal, 20% komanda ən yaxşısıdır
W_EMSAL = 0.8
komanda_ok = komanda_var and k_son is not None

if emsal_var or komanda_ok:
    # --- Nəticə (H/D/A) ---
    if emsal_var:
        real_o = oxsar["FTR"].value_counts(normalize=True)
        p_emsal = {k: real_o.get(k, 0) for k in "HDA"}
        M_emsal = poisson_matris(oxsar["FTHG"].mean(), oxsar["FTAG"].mean())
        q_emsal = {"btts": ((oxsar["FTHG"] > 0) & (oxsar["FTAG"] > 0)).mean(),
                   "o25": ((oxsar["FTHG"] + oxsar["FTAG"]) > 2.5).mean()}
        hd_o = oxsar.dropna(subset=["1HR"])
        r1 = hd_o["1HR"].value_counts(normalize=True)
        h1_emsal = {k: r1.get(k, 0) for k in "HDA"} if len(hd_o) else None
    if komanda_ok:
        M_kom = poisson_matris(k_son[0], k_son[1])
        p_kom = matris_hda(M_kom)
        q_kom = matris_qollar(M_kom)
        h1_kom = matris_hda(poisson_matris(k_1h[0], k_1h[1])) if k_1h else None

    if emsal_var and komanda_ok:
        p_yekun = qarisdir(p_emsal, p_kom, W_EMSAL)
        M_yekun = W_EMSAL * matris_normal(M_emsal) + (1 - W_EMSAL) * matris_normal(M_kom)
        q_yekun = qarisdir(q_emsal, q_kom, 0.5)
        h1_yekun = qarisdir(h1_emsal, h1_kom, W_EMSAL) if (h1_emsal and h1_kom) else (h1_emsal or h1_kom)
        menbe = "əmsallar (80%) + komandalar (20%)"
    elif emsal_var:
        p_yekun, M_yekun, q_yekun, h1_yekun = p_emsal, M_emsal, dict(q_emsal), h1_emsal
        menbe = f"əmsalları oxşar olan {len(oxsar)} keçmiş oyun"
    else:
        p_yekun, M_yekun, q_yekun, h1_yekun = p_kom, M_kom, dict(q_kom), h1_kom
        menbe = f"komandaların son {il} ildəki oyunları (Poisson)"

    # Bukmekerin qol əmsalları yazılıbsa, onlar daha dəqiqdir (testdə ən yaxşı nəticə)
    qol_menbe = []
    if btts_y and btts_n:
        q_yekun["btts"] = (1 / btts_y) / (1 / btts_y + 1 / btts_n)
        qol_menbe.append("BTTS əmsalı")
    if ust25 and alt25:
        q_yekun["o25"] = (1 / ust25) / (1 / ust25 + 1 / alt25)
        qol_menbe.append("2.5 əmsalı")

    netice = max(p_yekun, key=p_yekun.get)
    btts_beli = bool(q_yekun["btts"] >= 0.5)
    ust = bool(q_yekun["o25"] >= 0.5)
    # Hesab hər üç proqnoza uyğun seçilir: nəticə + hər iki komanda vurur + 2.5 üst/alt
    hesablar_top = netice_daxilinde_hesablar(M_yekun, netice, 3, btts=btts_beli, ust=ust,
                                             p_btts=q_yekun["btts"], p_ust=q_yekun["o25"])
    # Hesab hansı proqnozla uyğun gəlmir? (şərtlər birlikdə mümkün olmayanda)
    _e, _q = map(int, hesablar_top[0][0].split("-"))
    uygun_deyil = []
    if ((_e > 0) and (_q > 0)) != btts_beli:
        uygun_deyil.append("hər iki komanda vurur")
    if ((_e + _q) > 2.5) != ust:
        uygun_deyil.append("2.5 qol")

    with st.container(border=True):
        st.subheader("📌 Yekun proqnoz")
        y1, y2, y3, y4, y5 = st.columns(5)
        y1.metric("🏁 Nəticə", NETICE_ADI[netice].split(" (")[0], f"{p_yekun[netice]:.0%} · {etibar(p_yekun[netice])}",
                  delta_color="off")
        y2.metric("🎯 Gözlənilən hesab", hesablar_top[0][0], f"{hesablar_top[0][1]:.0%} ehtimal", delta_color="off")
        # Hesabla ziddiyyət təşkil edən zəif proqnoz "Qeyri-müəyyən" kimi göstərilir
        p_b = q_yekun["btts"] if btts_beli else 1 - q_yekun["btts"]
        if "hər iki komanda vurur" in uygun_deyil:
            y3.metric("⚽ Hər iki komanda vurur?", "⚖️ Qeyri-müəyyən",
                      f"Bəli {q_yekun['btts']:.0%} / Xeyr {1 - q_yekun['btts']:.0%}", delta_color="off")
        else:
            y3.metric("⚽ Hər iki komanda vurur?", "Bəli" if btts_beli else "Xeyr", f"{p_b:.0%} · {etibar(p_b)}",
                      delta_color="off")
        p_u = q_yekun["o25"] if ust else 1 - q_yekun["o25"]
        if "2.5 qol" in uygun_deyil:
            y4.metric("📈 2.5 qol", "⚖️ Qeyri-müəyyən",
                      f"Üst {q_yekun['o25']:.0%} / Alt {1 - q_yekun['o25']:.0%}", delta_color="off")
        else:
            y4.metric("📈 2.5 qol", "Üst" if ust else "Alt", f"{p_u:.0%} · {etibar(p_u)}", delta_color="off")
        if h1_yekun:
            n1 = max(h1_yekun, key=h1_yekun.get)
            y5.metric("⏱️ 1-ci hissə", NETICE_ADI[n1].split(" – ")[0] + " – " +
                      {"H": "Ev", "D": "Heç-heçə", "A": "Qonaq"}[n1],
                      f"{h1_yekun[n1]:.0%} · {etibar(h1_yekun[n1])}", delta_color="off")
        else:
            y5.metric("⏱️ 1-ci hissə", "—", "məlumat yoxdur", delta_color="off")

        st.markdown(
            f"**{NETICE_ADI[netice]}**: {p_yekun[netice]:.1%} · "
            f"Digərləri: " + ", ".join(f"{NETICE_ADI[k].split(' – ')[0]} {p_yekun[k]:.1%}" for k in "HDA" if k != netice))
        st.markdown("**Proqnozlara uyğun ən ehtimal olunan hesablar:** " +
                    " · ".join(f"**{h}** ({p:.1%})" for h, p in hesablar_top))
        tek = netice_daxilinde_hesablar(M_yekun, netice, 1)[0]
        if tek[0] != hesablar_top[0][0]:
            st.caption(f"ℹ️ Tək hesab kimi ən ehtimallısı **{tek[0]}** ({tek[1]:.1%}) olardı, amma o, "
                       f"digər proqnozlarla (hər iki komanda vurur: {'Bəli' if btts_beli else 'Xeyr'}, "
                       f"2.5 {'Üst' if ust else 'Alt'}) uyğun gəlmir.")
        if uygun_deyil:
            st.caption(f"⚖️ '{uygun_deyil[0].capitalize()}' proqnozu təxminən 50/50-dir və digər proqnozlarla "
                       f"birlikdə mümkün deyil, ona görə 'qeyri-müəyyən' göstərilir. Hesab daha etibarlı "
                       f"proqnozlara görə seçilib.")
        st.caption(f"Mənbə: {menbe}" + (f" · qol proqnozu: {', '.join(qol_menbe)}" if qol_menbe else "") +
                   " · Testdə: nəticə ~50%, hər iki komanda vurur ~55%, 2.5 qol ~57%, dəqiq hesab ~12% düz çıxır "
                   "(ətraflı: 📏 Dəqiqlik tabı).")

tab_adlari = ["🎯 Proqnoz", "⏱️ Hissələr", "📊 Favorit statistikası", "📋 Oxşar oyunlar", "📏 Dəqiqlik"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_adlari)

# =====================================================
# TAB 1: PROQNOZ
# =====================================================
with tab1:
    if emsal_var:
        st.header("💰 Əmsala görə")
        real = oxsar["FTR"].value_counts(normalize=True)
        p_emsal = {k: real.get(k, 0) for k in "HDA"}
        st.caption(f"Əmsalları oxşar (±{tol:.0%}) olan **{len(oxsar)} keçmiş oyun** · bukmeker marjası {(marja - 1):.1%}")
        yekun_goster(p_emsal, "Ən ehtimal olunan nəticə")
        hda_metrikalar(p_emsal, bukmeker)

        h1, h2 = st.columns([2, 1])
        hesablar = oxsar["hesab"].value_counts().head(10).rename_axis("Hesab").reset_index(name="Oyun")
        hesablar["Faiz"] = (hesablar["Oyun"] / len(oxsar) * 100).round(1)
        h1.markdown("**Ən çox gələn hesablar**")
        h1.bar_chart(hesablar.set_index("Hesab")["Faiz"], y_label="%")
        h2.dataframe(hesablar, hide_index=True, width="stretch")

        cem = oxsar["FTHG"] + oxsar["FTAG"]
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Ev sahibinin orta qolu", f"{oxsar['FTHG'].mean():.2f}")
        q2.metric("Qonağın orta qolu", f"{oxsar['FTAG'].mean():.2f}")
        q3.metric("2.5-dən çox qol", f"{(cem > 2.5).mean():.1%}")
        q4.metric("Hər iki komanda vurdu", f"{((oxsar['FTHG'] > 0) & (oxsar['FTAG'] > 0)).mean():.1%}")

    if komanda_var and k_son:
        if emsal_var:
            st.divider()
        st.header("👥 Komandalara görə (Poisson modeli)")
        lam_ev, lam_q, n_e, n_q = k_son
        st.caption(f"Son {il} il: **{ev_komanda}** evdə {n_e} oyun, **{qonaq_komanda}** səfərdə {n_q} oyun")

        M = poisson_matris(lam_ev, lam_q)
        p_kom = matris_hda(M)
        yekun_goster(p_kom, "Ən ehtimal olunan nəticə")
        hda_metrikalar(p_kom, bukmeker)

        g1, g2 = st.columns(2)
        g1.metric(f"{ev_komanda}: gözlənilən qol", f"{lam_ev:.2f}")
        g2.metric(f"{qonaq_komanda}: gözlənilən qol", f"{lam_q:.2f}")

        hesab_list = [(f"{i}-{j}", M[i, j] / M.sum() * 100) for i in range(6) for j in range(6)]
        hesab_df = pd.DataFrame(hesab_list, columns=["Hesab", "Faiz"]).sort_values("Faiz", ascending=False).head(10)
        hesab_df["Faiz"] = hesab_df["Faiz"].round(1)
        h1, h2 = st.columns([2, 1])
        h1.markdown("**Ən ehtimal olunan hesablar**")
        h1.bar_chart(hesab_df.set_index("Hesab")["Faiz"], y_label="%", sort=False)
        h2.dataframe(hesab_df, hide_index=True, width="stretch")

        i, j = np.meshgrid(QOLLAR, QOLLAR, indexing="ij")
        q1, q2 = st.columns(2)
        q1.metric("2.5-dən çox qol", f"{M[(i + j) > 2.5].sum() / M.sum():.1%}")
        q2.metric("Hər iki komanda vurur", f"{M[(i > 0) & (j > 0)].sum() / M.sum():.1%}")

        with st.expander("ℹ️ Bu model necə işləyir?"):
            st.markdown(
                f"- **{ev_komanda}** evdə orta hesabla nə qədər qol **vurur**, **{qonaq_komanda}** səfərdə nə qədər "
                f"**buraxır**: bu ikisinin ortası = ev sahibinin gözlənilən qolu (**{lam_ev:.2f}**).\n"
                f"- Eyni qayda ilə qonağın gözlənilən qolu: **{lam_q:.2f}**.\n"
                "- **Poisson paylanması** bu ortalamalardan hər hesabın (0-0, 1-0, 2-1...) ehtimalını çıxarır.\n"
                "- Hesabların ehtimallarını toplayanda H / D / A ehtimalları alınır.\n"
                "- ⚠️ Rəqiblərin gücü nəzərə alınmır. Liqalar fərqlidirsə (məs. LaLiga vs Premyer Liqa), "
                "nəticə daha az etibarlıdır.")

    if emsal_var and komanda_var and k_son:
        st.divider()
        st.header("⚖️ İki üsulun müqayisəsi")
        muq = pd.DataFrame({
            "Əmsala görə": [p_emsal[k] * 100 for k in "HDA"],
            "Komandalara görə": [p_kom[k] * 100 for k in "HDA"],
            "Bukmeker": [bukmeker[k] * 100 for k in "HDA"],
        }, index=[NETICE_ADI[k] for k in "HDA"]).round(1)
        st.dataframe(muq, width="stretch")
        st.bar_chart(muq, stack=False, y_label="%")

    if not emsal_var and not (komanda_var and k_son):
        st.caption("Proqnoz üçün məlumat daxil edilməyib.")

# =====================================================
# TAB 2: HİSSƏLƏR
# =====================================================
with tab2:
    if emsal_var:
        st.header("💰 Əmsala görə")
        hd = oxsar.dropna(subset=["1HHG", "1HAG"])
        st.caption(f"Birinci hissə məlumatı olan oxşar oyun: {len(hd)}")
        if len(hd):
            k1, k2 = st.columns(2)
            p1 = hd["1HR"].value_counts(normalize=True)
            p2 = hd["2HR"].value_counts(normalize=True)
            with k1:
                st.subheader("1️⃣ Birinci hissə")
                yekun_goster({k: p1.get(k, 0) for k in "HDA"}, "1-ci hissədə ən ehtimal olunan")
            with k2:
                st.subheader("2️⃣ İkinci hissə")
                yekun_goster({k: p2.get(k, 0) for k in "HDA"}, "2-ci hissədə ən ehtimal olunan")

            g1, g2, g3, g4, g5 = st.columns(5)
            g1.metric("1-ci hissə orta qol", f"{hd['1H_qol'].mean():.2f}")
            g2.metric("2-ci hissə orta qol", f"{hd['2H_qol'].mean():.2f}")
            g3.metric("1-ci hissədə çox qol", f"{(hd['1H_qol'] > hd['2H_qol']).mean():.1%}")
            g4.metric("2-ci hissədə çox qol", f"{(hd['1H_qol'] < hd['2H_qol']).mean():.1%}")
            g5.metric("Bərabər", f"{(hd['1H_qol'] == hd['2H_qol']).mean():.1%}")
            cox = "2-ci" if hd["2H_qol"].mean() > hd["1H_qol"].mean() else "1-ci"
            st.info(f"Bu tip oyunlarda qolların çoxu **{cox} hissədə** vurulur.")

            st.markdown("**🔁 Birinci hissə / Oyunun sonu**")
            qisa = {"H": "1", "D": "X", "A": "2"}
            kombo = (hd["1HR"].map(qisa) + " / " + hd["FTR"].map(qisa)).value_counts(normalize=True) * 100
            kombo = kombo.round(1).rename_axis("1-ci hissə / Son").reset_index(name="Faiz")
            b1, b2 = st.columns([2, 1])
            b1.bar_chart(kombo.set_index("1-ci hissə / Son")["Faiz"], y_label="%")
            b2.dataframe(kombo, hide_index=True, width="stretch")
            st.caption("Məsələn, **X / 1**: birinci hissə heç-heçə bitib, oyunu ev sahibi udub.")

    if komanda_var:
        if emsal_var:
            st.divider()
        st.header("👥 Komandalara görə")

        if k_1h and k_2h:
            k1, k2 = st.columns(2)
            with k1:
                st.subheader("1️⃣ Birinci hissə")
                yekun_goster(matris_hda(poisson_matris(k_1h[0], k_1h[1])), "1-ci hissədə ən ehtimal olunan")
                st.caption(f"Gözlənilən qol: {ev_komanda} {k_1h[0]:.2f} – {k_1h[1]:.2f} {qonaq_komanda}")
            with k2:
                st.subheader("2️⃣ İkinci hissə")
                yekun_goster(matris_hda(poisson_matris(k_2h[0], k_2h[1])), "2-ci hissədə ən ehtimal olunan")
                st.caption(f"Gözlənilən qol: {ev_komanda} {k_2h[0]:.2f} – {k_2h[1]:.2f} {qonaq_komanda}")
            cem1, cem2 = k_1h[0] + k_1h[1], k_2h[0] + k_2h[1]
            st.info(f"Bu oyunda qolların çoxu **{'2-ci' if cem2 > cem1 else '1-ci'} hissədə** gözlənilir "
                    f"(1-ci hissə: {cem1:.2f}, 2-ci hissə: {cem2:.2f} qol).")
        else:
            st.caption("Hissə proqnozu üçün kifayət qədər oyun yoxdur.")

        ev_oyun = df[df["homeTeam"] == ev_komanda].dropna(subset=["1HHG"])
        q_oyun = df[df["awayTeam"] == qonaq_komanda].dropna(subset=["1HHG"])
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**🏠 {ev_komanda}: evdə {len(ev_oyun)} oyun (bütün illər)**")
            if len(ev_oyun):
                st.dataframe(hisse_qol_cedveli(ev_oyun, "1HHG", "1HAG", "2HHG", "2HAG"), width="stretch")
        with c2:
            st.markdown(f"**✈️ {qonaq_komanda}: səfərdə {len(q_oyun)} oyun (bütün illər)**")
            if len(q_oyun):
                st.dataframe(hisse_qol_cedveli(q_oyun, "1HAG", "1HHG", "2HAG", "2HHG"), width="stretch")

        h2h = df[((df["homeTeam"] == ev_komanda) & (df["awayTeam"] == qonaq_komanda))
                 | ((df["homeTeam"] == qonaq_komanda) & (df["awayTeam"] == ev_komanda))]
        st.markdown(f"**⚔️ Qarşılıqlı oyunlar: {len(h2h)}**")
        if len(h2h):
            h = h2h.sort_values("tarix", ascending=False).copy()
            h["1-ci hissə"] = h["1HHG"].astype("Int64").astype(str) + "-" + h["1HAG"].astype("Int64").astype(str)
            h["Nəticə"] = h["FTR"].map(NETICE_ADI)
            st.dataframe(h[["tarix", "liqa", "homeTeam", "awayTeam", "1-ci hissə", "hesab", "Nəticə"]]
                         .rename(columns={"tarix": "Tarix", "liqa": "Turnir", "homeTeam": "Ev",
                                          "awayTeam": "Qonaq", "hesab": "Son"}),
                         hide_index=True, width="stretch")

    if not emsal_var and not komanda_var:
        st.caption("Hissə analizi üçün məlumat daxil edilməyib.")

# =====================================================
# TAB 3: FAVORİT STATİSTİKASI (yalnız əmsal olanda)
# =====================================================
with tab3:
    if emsal_var:
        fav_emsal = round(min(e1, e2), 2)
        st.subheader(f"Favorit: {'ev sahibi' if e1 <= e2 else 'qonaq'}, əmsal {fav_emsal}")
        bu = fav[fav["emsal"] == fav_emsal]
        if len(bu):
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Oyun", len(bu))
            s2.metric("Gəldi", int(bu["geldi"].sum()))
            s3.metric("Yandı", int(len(bu) - bu["geldi"].sum()))
            s4.metric("Real faiz", f"{bu['geldi'].mean():.1%}", delta=f"vəd: {1 / fav_emsal:.1%}", delta_color="off")
            s5.metric("ROI", f"{bu['qazanc'].mean() * 100:+.1f}%")
            if len(bu) < 100:
                st.warning(f"Bu əmsalla cəmi {len(bu)} oyun var: statistika etibarsız ola bilər.")
    else:
        st.caption("Bu tab üçün əmsalları yazmalısan. Aşağıdakı ümumi cədvəllər isə həmişə görünür.")

    st.subheader("Aralıqlar üzrə (bütün liqalar)")
    fav_ar = fav.copy()
    fav_ar["Aralıq"] = pd.cut(fav_ar["emsal"], bins=[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 5]).astype(str)
    araliq = fav_ar.groupby("Aralıq", sort=False).agg(
        Oyun=("geldi", "size"), Gəldi=("geldi", "sum"), Real_faiz=("geldi", "mean"), ROI=("qazanc", "mean"))
    araliq = araliq.loc[[a for a in araliq.index if a != "nan"]].sort_index()
    araliq["Yandı"] = araliq["Oyun"] - araliq["Gəldi"]
    araliq["Real_faiz"] = (araliq["Real_faiz"] * 100).round(1)
    araliq["ROI"] = (araliq["ROI"] * 100).round(2)
    st.dataframe(araliq[["Oyun", "Gəldi", "Yandı", "Real_faiz", "ROI"]], width="stretch")
    st.bar_chart(araliq["ROI"], y_label="ROI %")
    st.caption("ROI = hər oyuna 1 AZN qoysaydın, qoyduğun pulun neçə faizini qazanardın (+) və ya itirərdin (−).")

    st.subheader("Konkret əmsallar (ən azı 100 oyun)")
    konkret = fav.groupby("emsal").agg(Oyun=("geldi", "size"), Gəldi=("geldi", "sum"), ROI=("qazanc", "mean"))
    konkret = konkret[konkret["Oyun"] >= 100]
    konkret["Yandı"] = konkret["Oyun"] - konkret["Gəldi"]
    konkret["Real_faiz"] = (konkret["Gəldi"] / konkret["Oyun"] * 100).round(1)
    konkret["Vəd_faiz"] = (100 / konkret.index).round(1)
    konkret["ROI"] = (konkret["ROI"] * 100).round(2)
    secim = st.radio("Sırala:", ["Əmsala görə", "Ən çox uduzduran", "Ən az uduzduran"], horizontal=True)
    if secim == "Ən çox uduzduran":
        konkret = konkret.sort_values("ROI")
    elif secim == "Ən az uduzduran":
        konkret = konkret.sort_values("ROI", ascending=False)
    st.dataframe(konkret[["Oyun", "Gəldi", "Yandı", "Real_faiz", "Vəd_faiz", "ROI"]], width="stretch")
    st.caption("⚠️ 100-300 oyunda ±15% ROI dalğalanması təsadüfi ola bilər. Aralıqlar cədvəli daha etibarlıdır.")

# =====================================================
# TAB 4: OXŞAR OYUNLAR
# =====================================================
with tab4:
    if emsal_var:
        st.subheader(f"Oxşar əmsallı {len(oxsar)} oyun")
        goster = oxsar.sort_values("tarix", ascending=False)[
            ["tarix", "liqa", "homeTeam", "awayTeam", "H", "D", "A", "hesab", "FTR"]].copy()
        goster["FTR"] = goster["FTR"].map(NETICE_ADI)
        goster.columns = ["Tarix", "Liqa", "Ev", "Qonaq", "1", "X", "2", "Hesab", "Nəticə"]
        st.dataframe(goster, hide_index=True, width="stretch")
    else:
        st.caption("Bu tab üçün əmsalları yazmalısan.")


# =====================================================
# TAB 5: DƏQİQLİK (modelləri keçmiş oyunlarda yoxlamaq)
# =====================================================
@st.cache_data(show_spinner=False)
def deqiqlik_hesabla(df, il):
    SERHED = pd.Timestamp("2025-07-01")
    train, test = df[df["tarix"] < SERHED], df[df["tarix"] >= SERHED]
    y_hamisi = test["FTR"].map({"H": 0, "D": 1, "A": 2}).values

    # Bukmeker
    bukm = test[["pH", "pD", "pA"]].values

    hesab_real = test["hesab"].values
    btts_real = ((test["FTHG"] > 0) & (test["FTAG"] > 0)).values
    o25_real = ((test["FTHG"] + test["FTAG"]) > 2.5).values

    # Model 1: əmsala görə (cədvəl YALNIZ train datasından)
    m1, m1_btts, m1_o25, m1_hesab = [], [], [], []
    for h, d, a, ph, pd_, pa in zip(test["H"], test["D"], test["A"], test["pH"], test["pD"], test["pA"]):
        ox, _ = oxsar_tap(train, h, d, a, 0.04, 40)
        r = ox["FTR"].value_counts(normalize=True)
        p = [r.get("H", 0), r.get("D", 0), r.get("A", 0)]
        m1.append(p)
        m1_btts.append(((ox["FTHG"] > 0) & (ox["FTAG"] > 0)).mean())
        m1_o25.append(((ox["FTHG"] + ox["FTAG"]) > 2.5).mean())
        M = poisson_matris(ox["FTHG"].mean(), ox["FTAG"].mean())
        m1_hesab.append(netice_daxilinde_hesablar(M, "HDA"[int(np.argmax(p))], 1)[0][0])
    m1 = np.clip(np.array(m1), 0.01, 1)
    m1 = m1 / m1.sum(axis=1, keepdims=True)
    m1_btts, m1_o25, m1_hesab = np.array(m1_btts), np.array(m1_o25), np.array(m1_hesab)

    # Model 2: komandalara görə (hər oyun üçün yalnız ONDAN ƏVVƏLKİ oyunlar)
    m2, m2_btts, m2_o25, m2_hesab, idx = [], [], [], [], []
    for n, (ev, q, t) in enumerate(zip(test["homeTeam"], test["awayTeam"], test["tarix"])):
        k = komanda_lambda(df, ev, q, t, il, "FTHG", "FTAG")
        if k:
            M = poisson_matris(k[0], k[1])
            p = matris_hda(M)
            qq = matris_qollar(M)
            m2.append([p["H"], p["D"], p["A"]])
            m2_btts.append(qq["btts"])
            m2_o25.append(qq["o25"])
            m2_hesab.append(netice_daxilinde_hesablar(M, max(p, key=p.get), 1)[0][0])
            idx.append(n)
    m2, m2_btts, m2_o25, m2_hesab = np.array(m2), np.array(m2_btts), np.array(m2_o25), np.array(m2_hesab)
    qar = W_EMSAL * m1[idx] + (1 - W_EMSAL) * m2

    def olc(ad, p, y):
        return {"Model": ad, "Oyun": len(y), "Accuracy %": accuracy_score(y, p.argmax(1)) * 100,
                "Log loss": log_loss(y, p, labels=[0, 1, 2])}

    y2 = y_hamisi[idx]
    netice_cedvel = pd.DataFrame([
        olc("Həmişə ev sahibi", np.tile([1, 0, 0], (len(test), 1)) * 0.98 + 0.01, y_hamisi),
        olc("Bukmeker", bukm, y_hamisi),
        olc("💰 Əmsala görə", m1, y_hamisi),
        olc("Bukmeker (komanda modelinin oyunları)", bukm[idx], y2),
        olc("👥 Komandalara görə", m2, y2),
        olc("📌 Qarışıq: 80% əmsal + 20% komanda", qar, y2),
    ]).round(3)

    # Qol proqnozları: Bəli/Xeyr və Üst/Alt neçə faiz düz çıxdı
    def duz(p, real):
        return round(((np.asarray(p) >= 0.5) == real).mean() * 100, 1)

    b_btts = (1 / test["BTTSY"]) / (1 / test["BTTSY"] + 1 / test["BTTSN"])
    b_o25 = (1 / test["O25"]) / (1 / test["O25"] + 1 / test["U25"])
    ok_b, ok_o = b_btts.notna().values, b_o25.notna().values
    qol_cedvel = pd.DataFrame({
        "Model": ["Həmişə 'Bəli' / 'Üst'", "Bukmekerin qol əmsalları", "💰 Əmsala görə", "👥 Komandalara görə"],
        "Hər iki komanda vurur, düz %": [round(btts_real.mean() * 100, 1), duz(b_btts[ok_b], btts_real[ok_b]),
                                          duz(m1_btts, btts_real), duz(m2_btts, btts_real[idx])],
        "2.5 Üst/Alt, düz %": [round(o25_real.mean() * 100, 1), duz(b_o25[ok_o], o25_real[ok_o]),
                               duz(m1_o25, o25_real), duz(m2_o25, o25_real[idx])],
    })

    en_cox = pd.Series(hesab_real).value_counts()
    hesab_cedvel = pd.DataFrame({
        "Model": [f"Həmişə '{en_cox.index[0]}' (ən çox gələn hesab)", "💰 Əmsala görə", "👥 Komandalara görə"],
        "Dəqiq hesab, düz %": [round(en_cox.iloc[0] / len(hesab_real) * 100, 1),
                               round((m1_hesab == hesab_real).mean() * 100, 1),
                               round((m2_hesab == hesab_real[idx]).mean() * 100, 1)],
    })
    return netice_cedvel, qol_cedvel, hesab_cedvel, len(train), len(test)


with tab5:
    st.subheader("📏 Modellər nə qədər dəqiqdir?")
    st.markdown("Modellər **2025-ci ilin iyulundan əvvəlki** oyunlarla qurulur və **sonrakı** oyunlarda yoxlanılır. "
                "Yəni model yoxlanılan oyunların nəticəsini **görmür**, eynən real həyatda olduğu kimi.")
    if st.button("▶️ Hesabla (1-2 dəqiqə çəkə bilər)"):
        with st.spinner("Hər test oyunu üçün proqnoz hesablanır..."):
            netice_cedvel, qol_cedvel, hesab_cedvel, n_tr, n_te = deqiqlik_hesabla(df, il)
        st.caption(f"Train: {n_tr:,} oyun · Test: {n_te:,} oyun")

        st.markdown("#### 🏁 Nəticə (1 / X / 2)")
        st.dataframe(netice_cedvel, hide_index=True, width="stretch")
        st.bar_chart(netice_cedvel.set_index("Model")["Accuracy %"], y_label="Accuracy %")

        st.markdown("#### ⚽ Hər iki komanda vurur? · 📈 2.5 Üst/Alt")
        st.dataframe(qol_cedvel, hide_index=True, width="stretch")

        st.markdown("#### 🎯 Dəqiq hesab")
        st.dataframe(hesab_cedvel, hide_index=True, width="stretch")

        st.markdown(
            "- **Accuracy:** ən yüksək ehtimallı nəticə neçə faiz oyunda düz çıxdı.\n"
            "- **Log loss:** verilən ehtimallar nə qədər dəqiqdir. **Kiçik olması yaxşıdır.**\n"
            "- Komanda modeli yalnız hər iki komandanın kifayət qədər oyunu olan matçlarda işləyir. "
            "Ona görə bukmekerlə müqayisə də **həmin oyunlarda** aparılır.\n"
            "- ⚠️ 3 nəticəli futbol oyununda **50% accuracy** artıq yaxşı nəticədir: bukmeker də bu səviyyədədir.\n"
            "- ⚠️ 'Hər iki komanda vurur' və '2.5 qol' kimi iki variantlı suallarda təsadüfi təxmin **50%** verir. "
            "Ona görə 55-57% kiçik, amma real üstünlükdür.\n"
            "- ⚠️ Dəqiq hesab çox çətindir: ən yaxşı model belə təxminən **8 oyundan 1-ində** düz tapır.")
