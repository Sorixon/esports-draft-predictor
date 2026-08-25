import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import json
import requests

st.set_page_config(page_title="LoL AI Draft Assistant", layout="wide", page_icon="⚔️")

# =====================================================================
# 1. Architektura Sieci i Ładowanie Modelu
# =====================================================================
class DraftPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DraftPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

@st.cache_resource
def load_model_and_cols():
    with open('model_columns.json', 'r') as f:
        columns = json.load(f)
    model = DraftPredictor(len(columns))
    model.load_state_dict(torch.load('lol_draft_model.pth', weights_only=True))
    model.eval()
    return model, columns

model, columns = load_model_and_cols()
all_champions = sorted([c.replace('champion_', '') for c in columns if c.startswith('champion_')])

# =====================================================================
# 2. Integracja z Data Dragon API (Wersja Statyczna - Niezawodna)
# =====================================================================
PATCH_VERSION = "14.18.1"

def get_icon_url(champion_name):
    exceptions = {
        "Wukong": "MonkeyKing", "FiddleSticks": "Fiddlesticks", "LeBlanc": "Leblanc",
        "K'Sante": "KSante", "Kai'Sa": "Kaisa", "Kha'Zix": "Khazix", "Cho'Gath": "Chogath",
        "Vel'Koz": "Velkoz", "Rek'Sai": "RekSai", "Bel'Veth": "Belveth", "Nunu & Willump": "Nunu",
        "Renata Glasc": "Renata", "Dr. Mundo": "DrMundo", "Master Yi": "MasterYi",
        "Twisted Fate": "TwistedFate", "Xin Zhao": "XinZhao", "Aurelion Sol": "AurelionSol",
        "Tahm Kench": "TahmKench", "Miss Fortune": "MissFortune", "Jarvan IV": "JarvanIV",
        "Lee Sin": "LeeSin"
    }
    
    clean_name = exceptions.get(champion_name, champion_name.replace("'", "").replace(" ", "").replace(".", ""))
    
    # Oficjalne CDN Riotu z poprawnym patchem w chmurze zadziała bez problemu
    return f"https://ddragon.leagueoflegends.com/cdn/{PATCH_VERSION}/img/champion/{clean_name}.png"

def render_team_gallery(champions_list, title):
    if not champions_list:
        return
    st.markdown(f"**{title}**")
    cols = st.columns(len(champions_list))
    roles_icons = ["🛡️", "⚔️", "🔮", "🎯", "👁️"]
    for idx, champ in enumerate(champions_list):
        with cols[idx]:
            role_icon = roles_icons[idx] if idx < len(roles_icons) else "🎮"
            st.markdown(
                f"""
                <div style="background-color: #1e1e1e; padding: 10px; border-radius: 8px; text-align: center; border: 1px solid #333;">
                    <span style="font-size: 24px;">{role_icon}</span>
                    <p style="font-size: 14px; margin: 6px 0 0 0; font-weight: bold; color: #fff;">{champ}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# =====================================================================
# 3. Interfejs Użytkownika (UI)
# =====================================================================
st.title("🏆 LoL Draft Assistant: AI Predictor")
st.caption(f"Połączono z Riot Data Dragon (Patch {PATCH_VERSION}) | Model: PyTorch Deep Neural Network")

col_top1, col_top2 = st.columns([2, 4])
with col_top1:
    league_choice = st.selectbox("Wybierz ligę:", ["LEC", "LCK", "LPL", "LCS"])

st.divider()

col_blue, col_red = st.columns(2)

with col_blue:
    st.subheader("🔵 BLUE SIDE")
    blue_picks = st.multiselect("Picki Blue (maks. 5):", all_champions, max_selections=5, default=['Jax', 'Sejuani'])
    blue_bans = st.multiselect("Bany Blue (maks. 5):", all_champions, max_selections=5, default=['Aatrox', 'K\'Sante'])
    render_team_gallery(blue_picks, "Wybrana kompozycja Blue:")

with col_red:
    st.subheader("🔴 RED SIDE")
    red_picks = st.multiselect("Picki Red (maks. 5):", all_champions, max_selections=5, default=['Renekton', 'Vi'])
    red_bans = st.multiselect("Bany Red (maks. 5):", all_champions, max_selections=5, default=['Kalista'])
    render_team_gallery(red_picks, "Wybrana kompozycja Red:")

st.divider()

# =====================================================================
# 4. Obliczenia i Rekomendacja z Ikonami
# =====================================================================
if st.button("🚀 Przelicz Szanse i Pobierz Rekomendacje", type="primary", use_container_width=True):
    # Przygotowanie wektora wejściowego
    input_row = pd.DataFrame(0, index=[0], columns=columns)
    if 'side_Blue' in input_row.columns: input_row['side_Blue'] = 1
    if f'league_{league_choice}' in input_row.columns: input_row[f'league_{league_choice}'] = 1
    
    for c in blue_picks:
        if f'champion_{c}' in input_row.columns: input_row[f'champion_{c}'] = 1
    for i, b in enumerate(blue_bans[:5], start=1):
        if f'ban{i}_{b}' in input_row.columns: input_row[f'ban{i}_{b}'] = 1

    # Predykcja bazowa
    input_tensor = torch.tensor(input_row.values, dtype=torch.float32)
    with torch.no_grad():
        blue_prob = model(input_tensor).item() * 100
    red_prob = 100 - blue_prob

    # Prezentacja wyniku
    st.subheader("📊 Wynik Symulacji")
    m1, m2 = st.columns(2)
    m1.metric("🔵 Szansa Blue Side", f"{blue_prob:.2f}%")
    m2.metric("🔴 Szansa Red Side", f"{red_prob:.2f}%")
    st.progress(blue_prob / 100)

    # Rekomendacja kolejnego picka
    if len(blue_picks) < 5:
        st.subheader("💡 TOP 5 Rekomendowanych Postaci dla Blue Side")
        unavailable = set(blue_picks + red_picks + blue_bans + red_bans)
        available = [c for c in all_champions if c not in unavailable]
        
        recs = []
        for cand in available:
            sim_row = input_row.copy()
            if f'champion_{cand}' in sim_row.columns:
                sim_row[f'champion_{cand}'] = 1
            t = torch.tensor(sim_row.values, dtype=torch.float32)
            with torch.no_grad():
                winrate = model(t).item() * 100
                recs.append({'Bohater': cand, 'Winrate': winrate})
        
        df_top = pd.DataFrame(recs).sort_values(by='Winrate', ascending=False).head(5).reset_index(drop=True)
        
       rec_cols = st.columns(5)
        for i, row in df_top.iterrows():
            with rec_cols[i]:
                st.markdown(
                    f"""
                    <div style="background-color: #1e1e1e; padding: 12px; border-radius: 8px; text-align: center; border: 1px solid #444;">
                        <span style="color: #00ffcc; font-weight: bold; font-size: 12px;">#{i+1} REKOMENDACJA</span>
                        <p style="font-size: 15px; margin: 6px 0; font-weight: bold; color: #fff;">{row['Bohater']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                diff = row['Winrate'] - blue_prob
                diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
                st.metric(label="Winrate", value=f"{row['Winrate']:.2f}%", delta=diff_str)
