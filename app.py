# 필수 라이브러리 설치 명령어:
# pip install streamlit pandas matplotlib plotly numpy openai scikit-learn

import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from openai import OpenAI

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_FILE = 'basketball_stats.csv'

def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        df = pd.DataFrame(columns=[
            'Date', 'Position', 'Points', 'Rebounds', 'Assists', '3PT_Attempts', '3PT_Made',
            'FG_Attempts', 'FG_Made', 'FT_Attempts', 'FT_Made', 'Practice_3PT_Vol', 'Training_Note'
        ])
    
    # 기존 데이터에 없는 신규 컬럼 자동 추가 (호환성 유지)
    expected_cols = [
        'Date', 'Position', 'Points', 'Rebounds', 'Assists', '3PT_Attempts', '3PT_Made',
        'FG_Attempts', 'FG_Made', 'FT_Attempts', 'FT_Made', 'Practice_3PT_Vol', 'Training_Note'
    ]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = '' if col == 'Training_Note' else 0
            
    return df

st.set_page_config(page_title="농구 스탯 & AI 코칭 대시보드", page_icon="🏀", layout="wide")
st.title("🏀 농구 스탯 & AI 맞춤형 코칭 대시보드")

df = load_data()

# -------------------------
# 1. 사이드바: 데이터 입력 & AI 키 설정
# -------------------------
with st.sidebar.form(key='record_form', clear_on_submit=True):
    st.header("새로운 경기 기록 입력")
    date = st.date_input("경기 날짜")
    position = st.selectbox("플레이 포지션", ["Center", "Power Forward", "Small Forward", "Shooting Guard", "Point Guard"], index=3)
    
    st.subheader("기본 스탯")
    points = st.number_input("득점 (Points)", min_value=0, step=1)
    rebounds = st.number_input("리바운드 (Rebounds)", min_value=0, step=1)
    assists = st.number_input("어시스트 (Assists)", min_value=0, step=1)
    
    st.subheader("슈팅 지표 (고급 지표 계산용)")
    fg_attempts = st.number_input("총 야투 시도 (FGA)", min_value=0, step=1)
    fg_made = st.number_input("총 야투 성공 (FGM)", min_value=0, step=1)
    pt3_attempts = st.number_input("3점슛 시도 (3PA)", min_value=0, step=1)
    pt3_made = st.number_input("3점슛 성공 (3PM)", min_value=0, step=1)
    ft_attempts = st.number_input("자유투 시도 (FTA)", min_value=0, step=1)
    ft_made = st.number_input("자유투 성공 (FTM)", min_value=0, step=1)
    
    st.subheader("훈련 및 코칭")
    practice_3pt_vol = st.number_input("주간 3점슛 연습량 (개)", min_value=0, step=10)
    training_note = st.text_area("오늘의 훈련/경기 노트 (특이사항)")
    
    submit_button = st.form_submit_button(label='기록 저장하기')

st.sidebar.divider()
st.sidebar.header("🤖 AI 코치 설정")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", placeholder="sk-...")

if submit_button:
    new_data = pd.DataFrame({
        'Date': [str(date)], 'Position': [position], 'Points': [points], 
        'Rebounds': [rebounds], 'Assists': [assists], 
        '3PT_Attempts': [pt3_attempts], '3PT_Made': [pt3_made],
        'FG_Attempts': [fg_attempts], 'FG_Made': [fg_made],
        'FT_Attempts': [ft_attempts], 'FT_Made': [ft_made],
        'Practice_3PT_Vol': [practice_3pt_vol], 'Training_Note': [training_note]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.sidebar.success("기록이 성공적으로 저장되었습니다!")
    st.rerun()

# -------------------------
# 2. 고급 지표 (Advanced Metrics) 계산
# -------------------------
if not df.empty:
    # 0으로 나누기 방지 및 지표 계산
    df['3PT_Percentage'] = np.where(df['3PT_Attempts'] > 0, (df['3PT_Made'] / df['3PT_Attempts']) * 100, 0)
    df['eFG%'] = np.where(df['FG_Attempts'] > 0, (df['FG_Made'] + 0.5 * df['3PT_Made']) / df['FG_Attempts'] * 100, 0)
    df['TS%'] = np.where((df['FG_Attempts'] + 0.44 * df['FT_Attempts']) > 0, 
                         df['Points'] / (2 * (df['FG_Attempts'] + 0.44 * df['FT_Attempts'])) * 100, 0)
    
    recent_efg = df['eFG%'].tail(3).mean()
    recent_ts = df['TS%'].tail(3).mean()
    recent_3pt = df['3PT_Percentage'].tail(3).mean()

    # 상단 요약 메트릭
    m1, m2, m3 = st.columns(3)
    m1.metric("최근 3경기 평균 eFG% (유효 야투율)", f"{recent_efg:.1f}%")
    m2.metric("최근 3경기 평균 TS% (트루 슈팅율)", f"{recent_ts:.1f}%")
    m3.metric("최근 3경기 평균 3점슛 성공률", f"{recent_3pt:.1f}%")

    tab1, tab2, tab3 = st.tabs(["📊 기본 스탯 & 트렌드", "📈 ML 추세선 예측", "🧠 AI 코칭 피드백"])

    # --- TAB 1: 기본 스탯 시각화 (기존 기능 유지) ---
    with tab1:
        st.subheader("📋 최근 경기 기록 (Raw Data)")
        st.dataframe(df.tail(5).reset_index(drop=True), use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Date'], df['Points'], marker='o', label='득점 (Points)', color='royalblue')
        ax.plot(df['Date'], df['Rebounds'], marker='s', label='리바운드 (Rebounds)', color='darkorange')
        ax.plot(df['Date'], df['3PT_Attempts'], marker='^', label='3점슛 시도 (3PT)', color='forestgreen')
        ax.set_title("경기별 주요 스탯 변화")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # --- TAB 2: ML 기반 성장 추세선 예측 ---
    with tab2:
        st.subheader("🎯 주간 3점슛 연습량 vs 실전 성공률 예측")
        valid_df = df[(df['Practice_3PT_Vol'] > 0) & (df['3PT_Attempts'] > 0)].copy()
        
        if len(valid_df) >= 3:
            x = valid_df['Practice_3PT_Vol']
            y = valid_df['3PT_Percentage']
            
            # 선형 회귀 (numpy.polyfit)
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.array([x.min(), x.max()])
            line_y = slope * line_x + intercept
            
            fig_ml = go.Figure()
            fig_ml.add_trace(go.Scatter(x=x, y=y, mode='markers', name='실제 데이터', marker=dict(size=10, color='blue')))
            fig_ml.add_trace(go.Scatter(x=line_x, y=line_y, mode='lines', name='예측 추세선', line=dict(color='red', dash='dash')))
            
            fig_ml.update_layout(title='연습량에 따른 3점슛 성공률 상관관계', xaxis_title='주간 3점슛 연습량 (개)', yaxis_title='실전 3점슛 성공률 (%)')
            st.plotly_chart(fig_ml, use_container_width=True)
            
            st.info(f"💡 **분석 결과:** 연습량이 100개 증가할 때마다 실전 3점슛 성공률이 약 **{slope*100:.1f}%** 상승하는 추세를 보입니다.")
        else:
            st.warning("추세선을 그리기에 데이터가 부족합니다. 연습량과 3점슛 시도가 포함된 데이터를 3개 이상 입력해 주세요.")

    # --- TAB 3: AI 닥터 & 맞춤형 코칭 ---
    with tab3:
        st.subheader("🤖 전속 AI 농구 코치 피드백")
        if st.button("최근 데이터 기반 AI 코칭 받기"):
            recent_data = df.tail(3).to_dict('records')
            
            if not openai_api_key:
                st.warning("OpenAI API Key가 입력되지 않았습니다. 예시 피드백을 출력합니다.")
                st.success("""
                **[예시 AI 코칭 피드백]**
                1. **최근 경기력 및 슈팅 기복 원인 분석:** 최근 3경기에서 eFG%가 45%에서 52%로 변동성이 있습니다. 훈련 노트에 언급된 '하체 밸런스 불안정'이 실제 야투율 저하로 이어진 것으로 보입니다.
                2. **기술적 개선점 (슈터/가드 관점):** 슛 릴리즈 시 딥(Dip) 동작에서 공이 무릎 아래로 내려가는 현상을 줄이고, 코어 근육을 활용해 릴리즈 포인트를 일정하게 가져가야 합니다.
                3. **추천 훈련 루틴:** 
                   - 폼 슈팅 (림 근처 원핸드 릴리즈) 50개
                   - 캣치 앤 슛 (좌우 윙 이동 후 즉시 슛) 100개
                   - 코어 밸런스 보드 위에서 드리블 훈련 15분
                """)
            else:
                try:
                    client = OpenAI(api_key=openai_api_key)
                    
                    prompt = f"""
                    당신은 프로 농구 데이터 분석가이자 슈팅 코치입니다. 다음 선수의 최근 3경기 스탯과 훈련 노트를 바탕으로 피드백을 작성하세요.
                    
                    [최근 3경기 데이터]
                    {recent_data}
                    
                    다음 3가지 항목을 반드시 포함하여 작성하세요:
                    1. 최근 경기력 및 슈팅 기복 원인 분석 (eFG%, TS% 등 고급 지표 활용)
                    2. 외곽 슈터/가드 포지션 관점에서의 기술적 개선점
                    3. 다음 주차 추천 맞춤 훈련 루틴 3가지
                    """
                    
                    with st.spinner("AI 코치가 데이터를 분석하고 있습니다..."):
                        response = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "system", "content": "You are a professional basketball coach and data analyst."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.7
                        )
                        st.write(response.choices[0].message.content)
                except Exception as e:
                    st.error(f"AI 코칭 생성 중 오류가 발생했습니다: {e}")

else:
    st.info("👈 좌측 패널에서 첫 경기 기록을 입력해 주세요!")