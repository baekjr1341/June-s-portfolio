# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

# 한글 폰트 설정 (깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

DATA_FILE = 'basketball_stats.csv'

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame(columns=[
            'Date', 'Position', 'Points', 'Rebounds', 'Assists', '3PT_Attempts', '3PT_Made'
        ])

st.set_page_config(page_title="농구 스탯 & 성장 일지", page_icon="🏀", layout="wide")

st.title("🏀 나만의 농구 스탯 & 성장 일지")
st.markdown("센터에서 슈팅 가드로! 포지션 변경에 따른 내 플레이 스타일 변화를 추적합니다.")

df = load_data()

# 1. 사이드바: 기록 입력 폼 (st.form을 사용하여 갱신 오류 방지)
with st.sidebar.form(key='record_form', clear_on_submit=True):
    st.header("새로운 경기 기록 입력")
    date = st.date_input("경기 날짜")
    position = st.selectbox("플레이 포지션", ["Center", "Power Forward", "Small Forward", "Shooting Guard", "Point Guard"], index=3)
    points = st.number_input("득점", min_value=0, step=1)
    rebounds = st.number_input("리바운드", min_value=0, step=1)
    assists = st.number_input("어시스트", min_value=0, step=1)
    pt3_attempts = st.number_input("3점슛 시도", min_value=0, step=1)
    pt3_made = st.number_input("3점슛 성공", min_value=0, step=1)
    
    submit_button = st.form_submit_button(label='기록 저장하기')

if submit_button:
    new_data = pd.DataFrame({
        'Date': [str(date)], 'Position': [position], 'Points': [points], 
        'Rebounds': [rebounds], 'Assists': [assists], 
        '3PT_Attempts': [pt3_attempts], '3PT_Made': [pt3_made]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(DATA_FILE, index=False)
    st.sidebar.success("기록이 성공적으로 저장되었습니다!")
    st.rerun()

# 2. 메인 화면 시각화 영역
if not df.empty:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📋 최근 경기 기록 (Raw Data)")
        st.dataframe(df.tail(5).reset_index(drop=True), use_container_width=True)
        
    with col2:
        st.subheader("📈 나의 누적 성장 트렌드")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['Date'], df['Points'], marker='o', label='득점 (Points)', color='royalblue')
        ax.plot(df['Date'], df['Rebounds'], marker='s', label='리바운드 (Rebounds)', color='darkorange')
        ax.plot(df['Date'], df['3PT_Attempts'], marker='^', label='3점슛 시도 (3PT)', color='forestgreen')
        ax.set_title("경기별 주요 스탯 변화")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

    st.divider()
    
    st.subheader("🔄 포지션별 플레이 스타일 비교 (Center vs SG)")
    
    # 파생 변수: 3점슛 성공률
    df['3PT_Percentage'] = (df['3PT_Made'] / df['3PT_Attempts'] * 100).fillna(0)
    
    # 포지션별 평균 데이터 그룹화
    position_stats = df.groupby('Position')[['Points', 'Rebounds', '3PT_Attempts', 'Assists']].mean().round(1)
    
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**포지션별 평균 스탯**")
        st.dataframe(position_stats, use_container_width=True)
    with col4:
        st.markdown("**주요 스탯 비교 차트**")
        if not position_stats.empty:
            st.bar_chart(position_stats[['Rebounds', '3PT_Attempts']])
            st.caption("비교 포인트: 센터 시절의 리바운드 수치와 슈팅 가드 전향 후의 3점슛 시도 횟수 변화")
else:
    st.info("👈 아직 입력된 데이터가 없습니다. 좌측 패널에서 첫 경기 기록을 입력해 주세요!")

