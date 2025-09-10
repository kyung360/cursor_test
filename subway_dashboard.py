import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="지하철 혼잡도 대시보드",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 캐시된 데이터 로딩 함수
@st.cache_data
def load_data():
    """CSV 파일을 로드하고 전처리합니다."""
    try:
        # CSV 파일 로드 (한글 인코딩 고려)
        df = pd.read_csv('서울교통공사_지하철혼잡도정보_20250630.csv', encoding='cp949')
        
        # 시간 컬럼들 추출 (5시30분부터 00시30분까지)
        time_columns = [col for col in df.columns if '시' in col and '분' in col]
        
        # 숫자 데이터로 변환 (공백 제거)
        for col in time_columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
        
        # 결측값을 0으로 채움
        df[time_columns] = df[time_columns].fillna(0)
        
        return df, time_columns
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return None, None

def create_line_congestion_chart(df, time_columns, selected_line, selected_day):
    """노선별 평균 혼잡도 차트를 생성합니다."""
    # 선택된 노선과 요일로 필터링
    filtered_df = df[(df['호선'] == selected_line) & (df['요일구분'] == selected_day)]
    
    if filtered_df.empty:
        st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
        return None
    
    # 시간대별 평균 혼잡도 계산
    avg_congestion = filtered_df[time_columns].mean()
    
    # 시간 라벨 생성 (예: "05:30", "06:00" 등)
    time_labels = [col.replace('시', ':').replace('분', '') for col in time_columns]
    time_labels = [f"{label.split(':')[0].zfill(2)}:{label.split(':')[1].zfill(2)}" for label in time_labels]
    
    # 차트 생성
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_labels,
        y=avg_congestion.values,
        mode='lines+markers',
        name=f'{selected_line} 평균 혼잡도',
        line=dict(width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f'{selected_line} {selected_day} 시간대별 평균 혼잡도',
        xaxis_title='시간',
        yaxis_title='혼잡도 (%)',
        hovermode='x unified',
        height=500
    )
    
    # x축 라벨 회전
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_station_comparison_chart(df, time_columns, selected_stations, selected_day):
    """선택된 역들의 혼잡도 비교 차트를 생성합니다."""
    if not selected_stations:
        return None
    
    fig = go.Figure()
    
    # 시간 라벨 생성
    time_labels = [col.replace('시', ':').replace('분', '') for col in time_columns]
    time_labels = [f"{label.split(':')[0].zfill(2)}:{label.split(':')[1].zfill(2)}" for label in time_labels]
    
    colors = px.colors.qualitative.Set1
    
    for i, station in enumerate(selected_stations):
        # 해당 역의 데이터 (상선+하선 평균)
        station_data = df[(df['출발역'] == station) & (df['요일구분'] == selected_day)]
        
        if not station_data.empty:
            avg_congestion = station_data[time_columns].mean()
            
            fig.add_trace(go.Scatter(
                x=time_labels,
                y=avg_congestion.values,
                mode='lines+markers',
                name=station,
                line=dict(width=3, color=colors[i % len(colors)]),
                marker=dict(size=6)
            ))
    
    fig.update_layout(
        title=f'역별 혼잡도 비교 ({selected_day})',
        xaxis_title='시간',
        yaxis_title='혼잡도 (%)',
        hovermode='x unified',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_heatmap(df, time_columns, selected_line, selected_day):
    """역별 시간대별 혼잡도 히트맵을 생성합니다."""
    # 선택된 노선과 요일로 필터링
    filtered_df = df[(df['호선'] == selected_line) & (df['요일구분'] == selected_day)]
    
    if filtered_df.empty:
        return None
    
    # 역별로 상선/하선 평균 계산
    station_congestion = []
    stations = []
    
    for station in filtered_df['출발역'].unique():
        station_data = filtered_df[filtered_df['출발역'] == station][time_columns].mean()
        station_congestion.append(station_data.values)
        stations.append(station)
    
    # 시간 라벨 생성
    time_labels = [col.replace('시', ':').replace('분', '') for col in time_columns]
    time_labels = [f"{label.split(':')[0].zfill(2)}:{label.split(':')[1].zfill(2)}" for label in time_labels]
    
    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=station_congestion,
        x=time_labels,
        y=stations,
        colorscale='Reds',
        hoverongaps=False,
        colorbar=dict(title="혼잡도 (%)"),
        hovertemplate='<b>%{y}</b><br>시간: %{x}<br>혼잡도: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{selected_line} {selected_day} 역별 시간대별 혼잡도',
        xaxis_title='시간',
        yaxis_title='역명',
        height=max(400, len(stations) * 30)
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def main():
    # 제목
    st.title("🚇 지하철 혼잡도 대시보드")
    st.markdown("서울교통공사 지하철 혼잡도 데이터를 기반으로 한 분석 대시보드입니다.")
    
    # 데이터 로드
    df, time_columns = load_data()
    
    if df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return
    
    # 사이드바 필터
    st.sidebar.header("📊 필터 옵션")
    
    # 요일 선택
    day_options = df['요일구분'].unique()
    selected_day = st.sidebar.selectbox("요일 선택", day_options)
    
    # 노선 선택
    line_options = sorted(df['호선'].unique())
    selected_line = st.sidebar.selectbox("노선 선택", line_options)
    
    # 역 선택 (다중 선택)
    station_options = sorted(df[df['호선'] == selected_line]['출발역'].unique())
    selected_stations = st.sidebar.multiselect(
        "비교할 역 선택 (최대 5개)", 
        station_options, 
        default=station_options[:3] if len(station_options) >= 3 else station_options,
        max_selections=5
    )
    
    # 시간대 필터
    st.sidebar.subheader("시간대 필터")
    time_labels = [col.replace('시', ':').replace('분', '') for col in time_columns]
    time_labels_formatted = [f"{label.split(':')[0].zfill(2)}:{label.split(':')[1].zfill(2)}" for label in time_labels]
    
    time_range = st.sidebar.select_slider(
        "시간 범위 선택",
        options=range(len(time_labels_formatted)),
        value=(0, len(time_labels_formatted)-1),
        format_func=lambda x: time_labels_formatted[x]
    )
    
    # 선택된 시간 범위의 컬럼들
    selected_time_columns = time_columns[time_range[0]:time_range[1]+1]
    
    # 정렬 옵션
    st.sidebar.subheader("정렬 옵션")
    sort_options = ["역명순", "평균 혼잡도 높은순", "평균 혼잡도 낮은순", "최대 혼잡도 높은순"]
    selected_sort = st.sidebar.selectbox("정렬 기준", sort_options)
    
    # 메인 대시보드
    # 첫 번째 행: 개요 메트릭
    col1, col2, col3, col4 = st.columns(4)
    
    filtered_data = df[(df['호선'] == selected_line) & (df['요일구분'] == selected_day)]
    
    with col1:
        total_stations = len(filtered_data['출발역'].unique())
        st.metric("총 역 수", f"{total_stations}개")
    
    with col2:
        avg_congestion = filtered_data[selected_time_columns].mean().mean()
        st.metric("평균 혼잡도", f"{avg_congestion:.1f}%")
    
    with col3:
        max_congestion = filtered_data[selected_time_columns].max().max()
        st.metric("최대 혼잡도", f"{max_congestion:.1f}%")
    
    with col4:
        peak_time_idx = filtered_data[selected_time_columns].mean().idxmax()
        peak_time = peak_time_idx.replace('시', ':').replace('분', '')
        peak_time_formatted = f"{peak_time.split(':')[0].zfill(2)}:{peak_time.split(':')[1].zfill(2)}"
        st.metric("혼잡 피크 시간", peak_time_formatted)
    
    # 두 번째 행: 차트
    st.header("📈 시간대별 혼잡도 분석")
    
    # 탭으로 차트 구분
    tab1, tab2, tab3 = st.tabs(["노선 평균", "역별 비교", "히트맵"])
    
    with tab1:
        # 노선별 평균 혼잡도 차트
        line_chart = create_line_congestion_chart(df, selected_time_columns, selected_line, selected_day)
        if line_chart:
            st.plotly_chart(line_chart, use_container_width=True)
    
    with tab2:
        # 역별 비교 차트
        if selected_stations:
            comparison_chart = create_station_comparison_chart(df, selected_time_columns, selected_stations, selected_day)
            if comparison_chart:
                st.plotly_chart(comparison_chart, use_container_width=True)
        else:
            st.info("비교할 역을 선택해주세요.")
    
    with tab3:
        # 히트맵
        heatmap = create_heatmap(df, selected_time_columns, selected_line, selected_day)
        if heatmap:
            st.plotly_chart(heatmap, use_container_width=True)
    
    # 세 번째 행: 역별 상세 정보 테이블
    st.header("📋 역별 혼잡도 상세 정보")
    
    # 정렬 기준에 따라 데이터 정렬
    station_summary = []
    
    for station in filtered_data['출발역'].unique():
        station_data = filtered_data[filtered_data['출발역'] == station]
        avg_cong = station_data[selected_time_columns].mean().mean()
        max_cong = station_data[selected_time_columns].max().max()
        
        # 상선/하선 구분 정보
        upline_avg = station_data[station_data['상하구분'] == '상선'][selected_time_columns].mean().mean() if not station_data[station_data['상하구분'] == '상선'].empty else 0
        downline_avg = station_data[station_data['상하구분'] == '하선'][selected_time_columns].mean().mean() if not station_data[station_data['상하구분'] == '하선'].empty else 0
        
        station_summary.append({
            '역명': station,
            '평균 혼잡도': avg_cong,
            '최대 혼잡도': max_cong,
            '상선 평균': upline_avg,
            '하선 평균': downline_avg
        })
    
    summary_df = pd.DataFrame(station_summary)
    
    # 정렬 적용
    if selected_sort == "평균 혼잡도 높은순":
        summary_df = summary_df.sort_values('평균 혼잡도', ascending=False)
    elif selected_sort == "평균 혼잡도 낮은순":
        summary_df = summary_df.sort_values('평균 혼잡도', ascending=True)
    elif selected_sort == "최대 혼잡도 높은순":
        summary_df = summary_df.sort_values('최대 혼잡도', ascending=False)
    else:  # 역명순
        summary_df = summary_df.sort_values('역명')
    
    # 숫자 포맷팅
    summary_df_display = summary_df.copy()
    for col in ['평균 혼잡도', '최대 혼잡도', '상선 평균', '하선 평균']:
        summary_df_display[col] = summary_df_display[col].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(summary_df_display, use_container_width=True)
    
    # 데이터 다운로드 버튼
    csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 데이터 다운로드 (CSV)",
        data=csv,
        file_name=f"{selected_line}_{selected_day}_혼잡도_분석.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
