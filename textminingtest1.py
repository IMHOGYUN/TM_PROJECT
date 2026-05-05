import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE

# ==========================================
# [설정 1] 한글 폰트 깨짐 방지 설정 (OS에 맞게 선택)
# ==========================================
# Windows 사용자: 'Malgun Gothic' / Mac 사용자: 'AppleGothic'
import matplotlib as mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Malgun Gothic']
mpl.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# ==========================================
# [설정 2] 전처리 및 분석 설정
# ==========================================
DATA_DIR = './dataset' # 텍스트 파일이 들어있는 최상위 폴더 경로
CHUNK_SIZE = 500       # 어절 단위 분할 기준

# ★핵심: 기본 불용어 리스트 (수동 추가 필수)
# 주의: 아래 불용어는 필수적으로 포함할 것들만 입력하세요.
# 나머지 불용어는 자동 감지 기능이 추출합니다 (IDF 기반, TF 기반)
custom_stopwords = {
    # 주인공 이름 (반드시 수동으로 추가 필요)
    '문호', '란수', '광호', '준원', '금봉', '광진', '은봉', '명규', '인현',
    '점순', '점순이', '안해',
    # 주요 지명 (선택적)
    '경성', '동경', '상해'
}

# ==========================================
# 1단계: 텍스트 로드 및 전처리 함수
# ==========================================
def clean_text(text):
    """한자, 특수기호 제거 및 기본 정제 (숫자 제거 추가)"""
    text = re.sub(r'[^가-힣a-zA-Z\s.,!?\'\"]', '', text)  # 숫자 제거
    return re.sub(r'\s+', ' ', text).strip()

def chunk_text(text, chunk_size):
    """지정된 어절 단위로 텍스트 분할"""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_style_tokens(text):
    """한글 단어 기반 스타일 토큰 추출 (더 정교하게: 길이 필터링 강화)"""
    tokens = re.findall(r'[가-힣]{2,}', text)
    style_tokens = [t for t in tokens if t not in custom_stopwords and len(t) >= 2 and len(t) <= 10]  # 길이 제한 추가
    return " ".join(style_tokens)


def extract_style_features(text):
    sentences = re.split(r'[.!?]\s+|[。！？]\s*', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    word_counts = [len(s.split()) for s in sentences]
    avg_len = sum(word_counts) / len(word_counts) if word_counts else 0

    tokens = re.findall(r'[가-힣]{2,}', text)
    total = len(tokens)

    verb_count = sum(1 for t in tokens if t.endswith('다'))
    unique_rate = len(set(tokens)) / total if total > 0 else 0

    return {
        'avg_sent_len': avg_len,
        'dialogue_ratio': text.count('"') / len(text) if len(text) > 0 else 0,
        'verb_density': verb_count / total if total > 0 else 0,
        'unique_density': unique_rate,
        'short_sent_ratio': sum(1 for w in word_counts if w < 10) / len(word_counts) if word_counts else 0,
    }


def load_text(file_path):
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"인코딩을 감지할 수 없는 파일: {file_path}")


# ==========================================
# 2단계: 데이터셋 구축 (작가별 전체 작품 통합)
# ==========================================
print("데이터 로드 및 전처리 시작...")
author_corpus = {}

for author_name in os.listdir(DATA_DIR):
    author_path = os.path.join(DATA_DIR, author_name)
    if not os.path.isdir(author_path):
        continue

    all_texts = []
    for file_name in os.listdir(author_path):
        if not file_name.endswith('.txt'):
            continue
        file_path = os.path.join(author_path, file_name)
        raw_text = load_text(file_path)
        cleaned_text = clean_text(raw_text)
        all_texts.append(cleaned_text)

    if not all_texts:
        continue
    author_corpus[author_name] = " ".join(all_texts)

# 작가별 전체 문서를 하나로 합쳐서 스타일 기반 벡터화
df = pd.DataFrame([
    {'Author': author, 'Text': text}
    for author, text in author_corpus.items()
])

# 스타일 토큰과 통계 특성 추가
df['Style_Tokens'] = df['Text'].apply(extract_style_tokens)
stat_features = []
for _, row in df.iterrows():
    features = extract_style_features(row['Text'])
    stat_features.append([
        features['avg_sent_len'],
        features['dialogue_ratio'],
        features['verb_density'],
        features['unique_density'],
        features['short_sent_ratio']
    ])
stat_array = np.array(stat_features)

print(f"총 {len(df)}개의 작가별 통합 문서가 생성되었습니다.\n")

# ==========================================
# 3단계: TF-IDF + 통계 특징 결합
# ==========================================
print("TF-IDF 벡터 변환 중...")
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import LatentDirichletAllocation

# ★ 단계 3-1: 자동 불용어 감지 (IDF 기반)
print("  → 자동 불용어 감지 중...")
vectorizer_temp = TfidfVectorizer(min_df=1, max_features=3000, stop_words=list(custom_stopwords))
tfidf_matrix_temp = vectorizer_temp.fit_transform(df['Style_Tokens'])

# IDF 값 추출: IDF가 낮을수록 모든 문서에 자주 나타나는 단어
idf_values = vectorizer_temp.idf_
feature_names_temp = vectorizer_temp.get_feature_names_out()

# IDF 백분위수 기반 자동 불용어 감지 (IDF 상위 30% = 자주 나타나는 단어들)
idf_threshold = np.percentile(idf_values, 30)  # 낮은 IDF = 자주 나타남
low_idf_words = set([feature_names_temp[i] for i in range(len(feature_names_temp)) 
                      if idf_values[i] < idf_threshold])

# TF 값이 높은 단어들도 자동 감지 (의미 기여도가 낮을 수 있음)
tfidf_array_temp = tfidf_matrix_temp.toarray()
mean_tfidf = np.mean(tfidf_array_temp, axis=0)
high_tfidf_threshold = np.percentile(mean_tfidf, 85)
high_tf_words = set([feature_names_temp[i] for i in range(len(feature_names_temp))
                     if mean_tfidf[i] > high_tfidf_threshold])

auto_detected_stopwords = low_idf_words | high_tf_words
print(f"  → 자동 감지된 불용어: {len(auto_detected_stopwords)}개")
custom_stopwords.update(auto_detected_stopwords)

# 최초 TF-IDF로 LDA 토픽 단어 추출
vectorizer = TfidfVectorizer(min_df=1, max_features=2000, stop_words=list(custom_stopwords))
tfidf_matrix = vectorizer.fit_transform(df['Style_Tokens'])
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(tfidf_matrix)

feature_names = vectorizer.get_feature_names_out()
topic_words = set()
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-30:]
    topic_words.update([feature_names[i] for i in top_words_idx])

print(f"  → LDA 추출 주제어: {len(topic_words)}개")
custom_stopwords.update(topic_words)

# 최종 TF-IDF (주제어 제거 후)
vectorizer = TfidfVectorizer(min_df=1, max_features=1500, stop_words=list(custom_stopwords))
tfidf_matrix = vectorizer.fit_transform(df['Style_Tokens'])
tfidf_array = tfidf_matrix.toarray()

scaler_tfidf = MinMaxScaler()
scaler_stat = MinMaxScaler()

tfidf_norm = scaler_tfidf.fit_transform(tfidf_array)
stat_norm = scaler_stat.fit_transform(stat_array)

combined = np.hstack([tfidf_norm * 0.5, stat_norm * 0.5])

# ==========================================
# 4단계: 계층적 클러스터링 및 덴드로그램 (Dendrogram)
# ==========================================
print("계층적 클러스터링 수행 중...")
# Ward 연결법 사용 (군집 내 분산을 최소화하여 문체 분류에 유리)
linkage_matrix = linkage(combined, method='ward')

plt.figure(figsize=(15, 8))
labels = [row['Author'] for _, row in df.iterrows()]

dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, leaf_font_size=10, truncate_mode='lastp', p=10)
plt.title("한국 근대 소설 작가 문체 덴드로그램")
plt.xlabel("작가")
plt.ylabel("문체적 거리 (Distance)")
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
print("덴드로그램이 'dendrogram.png'로 저장되었습니다.")
# plt.show()

# 클러스터 수 계산 (예: 높이 5로 자름)
clusters = fcluster(linkage_matrix, t=5, criterion='distance')
df['cluster'] = clusters
print(f"높이 5로 자른 클러스터 수: {len(set(clusters))}")

# 실루엣 스코어 계산
if len(df) > 1:
    n_clusters = len(set(clusters))
    if 1 < n_clusters < len(df):
        score = silhouette_score(combined, clusters)
        print(f"실루엣 스코어: {score:.4f}")
    else:
        print(f"실루엣 스코어 계산 불가: 클러스터 수 {n_clusters}개 (샘플 수 {len(df)}개)")

# ==========================================
# 5단계: 실루엣 스코어 검증 및 t-SNE 시각화
# ==========================================
# 덴드로그램에서 시각적으로 군집을 자를 높이를 정한 뒤, t-SNE로 2차원 투영
print("t-SNE 2차원 시각화 수행 중...")
perplexity_value = min(30, max(1, len(df) - 1))
print(f"t-SNE perplexity: {perplexity_value}")
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value)
tsne_results = tsne.fit_transform(combined)

df['tsne_x'] = tsne_results[:, 0]
df['tsne_y'] = tsne_results[:, 1]

plt.figure(figsize=(12, 8))
authors = df['Author'].unique()
colors = plt.cm.tab10(np.linspace(0, 1, len(authors)))

for i, author in enumerate(authors):
    subset = df[df['Author'] == author]
    plt.scatter(subset['tsne_x'], subset['tsne_y'], label=author, s=50, alpha=0.7, color=colors[i])

plt.title("작가별 문체 2차원 공간 투영 (t-SNE)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('tsne_authors.png', dpi=300, bbox_inches='tight')
print("t-SNE 시각화가 'tsne_authors.png'로 저장되었습니다.")
# plt.show()