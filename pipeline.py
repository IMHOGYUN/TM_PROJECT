"""
문체 DNA 분석 파이프라인
======================
사용법:
    python pipeline.py

데이터 구조:
    data/
    ├── 이상/
    │   ├── 날개.txt
    │   └── 봉별기.txt
    ├── 김유정/
    │   ├── 봄봄.txt
    │   └── 동백꽃.txt
    └── ...
"""

import os
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter
from pathlib import Path

from kiwipiepy import Kiwi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec

# ── 한글 폰트 설정 ──────────────────────────────────────────
def set_korean_font():
    font_candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for fp in font_candidates:
        if os.path.exists(fp):
            fm.fontManager.addfont(fp)
            fname = fm.FontProperties(fname=fp).get_name()
            plt.rcParams['font.family'] = fname
            break
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# ── 설정 ────────────────────────────────────────────────────
DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CHUNK_SIZE = 500        # 청크 단위 (어절 수)
W2V_DIM    = 100        # Word2Vec 임베딩 차원
W2V_WINDOW = 5
W2V_MIN    = 2

# 불용어 (필요 시 확장)
STOPWORDS = {
    '이', '가', '을', '를', '은', '는', '의', '에', '에서', '로', '으로',
    '와', '과', '도', '만', '하다', '있다', '없다', '되다', '이다',
    '그', '나', '내', '우리', '것', '수', '때', '곳', '말',
}

kiwi = Kiwi()


# ════════════════════════════════════════════════════════════
# 1. 데이터 로딩
# ════════════════════════════════════════════════════════════

def load_texts(data_dir: Path) -> dict[str, str]:
    """작가별 전체 텍스트 합산 반환 {작가명: 전체텍스트}"""
    texts = {}
    if not data_dir.exists():
        print(f"[경고] {data_dir} 폴더가 없습니다. 샘플 데이터로 실행합니다.")
        return _sample_texts()

    for author_dir in sorted(data_dir.iterdir()):
        if not author_dir.is_dir():
            continue
        combined = []
        for txt_file in sorted(author_dir.glob("*.txt")):
            try:
                text = txt_file.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                text = txt_file.read_text(encoding='cp949', errors='ignore')
            combined.append(text.strip())
        if combined:
            texts[author_dir.name] = "\n".join(combined)
            print(f"  로드: {author_dir.name} ({len(combined)}개 파일)")
    return texts


def _sample_texts() -> dict[str, str]:
    """데이터 없을 때 동작 확인용 샘플"""
    return {
        "이상":  "나는 거울 앞에 섰다. 거울 속의 나는 나인가. " * 80,
        "김유정": "봄이 왔다. 산에도 들에도 꽃이 피었다. 강냉이 밭에서 일하며 " * 80,
        "이효석": "메밀꽃이 피었다. 달빛 아래 하얀 꽃밭이 펼쳐졌다. " * 80,
        "이광수": "우리 민족은 반드시 깨어나야 한다. 교육이 중요하다. " * 80,
        "채만식": "세상이 돌아가는 꼴이 참으로 우습다. 비꼬아야 속이 시원하다. " * 80,
        "김동인": "그는 죽었다. 아무도 몰랐다. 어두운 방 안에서 홀로. " * 80,
    }


# ════════════════════════════════════════════════════════════
# 2. 전처리
# ════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """기본 노이즈 제거"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?\"\'()]', ' ', text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Kiwi 형태소 분석 후 명사/동사/형용사/부사 추출"""
    result = kiwi.analyze(text[:50000])  # 너무 길면 자름
    tokens = []
    for token in result[0][0]:           # result[0] = (token_list, score)
        if token.tag.startswith(('N', 'V', 'M', 'X')):
            lemma = token.form
            if len(lemma) > 1 and lemma not in STOPWORDS:
                tokens.append(lemma)
    return tokens


def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """어절 단위 청킹"""
    words = text.split()
    return [' '.join(words[i:i+chunk_size])
            for i in range(0, len(words), chunk_size)
            if len(words[i:i+chunk_size]) >= chunk_size // 2]


def preprocess_all(texts: dict[str, str]) -> dict[str, dict]:
    """작가별 전처리 결과 반환"""
    print("\n[전처리 시작]")
    processed = {}
    for author, text in texts.items():
        print(f"  처리 중: {author} ...", end=' ')
        cleaned   = clean_text(text)
        tokens    = tokenize(cleaned)
        chunks    = split_into_chunks(cleaned)
        processed[author] = {
            'raw':    cleaned,
            'tokens': tokens,
            'chunks': chunks,
        }
        print(f"토큰 {len(tokens)}개, 청크 {len(chunks)}개")
    return processed


# ════════════════════════════════════════════════════════════
# 3. 피처 추출
# ════════════════════════════════════════════════════════════

# ── 3-1. TF-IDF ─────────────────────────────────────────────
def extract_tfidf(processed: dict) -> np.ndarray:
    """작가별 TF-IDF 벡터 (작가 수 × 어휘 크기)"""
    corpus = [' '.join(v['tokens']) for v in processed.values()]
    vectorizer = TfidfVectorizer(max_features=500, min_df=1)
    matrix = vectorizer.fit_transform(corpus).toarray()
    print(f"  TF-IDF shape: {matrix.shape}")
    return matrix


# ── 3-2. Word2Vec 평균 벡터 ──────────────────────────────────
def train_word2vec(processed: dict) -> Word2Vec:
    """전체 코퍼스로 Word2Vec 학습"""
    sentences = []
    for v in processed.values():
        tokens = v['tokens']
        # 문장 단위로 쪼개서 학습 (50토큰씩)
        for i in range(0, len(tokens), 50):
            sentences.append(tokens[i:i+50])
    model = Word2Vec(
        sentences,
        vector_size=W2V_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN,
        workers=4,
        epochs=10,
        seed=42,
    )
    print(f"  Word2Vec 어휘 크기: {len(model.wv)}")
    return model


def extract_w2v_mean(processed: dict, w2v_model: Word2Vec) -> np.ndarray:
    """작가별 단어 임베딩 평균 벡터"""
    vectors = []
    for v in processed.values():
        token_vecs = [
            w2v_model.wv[t]
            for t in v['tokens']
            if t in w2v_model.wv
        ]
        if token_vecs:
            vectors.append(np.mean(token_vecs, axis=0))
        else:
            vectors.append(np.zeros(W2V_DIM))
    matrix = np.array(vectors)
    print(f"  Word2Vec 평균벡터 shape: {matrix.shape}")
    return matrix


# ── 3-3. 통계 피처 7종 ──────────────────────────────────────
def extract_stat_features(processed: dict) -> np.ndarray:
    """
    1. 평균 문장 길이 (어절)
    2. 단문 비율 (10어절 미만)
    3. 어휘 다양성 TTR
    4. 수식어 밀도 (형용사+부사 비율)
    5. 대화문 비율 (따옴표 포함 문장)
    6. 문장 길이 표준편차
    7. 명사 비율
    """
    results = []
    for author, v in processed.items():
        text   = v['raw']
        tokens = v['tokens']

        # 문장 분리
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_lens = [len(s.split()) for s in sentences] if sentences else [0]

        # Kiwi 품사 분석 (샘플링)
        sample_text = text[:10000]
        pos_result  = kiwi.analyze(sample_text)
        pos_tags    = [t.tag for t in pos_result[0][0]]
        total_tags  = len(pos_tags) if pos_tags else 1

        noun_count = sum(1 for t in pos_tags if t.startswith('N'))
        adj_count  = sum(1 for t in pos_tags if t.startswith('VA') or t.startswith('MM'))
        adv_count  = sum(1 for t in pos_tags if t.startswith('MA'))

        feat = [
            np.mean(sent_lens),                                  # 1. 평균 문장 길이
            sum(1 for l in sent_lens if l < 10) / len(sent_lens), # 2. 단문 비율
            len(set(tokens)) / len(tokens) if tokens else 0,    # 3. TTR
            (adj_count + adv_count) / total_tags,               # 4. 수식어 밀도
            sum(1 for s in sentences if '"' in s or "'" in s or '\u201c' in s)
                / len(sentences),                                # 5. 대화문 비율
            np.std(sent_lens),                                   # 6. 문장 길이 표준편차
            noun_count / total_tags,                             # 7. 명사 비율
        ]
        results.append(feat)
        print(f"  {author}: 평균문장길이={feat[0]:.1f}, TTR={feat[2]:.3f}, "
              f"수식어밀도={feat[3]:.3f}")

    matrix = np.array(results)
    print(f"  통계 피처 shape: {matrix.shape}")
    return matrix


# ── 3-4. 벡터 결합 ──────────────────────────────────────────
def combine_features(tfidf: np.ndarray,
                     w2v:   np.ndarray,
                     stat:  np.ndarray) -> np.ndarray:
    """TF-IDF + Word2Vec + 통계 피처 정규화 후 결합"""
    scaler = StandardScaler()

    # 각 피처 그룹 정규화
    tfidf_scaled = scaler.fit_transform(tfidf)
    w2v_scaled   = scaler.fit_transform(w2v)
    stat_scaled  = scaler.fit_transform(stat)

    combined = np.hstack([tfidf_scaled, w2v_scaled, stat_scaled])
    print(f"  결합 벡터 shape: {combined.shape}")
    return combined


# ════════════════════════════════════════════════════════════
# 4. 모델링 — Ward 계층적 클러스터링
# ════════════════════════════════════════════════════════════

def run_clustering(features: np.ndarray,
                   authors:  list[str],
                   n_clusters: int = 2) -> dict:
    """Ward 클러스터링 + 실루엣 점수"""
    Z = linkage(features, method='ward')

    # 클러스터 레이블
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    # 실루엣 점수 (샘플 수 >= 2 일때만)
    if len(authors) >= 3:
        sil = silhouette_score(features, labels)
    else:
        sil = float('nan')

    print(f"\n[클러스터링 결과]")
    print(f"  군집 수: {n_clusters}, 실루엣 점수: {sil:.3f}")
    for author, label in zip(authors, labels):
        print(f"  {author}: 군집 {label}")

    return {'linkage': Z, 'labels': labels, 'silhouette': sil}


# ════════════════════════════════════════════════════════════
# 5. 시각화
# ════════════════════════════════════════════════════════════

def plot_dendrogram(Z, authors: list[str], sil: float):
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=authors, ax=ax, leaf_rotation=30)
    ax.set_title(f'작가 문체 덴드로그램 (실루엣={sil:.3f})', fontsize=13)
    ax.set_ylabel('거리')
    plt.tight_layout()
    path = OUTPUT_DIR / 'dendrogram.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


def plot_stat_heatmap(stat: np.ndarray, authors: list[str]):
    feat_names = ['평균문장길이', '단문비율', 'TTR', '수식어밀도',
                  '대화문비율', '문장길이편차', '명사비율']
    df = pd.DataFrame(stat, index=authors, columns=feat_names)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(df.values, aspect='auto', cmap='RdYlGn')
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=30, ha='right', fontsize=9)
    ax.set_yticks(range(len(authors)))
    ax.set_yticklabels(authors, fontsize=10)
    plt.colorbar(im, ax=ax)
    ax.set_title('작가별 문체 통계 피처 히트맵', fontsize=13)
    plt.tight_layout()
    path = OUTPUT_DIR / 'stat_heatmap.png'
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  저장: {path}")


# ════════════════════════════════════════════════════════════
# 6. 사용자 입력 분류기
# ════════════════════════════════════════════════════════════

def classify_user_text(user_text: str,
                       author_features: np.ndarray,
                       authors: list[str],
                       processed: dict,
                       w2v_model: Word2Vec) -> dict:
    """사용자 글 → 작가 유사도 TOP3 반환"""
    # 전처리
    cleaned = clean_text(user_text)
    tokens  = tokenize(cleaned)

    # TF-IDF (학습된 어휘 기준 수동 계산은 복잡 → 코사인 유사도로 대체)
    # Word2Vec 평균 벡터
    token_vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
    if not token_vecs:
        return {'error': '분석 가능한 단어가 없습니다.'}

    user_w2v = np.mean(token_vecs, axis=0)

    # 작가별 Word2Vec 평균 벡터 추출
    author_w2v = []
    for v in processed.values():
        tvecs = [w2v_model.wv[t] for t in v['tokens'] if t in w2v_model.wv]
        author_w2v.append(np.mean(tvecs, axis=0) if tvecs else np.zeros(W2V_DIM))

    # 코사인 유사도
    sims = [1 - cosine(user_w2v, av) for av in author_w2v]

    # softmax
    sims_arr = np.array(sims)
    exp_s    = np.exp((sims_arr - sims_arr.max()) * 10)
    softmax  = exp_s / exp_s.sum()

    # TOP3
    top3_idx = np.argsort(softmax)[::-1][:3]
    top3 = [(authors[i], float(softmax[i])) for i in top3_idx]

    result = {
        'top3':   top3,
        'top1':   top3[0][0],
        'scores': {a: float(s) for a, s in zip(authors, softmax)},
    }

    # 문체 유형 결정
    result['style_type'] = _get_style_type(top3[0][0], top3[1][0])
    return result


STYLE_MAP = {
    ('이상',  '김유정'): ('불안한 현실주의자형', '내면 고민이 깊으면서도 현실 감각이 살아있는 글'),
    ('이효석', '이상'):  ('몽환적 자기성찰형',   '감각 묘사를 통해 내면 감정을 전달하는 글'),
    ('이효석', '김유정'):('감각적 자기성찰형',   '서정적 분위기와 생활 감각이 공존하는 글'),
    ('김유정', '채만식'):('생활풍자형',          '일상적 말투로 씁쓸하고 웃긴 장면을 만드는 글'),
    ('이광수', '염상섭'):('계몽적 관찰형',        '사회 문제를 냉정하게 분석하는 글'),
    ('김동인', '이상'):  ('냉소적 내면형',        '짧고 건조한 문장 속에 심리를 담는 글'),
}

def _get_style_type(top1: str, top2: str) -> tuple[str, str]:
    key = (top1, top2)
    rev = (top2, top1)
    return STYLE_MAP.get(key) or STYLE_MAP.get(rev) or \
           ('독창적 탐색형', '특정 작가에 가깝지 않은 독자적 문체')


# ════════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 50)
    print("  문체 DNA 분석 파이프라인")
    print("=" * 50)

    # 1. 데이터 로딩
    print("\n[데이터 로딩]")
    texts = load_texts(DATA_DIR)
    authors = list(texts.keys())
    print(f"  작가 {len(authors)}명: {authors}")

    # 2. 전처리
    processed = preprocess_all(texts)

    # 3. 피처 추출
    print("\n[피처 추출]")
    tfidf = extract_tfidf(processed)
    w2v_model = train_word2vec(processed)
    w2v   = extract_w2v_mean(processed, w2v_model)
    stat  = extract_stat_features(processed)
    features = combine_features(tfidf, w2v, stat)

    # 4. 클러스터링
    clust = run_clustering(features, authors, n_clusters=2)

    # 5. 시각화
    print("\n[시각화 저장]")
    plot_dendrogram(clust['linkage'], authors, clust['silhouette'])
    plot_stat_heatmap(stat, authors)

    # 6. 사용자 테스트
    print("\n[사용자 글 분류 테스트]")
    test_text = "나는 거울 앞에 섰다. 거울 속의 나는 과연 나인가. 이 방 안에서 나는 무엇을 찾고 있는가."
    result = classify_user_text(test_text, features, authors, processed, w2v_model)
    if 'error' not in result:
        print(f"\n  입력 글: \"{test_text[:40]}...\"")
        print(f"  문체 유형: {result['style_type'][0]}")
        print(f"  설명: {result['style_type'][1]}")
        print("  유사도 TOP3:")
        for rank, (author, score) in enumerate(result['top3'], 1):
            print(f"    {rank}위: {author} {score*100:.1f}%")

    # 결과 저장
    summary = {
        'authors':    authors,
        'silhouette': clust['silhouette'],
        'clusters':   {a: int(l) for a, l in zip(authors, clust['labels'])},
    }
    with open(OUTPUT_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {OUTPUT_DIR}/summary.json")
    print("\n완료!")


if __name__ == '__main__':
    main()
