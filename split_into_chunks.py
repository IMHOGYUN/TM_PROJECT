import os
import re
import numpy as np
import itertools
from pathlib import Path
from tqdm import tqdm
from kiwipiepy import Kiwi

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import json 
import joblib

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. 기본 설정
# ==========================================
DATA_DIR = Path("data")
TARGET_CHUNK_SIZE = 800  
STOPWORDS = {'이', '가', '을', '를', '은', '는', '의', '에', '에서', '로', '으로', '와', '과', '도', '만', '하다', '있다', '없다', '되다', '이다', '그', '나', '내', '우리', '것', '수', '때', '곳', '말'}

kiwi = Kiwi()

# ==========================================
# 🌟 [신규 추가] 거시적 문체 통계 지표 추출 함수
# ==========================================
def get_macro_stats(text):
    total_chars = len(text.replace(" ", ""))
    if total_chars == 0:
        return [0.0, 0.0, 0.0, 0.0]
        
    # 1. 평균 문장 길이
    sentences = [s.text for s in kiwi.split_into_sents(text)]
    num_sentences = len(sentences) if len(sentences) > 0 else 1
    avg_sentence_len = total_chars / num_sentences
    
    # 2. 대화문(따옴표)의 비율
    quote_contents = re.findall(r'["\'「『](.*?)["\'」』]', text)
    total_quote_chars = sum([len(qc.replace(" ", "")) for qc in quote_contents])
    quote_ratio = total_quote_chars / total_chars
    
    # 3. 단문과 복문의 비율 (정규식 그룹핑 수정)
    clause_connectors = len(re.findall(r'(?:[,며고나]|하야)\s', text))
    complex_sentence_ratio = clause_connectors / num_sentences
    
    # 4. 특수 문장 부호 빈도
    special_punctuations = len(re.findall(r'[!?…]+', text))
    punctuation_density = special_punctuations / num_sentences

    return [avg_sentence_len, quote_ratio, complex_sentence_ratio, punctuation_density]

# ==========================================
# 1 & 2. 데이터 로드 및 청크 단위 다중 피처 추출
# ==========================================
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ.,!?\"\'()]', ' ', text)
    return text.strip()

print(f"\n[데이터 로딩 및 청크 분할 시작 (청크 크기: {TARGET_CHUNK_SIZE}어절)]")

raw_docs, word_docs, pos_docs, stats_features, tagged_data, labels = [], [], [], [], [], []
authors_list = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])

chunk_id = 0
for author in authors_list:
    author_dir = DATA_DIR / author
    for txt_file in author_dir.glob("*.txt"):
        try:
            text = txt_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = txt_file.read_text(encoding='cp949', errors='ignore')
        
        cleaned_text = clean_text(text)
        words = cleaned_text.split()
        
        for i in range(0, len(words), TARGET_CHUNK_SIZE):
            chunk_words = words[i:i+TARGET_CHUNK_SIZE]
            
            if len(chunk_words) >= TARGET_CHUNK_SIZE // 2:
                chunk_text = ' '.join(chunk_words)
                
                # 🌟 거시적 통계 지표 추출 (전처리 전 원문 형태 유지)
                macro_stats = get_macro_stats(chunk_text)
                
                # 형태소 분석
                analyzed = kiwi.analyze(chunk_text)
                tokens, pos_tags = [], []
                for token in analyzed[0][0]:
                    pos_tags.append(token.tag)
                    if token.tag.startswith(('N', 'V', 'M', 'X')):
                        lemma = token.form
                        if len(lemma) > 1 and lemma not in STOPWORDS:
                            tokens.append(lemma)
                
                # 리스트에 담기
                raw_docs.append(chunk_text)
                word_docs.append(" ".join(tokens))
                pos_docs.append(" ".join(pos_tags))
                stats_features.append(macro_stats) # 추출한 4가지 통계 지표 저장
                tagged_data.append(TaggedDocument(words=tokens, tags=[str(chunk_id)]))
                labels.append(author)
                chunk_id += 1

y = np.array([authors_list.index(l) for l in labels])
print(f"  ✅ 75편의 소설이 총 {len(raw_docs)}개의 데이터(Chunk)로 증강되었습니다!")

# ==========================================
# 3. 벡터화 및 전면 정규화(Scaling)
# ==========================================
print("\n>> 기계학습용 벡터 변환 및 정규화(Scaling) 중...")

vec_char = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=500, min_df=2)
X_char_raw = vec_char.fit_transform(raw_docs).toarray()

vec_pos = TfidfVectorizer(ngram_range=(2, 3), max_features=300, min_df=2)
X_pos_raw = vec_pos.fit_transform(pos_docs).toarray()

d2v_model = Doc2Vec(tagged_data, vector_size=100, window=5, min_count=2, workers=4, epochs=40, seed=42)
X_d2v_raw = np.array([d2v_model.dv[str(i)] for i in range(len(raw_docs))])

# 🌟 각 4가지 피처의 스케일러 독립 생성
scaler_d2v = StandardScaler()
X_d2v = scaler_d2v.fit_transform(X_d2v_raw)

scaler_pos = StandardScaler()
X_pos = scaler_pos.fit_transform(X_pos_raw)

scaler_char = StandardScaler()
X_char = scaler_char.fit_transform(X_char_raw)

scaler_style = StandardScaler()
X_style = scaler_style.fit_transform(stats_features)

# ==========================================
# 4. 동적 가중치 튜닝 (소프트 보팅 앙상블 - 4중 결합)
# ==========================================
print("\n" + "="*70)
print("🚀 [1단계] 4중 동적 가중치 최적화 앙상블 (수석 채점관: Logistic Regression)")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
clf_voting = LogisticRegression(max_iter=1000, random_state=42)

prob_d2v = cross_val_predict(clf_voting, X_d2v, y, cv=cv, method='predict_proba')
prob_pos = cross_val_predict(clf_voting, X_pos, y, cv=cv, method='predict_proba')
prob_char = cross_val_predict(clf_voting, X_char, y, cv=cv, method='predict_proba')
prob_style = cross_val_predict(clf_voting, X_style, y, cv=cv, method='predict_proba')

best_acc = 0.0
best_weights = (0, 0, 0, 0)
weight_range = np.arange(0.0, 1.05, 0.05)

print(">> 4개의 피처에 대한 황금 가중치 탐색 중... (잠시만 기다려주세요)")
for w1, w2, w3, w4 in itertools.product(weight_range, repeat=4):
    if np.isclose(w1 + w2 + w3 + w4, 1.0):
        blended_prob = (w1 * prob_d2v) + (w2 * prob_pos) + (w3 * prob_char) + (w4 * prob_style)
        acc = accuracy_score(y, np.argmax(blended_prob, axis=1))
        if acc > best_acc:
            best_acc = acc
            best_weights = (w1, w2, w3, w4)

print(f"🥇 최적의 황금 비율 : Doc2Vec {best_weights[0]*100:.0f}% | POS {best_weights[1]*100:.0f}% | Char {best_weights[2]*100:.0f}% | Style {best_weights[3]*100:.0f}%")
print(f"✨ 소프트 보팅 정확도: {best_acc*100:.2f}%")

# ==========================================
# 5. [데이터 × 알고리즘] 크로스 벤치마크 
# ==========================================
print("\n" + "="*70)
print("⚔️ [2단계] 알고리즘 × 데이터 조합 크로스 벤치마크 (Matrix Search)")
print("="*70)

# 🌟 4가지 데이터를 물리적으로 결합한 거대한 904차원 공간 생성!
X_all_combined = np.hstack([X_d2v, X_pos, X_char, X_style]) 
datasets = {
    "Doc2Vec (의미)": X_d2v,
    "POS N-gram (구문)": X_pos,
    "Char TF-IDF (리듬)": X_char,
    "Macro Style (통계)": X_style,
    "Super Hybrid (결합)": X_all_combined
}

classifiers = {
    "Logistic Regression ": LogisticRegression(max_iter=1000, random_state=42),
    "Naïve Bayes         ": GaussianNB(),
    "Support Vector(SVM) ": SVC(kernel='linear', probability=True, random_state=42),
    "Decision Tree       ": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest       ": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
    "Multi-layer Percept.": MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
}

best_cross_acc = 0.0
best_combo = ("", "")

for clf_name, model in classifiers.items():
    print(f"\n▶ {clf_name.strip()}")
    for data_name, X_data in datasets.items():
        try:
            scores = cross_val_score(model, X_data, y, cv=cv, scoring='accuracy')
            mean_acc = scores.mean() * 100
            print(f"   - {data_name:20s} -> {mean_acc:5.1f}%")
            if mean_acc > best_cross_acc:
                best_cross_acc = mean_acc
                best_combo = (clf_name.strip(), data_name.strip())
        except Exception as e:
            print(f"   - {data_name:20s} -> 에러 발생")
            
print("-" * 70)
print(f"🏆 우승 콤비: {best_combo[0]} + {best_combo[1]} ({best_cross_acc:.2f}%)")

# ==========================================
# 6. 핵심 데이터 압축 벤치마크 (가설 검증)
# ==========================================
print("\n" + "="*70)
print("⚔️ [3단계] 핵심 피처 물리적 결합 벤치마크 (가설 검증)")
print("="*70)

X_core_hybrid = np.hstack([X_d2v, X_pos])
best_core_acc = 0.0
best_core_clf = ""

for clf_name, model in classifiers.items():
    try:
        scores = cross_val_score(model, X_core_hybrid, y, cv=cv, scoring='accuracy')
        mean_acc = scores.mean() * 100
        if mean_acc > best_core_acc:
            best_core_acc = mean_acc
            best_core_clf = clf_name.strip()
    except Exception as e:
        pass

print(f"💡 핵심 물리적 결합 최고 점수: {best_core_clf} ({best_core_acc:.2f}%)")
print("="*70)

# ==========================================
# 7. 🚀 [최종 배포용] 학습 완료된 뇌(Model) 영구 저장 (Export)
# ==========================================
print("\n" + "="*70)
print("💾 [마무리] 최종 서비스용 AI 모델 저장 중...")
print("="*70)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# 🌟 4명의 셰프(SVM)를 각 피처별로 학습
final_clf_d2v = SVC(kernel='linear', probability=True, random_state=42).fit(X_d2v, y)
final_clf_pos = SVC(kernel='linear', probability=True, random_state=42).fit(X_pos, y)
final_clf_char = SVC(kernel='linear', probability=True, random_state=42).fit(X_char, y)
final_clf_style = SVC(kernel='linear', probability=True, random_state=42).fit(X_style, y) # 스타일 셰프 추가

joblib.dump(final_clf_d2v, MODEL_DIR / "clf_d2v.pkl")
joblib.dump(final_clf_pos, MODEL_DIR / "clf_pos.pkl")
joblib.dump(final_clf_char, MODEL_DIR / "clf_char.pkl")
joblib.dump(final_clf_style, MODEL_DIR / "clf_style.pkl") # 저장

joblib.dump(vec_pos, MODEL_DIR / "vec_pos.pkl")
joblib.dump(vec_char, MODEL_DIR / "vec_char.pkl")

joblib.dump(scaler_d2v, MODEL_DIR / "scaler_d2v.pkl")
joblib.dump(scaler_pos, MODEL_DIR / "scaler_pos.pkl")
joblib.dump(scaler_char, MODEL_DIR / "scaler_char.pkl")
joblib.dump(scaler_style, MODEL_DIR / "scaler_style.pkl") # 스케일러 저장

d2v_model.save(str(MODEL_DIR / "doc2vec.model"))
joblib.dump(authors_list, MODEL_DIR / "authors_list.pkl")

# 황금 가중치 4개 저장
weights_dict = {
    "d2v": float(best_weights[0]),
    "pos": float(best_weights[1]),
    "char": float(best_weights[2]),
    "style": float(best_weights[3]) # 스타일 가중치 추가
}

with open(MODEL_DIR / "golden_weights.json", "w", encoding="utf-8") as f:
    json.dump(weights_dict, f, indent=4)

print("  ✅ 4중 앙상블 모델 및 가중치가 완벽하게 저장되었습니다!")
print("="*70)