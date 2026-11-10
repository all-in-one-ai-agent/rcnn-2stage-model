# -*- coding: utf-8 -*-
r"""
얼굴 인식 핵심 로직 (화면과 무관)

  [사진] → SCRFD 얼굴 검출 → 5점 정렬(112×112) → ArcFace ResNet-50 (동결) → 512차원 임베딩
        ├─ 방식 A: kNN  — 팀원별 저장 임베딩과 코사인 유사도 (Top-3 평균)
        └─ 방식 B: 분류 레이어 — 512 → 128 → N  MLP 를 직접 학습 (numpy 구현, Adam)
  두 방식 모두 "팀원이 아닌 사람"은 kNN 유사도 임계값으로 걸러 '모름' 처리
"""
import hashlib
import pickle
import threading
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "face_model.pkl"
CACHE_PATH = MODEL_DIR / "emb_cache.pkl"
CURVE_PATH = MODEL_DIR / "training_curve.png"
OUTPUT = ROOT / "output"
EVAL_DIR = ROOT / "eval"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
UNKNOWN = "모름"
CACHE_VER = 2          # 검출 방식이 바뀌면 올려서 임베딩 캐시 무효화
TOP_K = 3              # kNN: 사람별 가장 비슷한 사진 K장의 평균
MAX_SIDE = 2400        # 너무 큰 사진은 줄여서 처리
EMB_DIM = 512
INPUT_SCALE = np.sqrt(EMB_DIM)   # 단위벡터 → 원소 분산≈1 로 스케일 (MLP 학습 안정화)
PALETTE = [(46, 204, 113), (52, 152, 219), (155, 89, 182), (241, 196, 15),
           (230, 126, 34), (26, 188, 156), (236, 64, 122), (121, 85, 72)]
METHODS = {"knn": "최근접 이웃 (kNN)", "head": "분류 레이어 (MLP)"}


def ensure_dirs():
    for d in (DATASET, MODEL_DIR, OUTPUT, EVAL_DIR):
        d.mkdir(exist_ok=True)


ensure_dirs()

# ───────────────────────── 얼굴 엔진 (사전학습 모델) ─────────────────────────
_engine = None
_lock = threading.RLock()


DEVICE = "미확인"


def engine():
    """NVIDIA GPU(CUDA)가 있으면 GPU, 없거나 실패하면 CPU로 자동 전환"""
    global _engine, DEVICE
    with _lock:
        if _engine is None:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis
            providers = ["CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers():
                try:
                    ort.preload_dlls()  # pip로 설치된 CUDA/cuDNN DLL 로드
                except Exception:
                    pass
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            app = FaceAnalysis(name="buffalo_l",
                               allowed_modules=["detection", "recognition", "landmark_2d_106"],
                               providers=providers)
            gpu = "CUDAExecutionProvider" in app.models["recognition"].session.get_providers()
            app.prepare(ctx_id=0 if gpu else -1, det_size=(1024, 1024))
            DEVICE = "GPU (CUDA)" if gpu else "CPU"
            _engine = app
        return _engine


def load_rgb(path):
    """한글 경로·휴대폰 회전정보(EXIF) 처리 후 RGB 배열. 실패 시 None."""
    try:
        im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        return None
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    return np.array(im)


_sha_cache = {}


def thumbnail(path, size=256):
    """목록 표시용 작은 사진. JPEG는 디코딩 단계에서 바로 축소해 빠름."""
    try:
        im = Image.open(path)
        im.draft("RGB", (size * 2, size * 2))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((size, size))
        return np.array(im)
    except Exception:
        return None


def file_sha1(path):
    """파일 내용 해시 (중복·누수 검사용). 경로+수정시각+크기가 같으면 다시 읽지 않음."""
    p = Path(path)
    st = p.stat()
    key = (str(p.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _sha_cache:
        _sha_cache[key] = hashlib.sha1(p.read_bytes()).hexdigest()
    return _sha_cache[key]


def setup_plot_font():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _area(f):
    return (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])


def _iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _detect_raw(rgb):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    with _lock:
        return list(engine().get(bgr))


def _detect_padded(rgb, ratio=0.25):
    """사진 둘레에 회색 여백을 넣고 검출 → 좌표를 원본 기준으로 되돌림.
    얼굴이 화면을 꽉 채운 초근접 셀카는 검출기가 놓치기 쉬워서, 여백을 주면 '보통 크기 얼굴'이 되어 잡힘."""
    h, w = rgb.shape[:2]
    py, px = int(h * ratio), int(w * ratio)
    padded = cv2.copyMakeBorder(rgb, py, py, px, px, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    faces = _detect_raw(padded)
    off = np.array([px, py], dtype=np.float32)
    for f in faces:
        f.bbox = f.bbox - np.concatenate([off, off])
        f.bbox = np.clip(f.bbox, 0, [w, h, w, h]).astype(np.float32)
        if f.get("kps") is not None:
            f.kps = f.kps - off
        if f.get("landmark_2d_106") is not None:
            f.landmark_2d_106 = f.landmark_2d_106 - off
    return faces   # normed_embedding은 여백 넣은 사진에서 정렬·추출한 값이라 그대로 유효


def detect(rgb, close_up=False):
    """얼굴 목록 (큰 얼굴 순). 각 얼굴: bbox, kps, landmark_2d_106, normed_embedding(512)
    close_up=True(셀카 등록): 여백 검출을 항상 함께 돌려 합침 → 근접 얼굴도, 배경 얼굴도 모두 후보에 포함
    close_up=False(단체사진): 하나도 못 찾았을 때만 여백 검출로 재시도"""
    faces = _detect_raw(rgb)
    if close_up or not faces:
        for f in _detect_padded(rgb):
            if all(_iou(f.bbox, g.bbox) < 0.4 for g in faces):
                faces.append(f)
    return sorted(faces, key=lambda f: -_area(f))


def crop_face(rgb, bbox, pad=0.35):
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - bw * pad)), int(max(0, y1 - bh * pad))
    x2, y2 = int(min(w, x2 + bw * pad)), int(min(h, y2 + bh * pad))
    return rgb[y1:y2, x1:x2]


def selfie_face(rgb):
    """라벨링 사진에서 주인공 얼굴 1개. (face, 사유)"""
    faces = detect(rgb, close_up=True)
    if not faces:
        return None, "얼굴 없음 (얼굴이 가려졌거나 너무 흐림/어두움)"
    if len(faces) > 1:
        if _area(faces[1]) > 0.5 * _area(faces[0]):
            return None, f"비슷한 크기 얼굴 {len(faces)}개 (누가 주인공인지 모호)"
    return faces[0], "ok"


# ───────────────────────── 데이터셋 ─────────────────────────
def people():
    return sorted(p.name for p in DATASET.iterdir() if p.is_dir())


def person_images(name):
    d = DATASET / name
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else []


def dataset_hashes():
    """학습 데이터 전체의 파일 해시 → (이름, 파일명). 같은 사진 중복 등록 방지용."""
    out = {}
    for n in people():
        for p in person_images(n):
            out.setdefault(file_sha1(p), (n, p.name))
    return out


def group_sources():
    """단체사진에서 잘라 학습에 넣은 얼굴들의 원본 해시 앞 8자리 → 파일명 목록"""
    out = {}
    for n in people():
        for p in person_images(n):
            if p.name.startswith("group_"):
                out.setdefault(p.name.split("_")[1], []).append(f"{n}/{p.name}")
    return out


def safe_name(name):
    name = (name or "").strip()
    bad = '\\/:*?"<>|'
    if not name or any(c in bad for c in name) or name in (".", "..", UNKNOWN):
        return None
    return name


def embed_dataset(progress=None):
    """dataset 전체 → {이름: [(경로, 임베딩)]}, 제외목록. 임베딩은 캐시해서 재학습을 빠르게."""
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        cache = {}
    new_cache, data, skipped = {}, {}, []
    files = [(n, p) for n in people() for p in person_images(n)]
    for i, (name, path) in enumerate(files):
        if progress:
            progress((i + 1) / max(1, len(files)), desc=f"얼굴 특징 추출 {i + 1}/{len(files)}")
        st = path.stat()
        key = (CACHE_VER, str(path.relative_to(DATASET)), st.st_mtime, st.st_size)
        if key in cache:
            emb, reason = cache[key]
        else:
            rgb = load_rgb(path)
            if rgb is None:
                emb, reason = None, "열 수 없는 파일"
            else:
                face, reason = selfie_face(rgb)
                emb = face.normed_embedding.astype(np.float32) if face is not None else None
        new_cache[key] = (emb, reason)
        if emb is None:
            skipped.append((name, path.name, reason))
        else:
            data.setdefault(name, []).append((path, emb))
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(new_cache, f)
    return data, skipped


# ───────────────────────── 방식 A: kNN ─────────────────────────
def knn_matrix(E, gallery, names):
    """E (F,512) → 점수행렬 (F, 사람수): 사람별 Top-K 코사인 유사도 평균"""
    S = np.zeros((len(E), len(names)), dtype=np.float32)
    for j, n in enumerate(names):
        sims = E @ gallery[n].T
        k = min(TOP_K, sims.shape[1])
        S[:, j] = np.sort(sims, axis=1)[:, -k:].mean(axis=1)
    return S


def knn_loo(gallery):
    """Leave-one-out: 사진 1장씩 빼고 나머지로 맞히기 → (정확도, 사람별, 본인점수, 타인최고점수, 오답)"""
    names = list(gallery)
    correct = total = 0
    genuine, impostor, per, confusions = [], [], {}, []
    for ci, n in enumerate(names):
        X = gallery[n]
        ok = 0
        for i in range(len(X) if len(X) >= 2 else 0):   # 1장뿐이면 빼고 맞힐 수 없음
            sc = {}
            for m in names:
                s = gallery[m] @ X[i]
                if m == n:
                    s = np.delete(s, i)
                k = min(TOP_K, len(s))
                sc[m] = float(np.sort(s)[-k:].mean())
            pred = max(sc, key=sc.get)
            total += 1
            ok += pred == n
            correct += pred == n
            if pred != n:
                confusions.append((n, i, pred))
            genuine.append(sc[n])
            others = [v for k_, v in sc.items() if k_ != n]
            if others:
                impostor.append(max(others))
        per[n] = (ok, len(X))
    return (correct / total if total else None), per, genuine, impostor, confusions


def pick_threshold(genuine, impostor):
    """본인 점수 / 타인 점수를 가장 잘 가르는 임계값 (같은 성능 구간의 가운데)"""
    if not genuine or not impostor:
        return 0.35
    g, im = np.array(genuine), np.array(impostor)
    ts = np.arange(0.15, 0.70, 0.005)
    bal = np.array([((g >= t).mean() + (im < t).mean()) / 2 for t in ts])
    t = float(np.median(ts[bal >= bal.max() - 1e-9]))
    return float(np.clip(t, 0.25, 0.50))


# ───────────────────────── 방식 B: 분류 레이어 (MLP, numpy 구현) ─────────────────────────
def head_init(n_cls, hidden, seed=0):
    rng = np.random.default_rng(seed)
    p = {}
    d = EMB_DIM
    if hidden:
        p["W1"] = rng.normal(0, np.sqrt(2 / d), (d, hidden)).astype(np.float32)   # He 초기화
        p["b1"] = np.zeros(hidden, np.float32)
        d = hidden
    p["W2"] = rng.normal(0, np.sqrt(1 / d), (d, n_cls)).astype(np.float32)
    p["b2"] = np.zeros(n_cls, np.float32)
    return p


def head_forward(p, X, drop=0.0, rng=None):
    """X(B,512) → logits(B,N).  Linear(512→H) → ReLU → Dropout → Linear(H→N)"""
    c = {"X": X * INPUT_SCALE}
    if "W1" in p:
        c["h"] = c["X"] @ p["W1"] + p["b1"]
        a = np.maximum(c["h"], 0)
        if drop > 0:
            c["mask"] = (rng.random(a.shape) >= drop) / (1 - drop)
            a = a * c["mask"]
        c["a"] = a
        z = a @ p["W2"] + p["b2"]
    else:
        z = c["X"] @ p["W2"] + p["b2"]
    return z, c


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def head_proba(p, X):
    return softmax(head_forward(p, X)[0])


def head_loss_grad(p, X, Y, w, wd, drop, rng):
    """가중 크로스엔트로피 + L2 → (loss, 기울기)  (역전파 직접 계산)"""
    z, c = head_forward(p, X, drop, rng)
    P = softmax(z)
    wn = w / w.sum()
    loss = -(wn * np.log((P * Y).sum(1) + 1e-12)).sum()
    loss += 0.5 * wd * sum((p[k] ** 2).sum() for k in p if k[0] == "W")
    dz = (P - Y) * wn[:, None]
    g = {}
    inp = c["a"] if "W1" in p else c["X"]
    g["W2"] = inp.T @ dz + wd * p["W2"]
    g["b2"] = dz.sum(0)
    if "W1" in p:
        da = dz @ p["W2"].T
        if "mask" in c:
            da = da * c["mask"]
        dh = da * (c["h"] > 0)
        g["W1"] = c["X"].T @ dh + wd * p["W1"]
        g["b1"] = dh.sum(0)
    return float(loss), g


def _eval_head(p, X, y):
    P = head_proba(p, X)
    loss = float(-np.log(P[np.arange(len(y)), y] + 1e-12).mean())
    return loss, float((P.argmax(1) == y).mean())


def train_head(X, y, n_cls, hidden=128, epochs=300, lr=1e-3, wd=1e-3, drop=0.3, Xv=None, yv=None, seed=0):
    """전체배치 Adam 학습. 에폭마다 train/val 손실·정확도 기록."""
    rng = np.random.default_rng(seed)
    p = head_init(n_cls, hidden, seed)
    Y = np.eye(n_cls, dtype=np.float32)[y]
    counts = np.bincount(y, minlength=n_cls).astype(np.float32)
    w = (len(y) / (n_cls * np.maximum(counts, 1)))[y]            # 사람별 사진 수 불균형 보정
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v_ = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.999, 1e-8
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for t in range(1, epochs + 1):
        _, g = head_loss_grad(p, X, Y, w, wd, drop, rng)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * g[k]
            v_[k] = b2 * v_[k] + (1 - b2) * g[k] ** 2
            p[k] -= lr * (m[k] / (1 - b1 ** t)) / (np.sqrt(v_[k] / (1 - b2 ** t)) + eps)
        tl, ta = _eval_head(p, X, y)
        hist["train_loss"].append(tl)
        hist["train_acc"].append(ta)
        if Xv is not None and len(Xv):
            vl, va = _eval_head(p, Xv, yv)
            hist["val_loss"].append(vl)
            hist["val_acc"].append(va)
    return p, hist


def stratified_split(y, ratio=0.2, seed=0):
    """사람별로 20%를 검증용으로 떼어냄 (3장 미만인 사람은 전부 학습용)"""
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        nv = max(1, int(round(len(idx) * ratio))) if len(idx) >= 3 else 0
        va += idx[:nv].tolist()
        tr += idx[nv:].tolist()
    return np.array(tr), np.array(va)


def head_param_table(n_cls, hidden):
    rows = [("백본", "ArcFace ResNet-50 (Conv 53 · BN 26 · PReLU 25 · 잔차블록 24)", "512", "43,600,000", "🔒 동결 (사전학습)")]
    if hidden:
        rows.append(("헤드 1", f"Linear {EMB_DIM}→{hidden} + ReLU + Dropout(0.3)", str(hidden), f"{EMB_DIM * hidden + hidden:,}", "✏️ 직접 학습"))
        rows.append(("헤드 2", f"Linear {hidden}→{n_cls} + Softmax", str(n_cls), f"{hidden * n_cls + n_cls:,}", "✏️ 직접 학습"))
    else:
        rows.append(("헤드", f"Linear {EMB_DIM}→{n_cls} + Softmax", str(n_cls), f"{EMB_DIM * n_cls + n_cls:,}", "✏️ 직접 학습"))
    return rows


def plot_curve(hist, knn_val_acc, path):
    plt = setup_plot_font()
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, hist["train_loss"], label="학습 손실")
    if hist["val_loss"]:
        ax[0].plot(ep, hist["val_loss"], label="검증 손실")
    ax[0].set_xlabel("에폭")
    ax[0].set_title("손실 (Cross-Entropy)")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    ax[1].plot(ep, np.array(hist["train_acc"]) * 100, label="학습 정확도")
    if hist["val_acc"]:
        ax[1].plot(ep, np.array(hist["val_acc"]) * 100, label="검증 정확도 (MLP)")
    if knn_val_acc is not None:
        ax[1].axhline(knn_val_acc * 100, color="gray", ls="--", label="검증 정확도 (kNN)")
    ax[1].set_ylim(0, 102)
    ax[1].set_xlabel("에폭")
    ax[1].set_title("정확도 (%)")
    ax[1].legend(loc="lower right")
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ───────────────────────── 학습 (A + B) ─────────────────────────
def train(progress=None, hidden=128, epochs=300):
    data, skipped = embed_dataset(progress)
    gallery = {n: np.stack([e for _, e in v]) for n, v in data.items()}
    if not gallery:
        raise ValueError("학습할 데이터가 없습니다. 먼저 1. 데이터셋 등록에서 사진을 등록하세요.")
    names = list(gallery)
    X = np.concatenate([gallery[n] for n in names])
    y = np.concatenate([[i] * len(gallery[n]) for i, n in enumerate(names)])

    # A. kNN: Leave-one-out 검증 + 임계값
    loo_acc, per, genuine, impostor, confusions = knn_loo(gallery)
    threshold = pick_threshold(genuine, impostor)

    # B. 분류 레이어: 80/20 검증 → 곡선 / 이후 전체 데이터로 최종 학습
    head, comp = None, {}
    if len(names) >= 2:
        tr, va = stratified_split(y)
        Xv, yv = (X[va], y[va]) if len(va) else (None, None)
        _, hist = train_head(X[tr], y[tr], len(names), hidden, epochs, Xv=Xv, yv=yv)
        knn_val = None
        if len(va):
            g_tr = {n: X[tr][y[tr] == i] for i, n in enumerate(names) if (y[tr] == i).any()}
            nm_tr = list(g_tr)
            pred = knn_matrix(X[va], g_tr, nm_tr).argmax(1)
            knn_val = float(np.mean([names.index(nm_tr[p_]) == t for p_, t in zip(pred, yv)]))
        plot_curve(hist, knn_val, CURVE_PATH)
        params, _ = train_head(X, y, len(names), hidden, epochs)
        head = {"params": params, "hidden": hidden, "epochs": epochs, "history": hist}
        comp = {"n_train": len(tr), "n_val": len(va), "knn_val": knn_val,
                "head_val": hist["val_acc"][-1] if hist["val_acc"] else None,
                "head_train": hist["train_acc"][-1]}

    model = {"gallery": gallery, "names": names, "threshold": threshold, "head": head,
             "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "loo_acc": loo_acc, "comparison": comp}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # ── 리포트
    L = [f"## 학습 완료 ({model['trained_at']})", "", "### 데이터", "",
         "| 이름 | 사용된 사진 | kNN 자기검증 (LOO) |", "|---|---|---|"]
    for n in names:
        ok, cnt = per[n]
        L.append(f"| {n} | {cnt}장 | " + (f"{ok / cnt * 100:.0f}% ({ok}/{cnt})" if cnt > 1 else "2장 이상 필요") + " |")
    if loo_acc is not None:
        L += ["", f"- **kNN 자기검증 정확도: {loo_acc * 100:.1f}%** (사진 1장씩 빼고 나머지로 맞히기)"]
    L += [f"- **'{UNKNOWN}' 판정 임계값: {threshold:.3f}** (팀원 사진과의 유사도가 이보다 낮으면 외부인)"]
    if genuine and impostor:
        L += [f"- 본인 유사도 평균 {np.mean(genuine):.3f} / 다른 팀원 유사도 평균 {np.mean(impostor):.3f}"]

    if head:
        L += ["", "### 분류 레이어 구조", "", "| 단계 | 레이어 | 출력 | 파라미터 | 학습 |", "|---|---|---|---|---|"]
        L += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in head_param_table(len(names), hidden)]
        L += ["", f"학습 설정: Adam(lr=0.001), 에폭 {epochs}, L2 가중감쇠 0.001, 드롭아웃 0.3, 클래스 불균형 가중치, 입력 √512 스케일"]
        L += ["", "### 방식 비교 (같은 80/20 분할)", "",
              f"학습용 {comp['n_train']}장 / 검증용 {comp['n_val']}장 (검증용은 학습에 사용 안 함)", "",
              "| 방식 | 학습 정확도 | 검증 정확도 |", "|---|---|---|"]
        hv = f"{comp['head_val'] * 100:.1f}%" if comp["head_val"] is not None else "-"
        kv = f"{comp['knn_val'] * 100:.1f}%" if comp["knn_val"] is not None else "-"
        L += [f"| 최근접 이웃 (kNN) | - (학습 없음) | {kv} |",
              f"| 분류 레이어 (MLP) | {comp['head_train'] * 100:.1f}% | {hv} |"]
        L += ["", "최종 모델은 검증 후 **전체 데이터로 다시 학습**해 저장했습니다. 학습 곡선은 아래 그래프를 보세요."]
    else:
        L += ["", "ℹ️ 분류 레이어는 2명 이상 등록해야 학습됩니다."]

    if confusions:
        L += ["", "### 헷갈린 사진 (라벨이 틀렸거나 얼굴이 잘 안 보이는 사진일 수 있음)"]
        L += [f"- `{n}/{data[n][i][0].name}` → {p_}(으)로 예측" for n, i, p_ in confusions[:20]]
    if skipped:
        L += ["", f"### 학습에서 제외된 사진 {len(skipped)}장"]
        L += [f"- `{n}/{f}`: {r}" for n, f, r in skipped[:30]]
    small = [n for n in names if len(gallery[n]) < 10]
    if small:
        L += ["", f"⚠️ 사진이 10장 미만인 사람: {', '.join(small)} — 15장 이상 권장"]
    return "\n".join(L)


def load_model():
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    m.setdefault("names", list(m["gallery"]))
    m.setdefault("head", None)
    m.setdefault("comparison", {})
    return m


# ───────────────────────── 인식 ─────────────────────────
def score_matrices(E, model, method):
    """→ (S_assign: 배정에 쓰는 점수, S_gate: '모름' 판정용 kNN 유사도, S_show: 표시용 점수)"""
    S_knn = knn_matrix(E, model["gallery"], model["names"])
    if method == "head" and model.get("head"):
        P = head_proba(model["head"]["params"], E)
        return np.log(P + 1e-9), S_knn, P
    return S_knn, S_knn, S_knn


def assign_names(S_assign, S_gate, th, unique=True):
    """얼굴별 사람 인덱스 (-1 = 모름). unique면 한 사진에 같은 사람 1번만 (헝가리안 알고리즘)."""
    F = S_assign.shape[0]
    out = [-1] * F
    if F == 0 or S_assign.shape[1] == 0:
        return out
    if unique:
        rows, cols = linear_sum_assignment(-S_assign)
        pairs = zip(rows, cols)
    else:
        pairs = ((r, int(S_assign[r].argmax())) for r in range(F))
    for r, c in pairs:
        if S_gate[r, c] >= th:
            out[r] = int(c)
    return out


# ───────────────────────── 몸 전체 윤곽: Mask R-CNN (COCO 사전학습) ─────────────────────────
_body = None
BODY_DEVICE = "미사용"
BODY_SCORE = 0.6       # 사람 검출 신뢰도 하한
BODY_MATCH = 0.35      # 얼굴 박스의 몇 % 이상이 그 몸 마스크 안에 있어야 같은 사람으로 연결할지


def body_available():
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


def body_model():
    """torchvision Mask R-CNN ResNet-50-FPN v2 (COCO 사전학습, 'person' 클래스 사용). 처음 한 번 가중치 다운로드."""
    global _body, BODY_DEVICE
    with _lock:
        if _body is None:
            import torch
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)
            _body = (m, dev)
            BODY_DEVICE = "GPU (CUDA)" if dev == "cuda" else "CPU"
        return _body


def detect_bodies(rgb):
    """사람별 몸 전체 마스크 (P, H, W) bool"""
    import torch
    m, dev = body_model()
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).to(dev)
    with _lock, torch.inference_mode():
        out = m([t])[0]
    keep = (out["labels"] == 1) & (out["scores"] >= BODY_SCORE)          # COCO 1 = person
    return (out["masks"][keep, 0] > 0.5).cpu().numpy()


def match_bodies(faces, masks):
    """얼굴 ↔ 몸 마스크 짝짓기: 얼굴 박스가 마스크 안에 들어가는 비율로 점수 → 헝가리안 (몸 하나에 얼굴 하나)"""
    out = [None] * len(faces)
    if not len(faces) or not len(masks):
        return out
    H, W = masks.shape[1:]
    S = np.zeros((len(faces), len(masks)))
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = np.clip(f.bbox, 0, [W, H, W, H]).astype(int)
        if x2 > x1 and y2 > y1:
            S[i] = masks[:, y1:y2, x1:x2].reshape(len(masks), -1).mean(1)
    rows, cols = linear_sum_assignment(-S)
    for r, c in zip(rows, cols):
        if S[r, c] >= BODY_MATCH:
            out[r] = masks[c]
    return out


def recognize(rgb, model, threshold=None, unique=True, method="knn", body=False):
    """→ 얼굴별 dict(face, name, score, sim, alt, alt_score, text, body)
    body=True: Mask R-CNN으로 몸 전체 마스크를 찾아 얼굴과 연결 (r['body'] = 마스크 또는 None)"""
    th = model["threshold"] if threshold is None else threshold
    faces = detect(rgb)
    if not faces:
        return []
    names = model["names"]
    E = np.stack([f.normed_embedding for f in faces])
    S_a, S_g, S_show = score_matrices(E, model, method)
    idx = assign_names(S_a, S_g, th, unique)
    fmt = (lambda v: f"{v * 100:.0f}%") if (method == "head" and model.get("head")) else (lambda v: f"{v:.2f}")
    res = []
    for i, f in enumerate(faces):
        order = np.argsort(-S_show[i])
        c = idx[i]
        name = names[c] if c >= 0 else UNKNOWN
        main = c if c >= 0 else int(order[0])
        alt = next((j for j in order if j != c), None)
        # MLP의 소프트맥스 확률은 외부인에게도 높게 나오므로(닫힌 집합 분류의 한계) '모름'은 유사도로 표시
        text = f"{name} {fmt(S_show[i, main])}" if c >= 0 else f"{UNKNOWN} {S_g[i, main]:.2f}"
        res.append({"face": f, "name": name, "score": float(S_show[i, main]), "sim": float(S_g[i, main]),
                    "alt": names[alt] if alt is not None else "", "alt_score": float(S_show[i, alt]) if alt is not None else 0.0,
                    "text": text, "fmt": fmt, "body": None})
    if body and body_available():
        for r, mk in zip(res, match_bodies(faces, detect_bodies(rgb))):
            r["body"] = mk
    return res


# ───────────────────────── 그리기 ─────────────────────────
def color_of(name, model):
    if name == UNKNOWN:
        return (220, 50, 50)
    if model and name in model["gallery"]:
        return PALETTE[list(model["gallery"]).index(name) % len(PALETTE)]
    ps = people()
    if name in ps:
        return PALETTE[ps.index(name) % len(PALETTE)]
    return (140, 140, 140)


# 106점 랜드마크 중 턱선 33점: 왼쪽 관자놀이 → 턱 → 오른쪽 관자놀이 순서
JAW_ORDER = [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
             24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17]


def face_outline(f):
    """얼굴 윤곽 다각형. 턱선=랜드마크, 이마=관자놀이를 잇는 타원 호(추정)."""
    lm = getattr(f, "landmark_2d_106", None)
    if lm is None:
        x1, y1, x2, y2 = f.bbox
        return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    jaw = lm[JAW_ORDER]
    left, right, chin = jaw[0], jaw[-1], lm[0]
    c = (left + right) / 2
    a = np.linalg.norm(right - left) / 2
    ux = (right - left) / (2 * a)
    uy = np.array([ux[1], -ux[0]])
    if np.dot(chin - c, uy) > 0:
        uy = -uy
    b = 0.55 * np.linalg.norm(chin - c)
    t = np.linspace(0, np.pi, 24)[1:-1]
    forehead = c + np.outer(np.cos(t) * a, ux) + np.outer(np.sin(t) * b, uy)
    return np.vstack([jaw, forehead])


def draw_results(rgb, results, model, show_index=False):
    """results: dict(face, name, text[, body]) 목록
    → 몸 전체(있으면) 반투명 채우기+테두리, 이름표(이름+수치). show_index=True면 번호와 얼굴 윤곽도 표시(평가 정답 지정용)"""
    fs = max(16, int(max(rgb.shape[:2]) / 55))
    try:
        font = ImageFont.truetype(FONT_PATH, fs)
    except Exception:
        font = ImageFont.load_default()
    lw = max(2, fs // 8)

    # 1) 몸 전체: 마스크 영역을 사람 색으로 반투명하게 + 바깥 테두리
    arr = rgb.astype(np.float32).copy()
    contours = []
    for r in results:
        mk = r.get("body")
        if mk is None:
            continue
        col = np.array(color_of(r["name"], model), np.float32)
        arr[mk] = arr[mk] * 0.72 + col * 0.28
        cs, _ = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            big = max(cv2.contourArea(c) for c in cs)
            contours.append(([c for c in cs if cv2.contourArea(c) >= 0.02 * big], color_of(r["name"], model)))
    arr = arr.clip(0, 255).astype(np.uint8)
    for cs, col in contours:
        cv2.drawContours(arr, cs, -1, col, lw, lineType=cv2.LINE_AA)

    # 2) 얼굴 윤곽: 몸 윤곽이 없는 사람(몸이 안 보이거나 Mask R-CNN이 못 찾은 경우)에게만 표시
    im = Image.fromarray(arr).convert("RGBA")
    overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    face_lines = []
    for r in results:
        if r.get("body") is None or show_index:
            pts = [tuple(map(float, p)) for p in face_outline(r["face"])]
            od.polygon(pts, fill=color_of(r["name"], model) + (70,))
            face_lines.append((pts, color_of(r["name"], model)))
    im = Image.alpha_composite(im, overlay).convert("RGB")
    d = ImageDraw.Draw(im)
    for pts, col in face_lines:
        d.line(pts + [pts[0]], fill=col, width=max(2, lw - 1), joint="curve")

    # 3) 이름표: 서로 겹치지 않는 자리를 골라 배치, 얼굴에서 떨어지면 연결선
    labels = [(f"{i}. {r['text']}" if show_index else r["text"]) for i, r in enumerate(results, 1)]
    sizes = []
    for t in labels:
        tb = d.textbbox((0, 0), t, font=font)
        sizes.append((tb[2] - tb[0] + 10, tb[3] - tb[1] + 10, tb))
    boxes = place_labels([r["face"].bbox for r in results], [(w, h) for w, h, _ in sizes], im.width, im.height)
    for r, t, (w, h, tb), (x, y, near) in zip(results, labels, sizes, boxes):
        col = color_of(r["name"], model)
        if not near:
            fx1, fy1, fx2, fy2 = r["face"].bbox
            fc_ = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)
            lx = min(max(fc_[0], x), x + w)
            ly = min(max(fc_[1], y), y + h)
            d.line([(lx, ly), fc_], fill=col, width=max(2, lw - 1))
            d.ellipse([fc_[0] - lw, fc_[1] - lw, fc_[0] + lw, fc_[1] + lw], fill=col)
        d.rectangle([x, y, x + w, y + h], fill=col, outline=(255, 255, 255), width=1)
        d.text((x + 5, y + 5 - tb[1]), t, font=font, fill=(255, 255, 255))
    return np.array(im)


def _overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def place_labels(face_boxes, sizes, W, H):
    """이름표 위치 정하기 (탐욕적 배치).
    얼굴 위 → 아래 → 좌/우 → 점점 멀리 순서로 후보를 만들고,
    '다른 이름표와 겹침(가장 큰 벌점) + 다른 얼굴 가림 + 얼굴과의 거리'가 가장 작은 후보를 고름.
    → [(x, y, 얼굴에 붙어있는지)]"""
    order = sorted(range(len(face_boxes)), key=lambda i: (face_boxes[i][1], face_boxes[i][0]))
    placed, out = [], [None] * len(face_boxes)
    for i in order:
        x1, y1, x2, y2 = face_boxes[i]
        w, h = sizes[i]
        cx, cy, fw = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1
        g = 4
        cands = [(cx - w / 2, y1 - h - g, True), (cx - w / 2, y2 + g, True),
                 (x1 - w - g, cy - h / 2, True), (x2 + g, cy - h / 2, True)]
        for k in range(1, 9):                                   # 근처에 자리가 없으면 점점 멀리
            for dx in (0, -0.6, 0.6):
                cands.append((cx - w / 2 + dx * (w + fw), y1 - h - g - k * (h + 3), False))
                cands.append((cx - w / 2 + dx * (w + fw), y2 + g + k * (h + 3), False))
        best = None
        for x, y, near in cands:
            x = min(max(0, x), W - w)
            y = min(max(0, y), H - h)
            r = (x, y, x + w, y + h)
            cost = sum(_overlap(r, p) for p in placed) * 50
            cost += sum(_overlap(r, fb) for j, fb in enumerate(face_boxes) if j != i) * 5
            cost += _overlap(r, face_boxes[i]) * 2                # 자기 얼굴도 되도록 가리지 않기
            cost += np.hypot(x + w / 2 - cx, y + h / 2 - cy)
            if best is None or cost < best[0]:
                best = (cost, x, y, near and _overlap(r, (x1 - g * 3, y1 - h - g * 3, x2 + g * 3, y2 + h + g * 3)) > 0)
        _, x, y, near = best
        placed.append((x, y, x + w, y + h))
        out[i] = (int(x), int(y), near)
    return out
