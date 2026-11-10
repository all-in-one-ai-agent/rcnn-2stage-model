# -*- coding: utf-8 -*-
r"""
평가: 정답(누가 어디 있는지)을 적어둔 단체사진으로 실제 정확도 측정

  eval/<id>.jpg + eval/<id>.json  {"sha1": 원본해시, "faces": [{"bbox": [x1,y1,x2,y2], "label": 이름|"모름"}]}
  - 라벨 "모름" = 팀원이 아닌 사람 → 프로그램이 '모름'이라고 해야 정답
  - 평가 때 얼굴을 다시 검출해 정답 박스와 IoU로 짝지은 뒤 kNN / 분류 레이어 결과를 각각 채점
"""
import json
import time
import uuid

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

import face_core as fc

UNDETECTED = "미검출"


def eval_items():
    return sorted(fc.EVAL_DIR.glob("*.json"))


def eval_hashes():
    return {json.loads(p.read_text(encoding="utf-8"))["sha1"]: p.stem for p in eval_items()}


def eval_conflict(sha):
    """이 사진을 평가셋에 넣으면 안 되는 이유 (없으면 None)"""
    if sha in eval_hashes():
        return "이미 평가셋에 들어 있는 사진입니다."
    used = fc.group_sources().get(sha[:8])
    if used:
        return f"이 단체사진의 얼굴이 이미 학습 데이터로 쓰였습니다 ({len(used)}개, 예: {used[0]}). 학습에 쓴 사진은 평가에 쓸 수 없습니다."
    known = fc.dataset_hashes()
    if sha in known:
        n, f = known[sha]
        return f"학습 데이터에 같은 사진이 있습니다 ({n}/{f}). 학습에 쓴 사진은 평가에 쓸 수 없습니다."
    return None


def save_eval_item(src_path, rgb, bboxes, labels):
    sha = fc.file_sha1(src_path)
    why = eval_conflict(sha)
    if why:
        raise ValueError(why)
    faces = [{"bbox": [float(v) for v in b], "label": l} for b, l in zip(bboxes, labels) if l not in ("?", "제외")]
    if not faces:
        raise ValueError("정답이 지정된 얼굴이 없습니다.")
    item_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    Image.fromarray(rgb).save(fc.EVAL_DIR / f"{item_id}.jpg", quality=95)  # 박스 좌표와 같은 해상도로 저장
    meta = {"sha1": sha, "faces": faces, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (fc.EVAL_DIR / f"{item_id}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return item_id, len(faces)


def clear_eval():
    for p in fc.EVAL_DIR.glob("*"):
        if p.suffix in (".json", ".jpg"):
            p.unlink()


def eval_summary():
    items = eval_items()
    n_faces = sum(len(json.loads(p.read_text(encoding="utf-8"))["faces"]) for p in items)
    return len(items), n_faces


def leakage_check():
    """평가 사진이 학습 데이터에도 들어있으면 경고 (같은 사진으로 시험 보는 것 = 부정행위)"""
    ev = set(eval_hashes())
    prefixes = {s[:8] for s in ev}
    hits = []
    for n in fc.people():
        for p in fc.person_images(n):
            if p.name.startswith("group_"):
                if p.name.split("_")[1] in prefixes:
                    hits.append(f"{n}/{p.name}")
            elif fc.file_sha1(p) in ev:
                hits.append(f"{n}/{p.name}")
    return hits


def run_evaluation(model, threshold=None, unique=True, progress=None):
    items = eval_items()
    if not items:
        raise ValueError("평가셋이 비어 있습니다. 먼저 정답이 있는 단체사진을 평가셋에 저장하세요.")
    th = model["threshold"] if threshold is None else threshold
    names = model["names"]
    methods = ["knn"] + (["head"] if model.get("head") else [])

    # 1) 사진마다 한 번만 검출·임베딩 → 점수행렬 저장
    photos = []
    for k, jp in enumerate(items):
        if progress:
            progress((k + 1) / len(items), desc=f"평가 사진 처리 {k + 1}/{len(items)}")
        meta = json.loads(jp.read_text(encoding="utf-8"))
        rgb = fc.load_rgb(jp.with_suffix(".jpg"))
        faces = fc.detect(rgb) if rgb is not None else []
        gt = meta["faces"]
        match = [-1] * len(gt)
        if faces:
            iou = np.array([[fc._iou(g["bbox"], f.bbox) for f in faces] for g in gt])
            r, c = linear_sum_assignment(-iou)
            for a, b in zip(r, c):
                if iou[a, b] >= 0.4:
                    match[a] = int(b)
        S = {}
        if faces:
            E = np.stack([f.normed_embedding for f in faces])
            for m in methods:
                S_a, S_g, _ = fc.score_matrices(E, model, m)
                S[m] = (S_a, S_g)
        photos.append({"id": jp.stem, "rgb": rgb, "gt": gt, "match": match, "S": S})

    # 2) 채점
    def score(method, t):
        rows = []  # (사진id, 얼굴번호, 정답, 예측, bbox)
        for ph in photos:
            pred_idx = fc.assign_names(*ph["S"][method], t, unique) if ph["S"] else []
            for gi, g in enumerate(ph["gt"]):
                truth = g["label"] if g["label"] in names else fc.UNKNOWN
                d = ph["match"][gi]
                pred = UNDETECTED if d < 0 else (names[pred_idx[d]] if pred_idx[d] >= 0 else fc.UNKNOWN)
                rows.append((ph["id"], gi + 1, truth, pred, g["bbox"]))
        return rows

    def stats(rows):
        tot = len(rows)
        ok = sum(t == p for _, _, t, p, _ in rows)
        mem = [(t, p) for _, _, t, p, _ in rows if t != fc.UNKNOWN]
        out = [(t, p) for _, _, t, p, _ in rows if t == fc.UNKNOWN]
        return {"acc": ok / tot if tot else 0, "total": tot, "correct": ok,
                "m_total": len(mem), "m_ok": sum(t == p for t, p in mem),
                "m_wrong": sum(p in names and p != t for t, p in mem),
                "m_unknown": sum(p == fc.UNKNOWN for t, p in mem),
                "undet": sum(r[3] == UNDETECTED for r in rows),
                "o_total": len(out), "o_ok": sum(p == fc.UNKNOWN for t, p in out),
                "o_false": sum(p in names for t, p in out)}

    ts = np.round(np.arange(0.15, 0.701, 0.01), 3)
    sweep = {m: [stats(score(m, t)) for t in ts] for m in methods}
    now = {m: score(m, th) for m in methods}
    st = {m: stats(now[m]) for m in methods}

    # 3) 리포트
    n_photo, n_face = len(photos), st["knn"]["total"]
    L = [f"## 평가 결과 — 사진 {n_photo}장, 정답 얼굴 {n_face}개 (임계값 {th:.3f})", "",
         "| 지표 | " + " | ".join(fc.METHODS[m] for m in methods) + " |", "|---|" + "---|" * len(methods)]

    def pct(a, b):
        return f"{a / b * 100:.1f}% ({a}/{b})" if b else "-"

    L.append("| **전체 정확도** | " + " | ".join(f"**{pct(st[m]['correct'], st[m]['total'])}**" for m in methods) + " |")
    L.append("| 팀원을 맞게 인식 | " + " | ".join(pct(st[m]["m_ok"], st[m]["m_total"]) for m in methods) + " |")
    L.append("| 팀원을 다른 팀원으로 착각 | " + " | ".join(pct(st[m]["m_wrong"], st[m]["m_total"]) for m in methods) + " |")
    L.append("| 팀원인데 '모름' 처리 | " + " | ".join(pct(st[m]["m_unknown"], st[m]["m_total"]) for m in methods) + " |")
    L.append("| 외부인을 '모름'으로 맞게 처리 | " + " | ".join(pct(st[m]["o_ok"], st[m]["o_total"]) for m in methods) + " |")
    L.append("| 외부인을 팀원으로 오인식 | " + " | ".join(pct(st[m]["o_false"], st[m]["o_total"]) for m in methods) + " |")
    L.append("| 얼굴 검출 실패 | " + " | ".join(pct(st[m]["undet"], st[m]["total"]) for m in methods) + " |")

    for m in methods:
        accs = np.array([s["acc"] for s in sweep[m]])
        best = ts[accs >= accs.max() - 1e-9]
        L.append(f"- {fc.METHODS[m]}: 이 평가셋에서 정확도가 가장 높은 임계값 구간 **{best.min():.2f} ~ {best.max():.2f}** → {accs.max() * 100:.1f}%")
    L.append("  - ⚠️ 평가셋에 맞춰 임계값을 고르면 점수가 부풀려집니다. 보고서에는 '학습 때 정한 임계값' 결과를 쓰세요.")

    labels = names + [fc.UNKNOWN]
    for m in methods:
        cols = labels + [UNDETECTED]
        L += ["", f"### 혼동행렬 — {fc.METHODS[m]} (행: 정답, 열: 예측)", "",
              "| 정답 \\ 예측 | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        for t in labels:
            cnt = [sum(1 for *_, tt, pp, _ in now[m] if tt == t and pp == c) for c in cols]
            if sum(cnt):
                L.append(f"| **{t}** | " + " | ".join(f"**{v}**" if c == t and v else str(v) for v, c in zip(cnt, cols)) + " |")

    hits = leakage_check()
    if hits:
        L += ["", f"### ⚠️ 데이터 누수 경고: 평가 사진이 학습 데이터에도 {len(hits)}개 들어 있습니다",
              "같은 사진으로 학습하고 시험 보면 점수가 부풀려집니다. 아래 파일을 dataset에서 지우고 다시 학습하세요."]
        L += [f"- `{h}`" for h in hits[:20]]

    # 4) 틀린 얼굴 모음 (주 방식 = 마지막 방식)
    main_m = methods[-1]
    errs = []
    rgb_of = {ph["id"]: ph["rgb"] for ph in photos}
    for pid, fi, t, p, bb in now[main_m]:
        if t != p and rgb_of[pid] is not None:
            errs.append((fc.crop_face(rgb_of[pid], bb), f"정답 {t} → 예측 {p}"))

    fig_path = fc.MODEL_DIR / "eval_threshold.png"
    _plot_sweep(ts, sweep, th, fig_path)
    return "\n".join(L), str(fig_path), errs, main_m


def _plot_sweep(ts, sweep, th, path):
    plt = fc.setup_plot_font()
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, ss in sweep.items():
        ax.plot(ts, [s["acc"] * 100 for s in ss], lw=2, label=f"전체 정확도 · {fc.METHODS[m]}")
        if ss[0]["o_total"]:
            ax.plot(ts, [s["o_false"] / s["o_total"] * 100 for s in ss], ls="--", label=f"외부인 오인식률 · {fc.METHODS[m]}")
    ax.axvline(th, color="gray", ls=":", label=f"현재 임계값 {th:.2f}")
    ax.set_xlabel("임계값")
    ax.set_ylabel("%")
    ax.set_ylim(0, 102)
    ax.set_title("임계값에 따른 정확도 변화")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
