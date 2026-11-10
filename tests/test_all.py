# -*- coding: utf-8 -*-
"""
전체 기능 테스트 (화면의 모든 버튼 동작을 코드로 직접 호출)

  실행:  venv\\Scripts\\python.exe tests\\test_all.py

- 실제 dataset/model/eval/output 폴더는 건드리지 않습니다. 임시 폴더에서만 동작합니다.
- 팀원 사진 없이 InsightFace에 포함된 공개 샘플 사진(t1.jpg, 6명)으로
  가짜 팀원 3명(가/나/다)의 셀카를 만들어 테스트합니다.
- 모두 통과하면 종료 코드 0, 하나라도 실패하면 1.
"""
import os
import pathlib
import re
import sys
import tempfile
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import insightface  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402

import face_core as fc  # noqa: E402
import face_eval as fe  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ✅ " if cond else "  ❌ ") + msg)
    if not cond:
        FAILS.append(msg)


def expect_error(fn, *a, **k):
    import gradio as gr
    try:
        fn(*a, **k)
    except gr.Error as e:
        return str(e)
    return None


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="face_test_"))
    fc.DATASET, fc.MODEL_DIR, fc.OUTPUT, fc.EVAL_DIR = tmp / "dataset", tmp / "model", tmp / "output", tmp / "eval"
    fc.MODEL_PATH, fc.CACHE_PATH, fc.CURVE_PATH = tmp / "model/m.pkl", tmp / "model/c.pkl", tmp / "model/curve.png"
    fc.ensure_dirs()
    print(f"임시 폴더: {tmp}")

    # ── 테스트 데이터: 공개 샘플 단체사진에서 왼쪽 3명을 잘라 밝기·각도를 바꾼 '셀카' 6장씩
    t1 = pathlib.Path(insightface.__file__).parent / "data/images/t1.jpg"
    rgb = fc.load_rgb(t1)
    faces = sorted(fc.detect(rgb), key=lambda f: f.bbox[0])
    src = tmp / "src"
    src.mkdir()
    files = {}
    for k, name in enumerate(["가", "나", "다"]):
        crop = Image.fromarray(fc.crop_face(rgb, faces[k].bbox, pad=1.0))
        files[name] = []
        for j in range(6):
            p = src / f"{k}_{j}.jpg"
            ImageEnhance.Brightness(crop).enhance(0.6 + 0.1 * j).rotate(2 * j - 5).save(p)
            files[name].append(str(p))
    ev_src = src / "group_eval.jpg"                      # 평가용: 같은 장면이지만 다른 파일
    ImageEnhance.Brightness(Image.fromarray(rgb)).enhance(0.8).save(ev_src)

    import face_app
    demo = face_app.build_ui()
    H = {f.name: f.fn for f in demo.fns.values()}       # 화면 버튼에 연결된 함수들
    P = lambda *a, **k: None                            # noqa: E731  진행바 대신

    print("\n[0] 얼굴 엔진")
    check(len(faces) == 6, f"샘플 사진 얼굴 검출 {len(faces)}/6 · 장치 {fc.DEVICE}")
    check(faces[0].normed_embedding.shape == (512,), "얼굴 특징 512차원")

    print("\n[1] 데이터셋 등록")
    for n in files:
        H["register"](n, files[n], progress=P)
    check({n: len(fc.person_images(n)) for n in fc.people()} == {"가": 6, "나": 6, "다": 6}, "3명 × 6장 등록")
    msg = H["register"]("가", files["가"], progress=P)[0]
    check("0장 새로 등록" in msg and "중복" in msg, "같은 사진 재등록 → 중복 건너뜀")
    msg = H["register"]("나", files["가"][:1], progress=P)[0]
    check("이미 **가**" in msg, "다른 사람으로 이미 등록된 사진 → 경고 후 제외")
    msg = H["register"]("라", [str(t1)], progress=P)[0]
    check("비슷한 크기 얼굴" in msg and "라" not in fc.people(), "단체사진을 셀카로 등록 → 제외, 빈 폴더 정리")
    g, _ = H["browse"]("가")
    check(len(g) == 6 and max(g[0][0].shape[:2]) <= 256, "등록 사진 목록(썸네일)")
    gone = fc.file_sha1(fc.person_images("가")[0])
    gone_src = next(f for f in files["가"] if fc.file_sha1(f) == gone)
    H["delete_img"]("가", 0)
    check(len(fc.person_images("가")) == 5, "선택한 사진 삭제")
    check(expect_error(H["delete_img"], "가", None) is not None, "선택 없이 삭제 → 오류 안내")
    H["register"]("라", [gone_src], progress=P)
    check("라" in fc.people(), "삭제한 사진은 다시 등록 가능")
    H["delete_person"]("라")
    check("라" not in fc.people(), "사람 전체 삭제")

    print("\n[2] 모델 학습")
    rep, curve, status, *_ = H["do_train"](128, 150, progress=P)
    check("학습 완료" in rep and curve and pathlib.Path(curve).exists(), "학습 리포트 + 학습 곡선")
    m = fc.load_model()
    check(m["head"] is not None and m["loo_acc"] >= 0.9, f"kNN 자기검증 {m['loo_acc'] * 100:.0f}% · 분류 레이어 생성")
    check("데이터가 바뀜" not in status, "상태 카드: 학습 직후 변경 경고 없음")
    time.sleep(1.2)
    H["delete_img"]("다", 0)
    check("데이터가 바뀜" in H["status_html"](), "학습 후 사진 삭제 → '다시 학습' 경고")
    H["do_train"](128, 150, progress=P)

    print("\n[3] 단체사진 인식")
    vis, md, crops, state, *_ = H["do_recognize"](str(t1), None, True, "knn", fc.body_available())
    by_x = sorted(fc.recognize(rgb, fc.load_model()), key=lambda r: r["face"].bbox[0])
    names = [r["name"] for r in by_x]
    check(names[:3] == ["가", "나", "다"] and set(names[3:]) == {fc.UNKNOWN}, f"kNN: {names}")
    check(vis.shape == rgb.shape and len(crops) == 6 and not re.search(r"\| \d+ \|", md), "결과 이미지 · 얼굴 목록 · 번호 없는 표")
    _, md2, *_ = H["do_recognize"](str(t1), None, True, "head", False)
    check("%" in md2, "MLP: 확률(%) 표시")
    if fc.body_available():
        res = fc.recognize(rgb, fc.load_model(), body=True)
        check(sum(r["body"] is not None for r in res) >= 4, f"몸 윤곽 연결 {sum(r['body'] is not None for r in res)}/6")

    print("\n[4] 결과 바로잡기")
    msg = H["save_label"](state, 0, "가")[0]
    check("저장" in msg or "고쳤" in msg or "이미" in msg, "선택한 얼굴 저장")
    check("이미" in H["save_label"](state, 0, "가")[0], "같은 얼굴 다시 저장 → 중복 안 함")
    msg = H["save_all"](state)[0]
    check("새로 저장" in msg, "결과 전부 저장")

    print("\n[5] 성능 평가")
    st, _, _, emsg, *_ = H["ev_detect"](str(ev_src))
    check(len(st["faces"]) == 6, f"평가 사진 얼굴 검출 ({emsg})")
    for i, lab in enumerate(["가", "나", "다", face_app.EV_UNKNOWN, face_app.EV_UNKNOWN, face_app.EV_SKIP]):
        st, _, _, emsg, _ = H["ev_set"](st, i, lab)
    check("미지정 0개" in emsg, "모든 얼굴 정답 지정")
    msg = H["ev_save"](st)[0]
    check("정답 얼굴 5개" in msg, "평가셋 저장 ('제외' 1개는 빠짐)")
    check("이미 평가셋" in (expect_error(H["ev_detect"], str(ev_src)) or ""), "같은 사진 → 이미 평가셋에 있다고 안내")
    check("학습" in (expect_error(H["ev_detect"], str(t1)) or ""), "학습에 쓴 단체사진 → 평가 차단 (데이터 누수 방지)")
    check(expect_error(H["save_all"], {"sha": fc.file_sha1(ev_src), "faces": [], "rgb": rgb}) is not None,
          "평가셋 사진 → 학습 데이터 저장 차단")
    rep, fig, errs, _ = H["ev_run"](None, True, progress=P)
    acc = re.findall(r"\*\*전체 정확도\*\* \| \*\*([\d.]+)%", rep)
    check(acc and float(acc[0]) >= 80 and pathlib.Path(fig).exists(), f"평가 실행 · kNN 정확도 {acc[0] if acc else '?'}% · 그래프")
    check(fe.leakage_check() == [], "데이터 누수 없음")
    H["ev_clear"]()
    check(fe.eval_summary() == (0, 0), "평가셋 전체 삭제")

    print("\n[6] MLP 기울기 검사 (직접 구현한 역전파가 수치 미분과 일치하는지)")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, fc.EMB_DIM)).astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array([0, 1, 2, 0, 1, 2, 0, 1])
    Y = np.eye(3)[y]
    p = {k: v.astype(np.float64) for k, v in fc.head_init(3, 16).items()}
    w = np.ones(len(y))
    _, g = fc.head_loss_grad(p, X, Y, w, 1e-3, 0.0, None)
    err = 0.0
    for k in p:
        for idx in [tuple(rng.integers(0, s) for s in p[k].shape) for _ in range(5)]:
            old = p[k][idx]
            p[k][idx] = old + 1e-5
            lp, _ = fc.head_loss_grad(p, X, Y, w, 1e-3, 0.0, None)
            p[k][idx] = old - 1e-5
            lm, _ = fc.head_loss_grad(p, X, Y, w, 1e-3, 0.0, None)
            p[k][idx] = old
            err = max(err, abs((lp - lm) / 2e-5 - g[k][idx]))
    check(err < 1e-6, f"최대 오차 {err:.1e}")

    print("\n" + ("🎉 모든 테스트 통과" if not FAILS else f"❌ 실패 {len(FAILS)}개: {FAILS}"))
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)             # 임시 폴더 정리
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
