# -*- coding: utf-8 -*-
r"""
팀원 얼굴 라벨링 · 학습 · 인식 · 평가 프로그램 (100% 로컬)

  실행:  run.bat 더블클릭  (또는 venv\Scripts\python.exe face_app.py)
  CLI :  python face_app.py train [--hidden 128] [--epochs 300]
         python face_app.py recognize 사진.jpg ... [--method knn|head]
         python face_app.py eval

  face_core.py  얼굴 검출/임베딩(사전학습), kNN, 분류 레이어(MLP) 학습, 인식, 그리기
  face_eval.py  정답 단체사진으로 정확도 평가
  face_app.py   화면(Gradio) + 명령어

  dataset/<이름>/   라벨링된 사진 (폴더 이름 = 정답)
  eval/             평가용 정답 사진
  model/            학습 결과(face_model.pkl), 학습곡선, 평가 그래프
  output/           인식 결과 이미지
"""
import os

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # 외부 통계 전송 끔

import argparse
import html
import shutil
import sys
import time
import uuid
from pathlib import Path

from PIL import Image

import face_core as fc
import face_eval as fe

EV_UNKNOWN = "모름 (외부인)"
EV_SKIP = "제외"


# ───────────────────────── 화면 꾸미기 (HTML 조각 · CSS · 테마) ─────────────────────────
def esc(s):
    return html.escape(str(s))


def card(title, value, sub=""):
    return (f'<div class="stat"><div class="k">{esc(title)}</div>'
            f'<div class="v">{esc(value)}</div><div class="s">{sub}</div></div>')


def step_header(n, title, desc, tips=(), warn=None, optional=False):
    """각 탭 맨 위의 단계 안내 카드. 사용자 입력이 들어가지 않는 고정 문구만 사용."""
    tag = '<span class="opt">선택</span>' if optional else ""
    lis = "".join(f"<li>{t}</li>" for t in tips)
    return (f'<div class="step"><div class="num">{n}</div><div class="body">'
            f'<h3>{title}{tag}</h3><p>{desc}</p>'
            + (f"<ul>{lis}</ul>" if lis else "")
            + (f'<div class="warn">{warn}</div>' if warn else "")
            + "</div></div>")


def section(title, desc=""):
    return f'<div class="sec"><b>{title}</b>' + (f"<span>{desc}</span>" if desc else "") + "</div>"


HERO = """
<div class="hero">
  <div class="hero-top">
    <span class="badge">🔒 100% 로컬 처리 · 사진이 이 PC 밖으로 나가지 않습니다</span>
  </div>
  <h1>팀원 얼굴 인식 스튜디오</h1>
  <p>셀카로 팀원을 등록하고 학습한 뒤, 단체사진에서 누가 누구인지 이름으로 찾아줍니다.</p>
  <div class="flow">
    <span><b>1</b> 데이터셋 등록</span><em>→</em>
    <span><b>2</b> 모델 학습</span><em>→</em>
    <span><b>3</b> 단체사진 인식</span><em>→</em>
    <span class="dim"><b>4</b> 결과 바로잡기</span><em>·</em>
    <span class="dim"><b>5</b> 성능 평가</span>
  </div>
</div>
"""

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }

/* 상단 배너 */
.hero { position: relative; overflow: hidden; border-radius: 22px; padding: 26px 30px 22px;
        color: #fff; background: linear-gradient(130deg, #312e81 0%, #4f46e5 45%, #7c3aed 75%, #c026d3 100%);
        box-shadow: 0 10px 30px -12px rgba(79, 70, 229, .55); }
.hero::after { content: ""; position: absolute; right: -80px; top: -90px; width: 300px; height: 300px;
               border-radius: 50%; background: radial-gradient(circle, rgba(255,255,255,.22), transparent 70%); }
.hero h1 { margin: 10px 0 6px; font-size: 30px; font-weight: 800; letter-spacing: -.02em; color: #fff; }
.hero p { margin: 0; font-size: 15px; opacity: .9; color: #fff; }
.hero .badge { display: inline-block; font-size: 12.5px; padding: 4px 12px; border-radius: 999px;
               background: rgba(255,255,255,.16); border: 1px solid rgba(255,255,255,.28); }
.hero .flow { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 18px; }
.hero .flow span { font-size: 13px; padding: 5px 12px 5px 6px; border-radius: 999px;
                   background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25); }
.hero .flow span b { display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px;
                     margin-right: 6px; border-radius: 50%; background: #fff; color: #4f46e5; font-size: 11.5px; }
.hero .flow span.dim { opacity: .75; border-style: dashed; }
.hero .flow em { font-style: normal; opacity: .6; }

/* 상태 카드 */
.stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
@media (max-width: 900px) { .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 520px) { .stats { grid-template-columns: 1fr; } .hero { padding: 20px; } .hero h1 { font-size: 24px; } }
.stat { padding: 14px 16px; border-radius: 16px; background: var(--block-background-fill);
        border: 1px solid var(--border-color-primary); box-shadow: 0 1px 2px rgba(0,0,0,.04); }
.stat .k { font-size: 12px; font-weight: 600; color: var(--body-text-color-subdued); }
.stat .v { font-size: 20px; font-weight: 800; margin-top: 2px; letter-spacing: -.01em; color: var(--body-text-color); }
.stat .s { font-size: 12.5px; line-height: 1.55; margin-top: 6px; color: var(--body-text-color-subdued); }
.chip { display: inline-flex; align-items: center; margin: 2px 4px 2px 0; padding: 2px 9px; border-radius: 999px;
        font-size: 12px; color: var(--body-text-color); background: var(--background-fill-secondary);
        border: 1px solid var(--border-color-primary); }
.chip i { width: 8px; height: 8px; margin-right: 6px; border-radius: 50%; }
.muted { color: var(--body-text-color-subdued); }
.warn-inline { margin-top: 4px; color: #d97706; font-weight: 600; }

/* 탭 */
.main-tabs > .tab-wrapper button[role="tab"], .main-tabs button[role="tab"] { font-size: 15px; font-weight: 700; padding: 10px 16px; }

/* 단계 안내 카드 */
.step { display: flex; gap: 16px; align-items: flex-start; padding: 18px 20px; border-radius: 18px;
        background: var(--background-fill-secondary); border: 1px solid var(--border-color-primary); }
.step .num { flex: none; display: flex; align-items: center; justify-content: center; width: 46px; height: 46px;
             border-radius: 14px; font-size: 21px; font-weight: 800; color: #fff;
             background: linear-gradient(135deg, #4f46e5, #a21caf); box-shadow: 0 6px 16px -6px rgba(79,70,229,.6); }
.step h3 { margin: 2px 0 4px; font-size: 18px; font-weight: 800; color: var(--body-text-color); }
.step p { margin: 0; font-size: 14px; line-height: 1.6; color: var(--body-text-color-subdued); }
.step ul { margin: 8px 0 0; padding-left: 18px; font-size: 13.5px; line-height: 1.7; color: var(--body-text-color); }
.step .opt { margin-left: 8px; padding: 1px 8px; border-radius: 999px; font-size: 11.5px; font-weight: 700;
             vertical-align: 3px; color: #7c3aed; background: rgba(124,58,237,.12); }
.step .warn { margin-top: 10px; padding: 8px 12px; border-radius: 10px; font-size: 13px; line-height: 1.55;
              color: var(--body-text-color); background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.35); }

/* 구역 제목 */
.sec { display: flex; align-items: baseline; gap: 10px; margin: 10px 0 2px; padding-top: 14px;
       border-top: 1px solid var(--border-color-primary); }
.sec b { font-size: 16px; font-weight: 800; color: var(--body-text-color); }
.sec span { font-size: 13px; color: var(--body-text-color-subdued); }
.sec.first { border-top: none; padding-top: 0; }

/* 큰 실행 버튼 */
.big-btn { min-height: 48px !important; font-size: 16px !important; font-weight: 700 !important; }
"""


def make_theme():
    import gradio as gr
    # 글꼴은 PC에 설치된 것만 사용 (외부 웹폰트 요청 없음)
    return gr.themes.Soft(
        primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", radius_size="lg",
        font=["Pretendard", "Malgun Gothic", "Apple SD Gothic Neo", "system-ui", "sans-serif"],
        font_mono=["Consolas", "ui-monospace", "monospace"],
    ).set(
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_950",
        block_radius="16px",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(0,0,0,.04)",
        button_primary_background_fill="linear-gradient(120deg, *primary_600, *secondary_600)",
        button_primary_background_fill_hover="linear-gradient(120deg, *primary_700, *secondary_700)",
        button_primary_background_fill_dark="linear-gradient(120deg, *primary_500, *secondary_500)",
        button_primary_text_color="white",
        button_primary_border_color="transparent",
        button_cancel_background_fill="#fef2f2",
        button_cancel_background_fill_hover="#fee2e2",
        button_cancel_border_color="#fecaca",
        button_cancel_text_color="#dc2626",
        button_cancel_text_color_hover="#b91c1c",
        button_cancel_background_fill_dark="rgba(220,38,38,.15)",
        button_cancel_background_fill_hover_dark="rgba(220,38,38,.25)",
        button_cancel_border_color_dark="rgba(220,38,38,.4)",
        button_cancel_text_color_dark="#fca5a5",
    )


def build_ui():
    import gradio as gr

    def dd(value=None, label="이름"):
        return gr.Dropdown(choices=fc.people(), value=value, label=label, allow_custom_value=True, interactive=True)

    def ev_dd(value=None):
        return gr.Dropdown(choices=fc.people() + [EV_UNKNOWN, EV_SKIP], value=value, label="정답", interactive=True)

    def status_html():
        """상단 상태 카드 4개: 데이터셋 / 학습 모델 / 평가셋 / 연산 장치"""
        ps = fc.people()
        m = fc.load_model()
        counts = {p: len(fc.person_images(p)) for p in ps}
        chips = "".join(f'<span class="chip"><i style="background:rgb{fc.color_of(p, m)}"></i>{esc(p)} {c}</span>'
                        for p, c in counts.items()) or '<span class="muted">1단계에서 팀원 사진을 등록하세요</span>'
        cards = [card("데이터셋", f"{len(ps)}명 · {sum(counts.values())}장", chips)]

        if m:
            c = m.get("comparison") or {}
            acc = []
            if m.get("loo_acc") is not None:
                acc.append(f"kNN {m['loo_acc'] * 100:.0f}%")
            if c.get("head_val") is not None:
                acc.append(f"MLP {c['head_val'] * 100:.0f}%")
            sub = f"{m['trained_at'][5:16]} 학습 · 임계값 {m['threshold']:.3f}"
            if dataset_changed_after_training():
                sub += '<div class="warn-inline">⚠️ 그 뒤 데이터가 바뀜 → 2단계에서 다시 학습</div>'
            cards.append(card("학습 모델 (검증 정확도)", " · ".join(acc) or f"{len(m['names'])}명", sub))
        else:
            cards.append(card("학습 모델", "아직 없음", "2단계에서 학습하세요"))

        n_ev, n_ef = fe.eval_summary()
        cards.append(card("평가셋", f"사진 {n_ev}장 · 얼굴 {n_ef}개",
                          "5단계에서 정답을 지정한 단체사진" if n_ev else "5단계에서 정답 단체사진을 저장하세요"))
        gpu = "GPU" in fc.DEVICE
        cards.append(card("연산 장치", "⚡ GPU (CUDA)" if gpu else fc.DEVICE,
                          f"얼굴 검출·특징 = {fc.DEVICE}<br>몸 윤곽 = {fc.BODY_DEVICE} · 분류 레이어 = CPU"))
        return f'<div class="stats">{"".join(cards)}</div>'

    def dataset_changed_after_training():
        """학습 이후 사진이 추가·삭제됐는지 (폴더·파일 수정 시각 비교)"""
        if not fc.MODEL_PATH.exists():
            return False
        t = fc.MODEL_PATH.stat().st_mtime
        paths = [fc.DATASET] + [fc.DATASET / p for p in fc.people()]
        paths += [f for p in fc.people() for f in fc.person_images(p)]
        return any(x.stat().st_mtime > t + 1 for x in paths)

    def pick(evt: gr.SelectData):
        return evt.index

    # ── 1. 데이터셋 등록
    def register(name, files, progress=gr.Progress()):
        name = fc.safe_name(name)
        if not name:
            raise gr.Error("이름을 입력하세요. (\\ / : * ? \" < > | 와 '모름' 사용 불가)")
        if not files:
            raise gr.Error("사진 파일을 선택하세요.")
        dst = fc.DATASET / name
        dst.mkdir(exist_ok=True)
        gallery, bad, dup = [], [], []
        known = fc.dataset_hashes()          # 이미 등록된 사진 (전체 팀원)
        in_eval = fe.eval_hashes()           # 평가용 사진
        for i, fp in enumerate(files):
            progress((i + 1) / len(files), desc=f"확인 중 {i + 1}/{len(files)}")
            src = Path(fp)
            if src.suffix.lower() not in fc.IMG_EXT:
                bad.append(f"{src.name}: 지원하지 않는 형식 (jpg/png로 변환 필요)")
                continue
            sha = fc.file_sha1(src)
            if sha in known:
                who, fname = known[sha]
                if who == name:
                    dup.append(f"{src.name}: 이미 등록된 사진 (건너뜀)")
                else:
                    bad.append(f"{src.name}: ⚠️ 이미 **{who}**(으)로 등록된 사진입니다. 라벨이 맞는지 확인하세요 ({who}/{fname})")
                continue
            if sha in in_eval:
                bad.append(f"{src.name}: 평가셋에 쓰인 사진이라 학습에 넣지 않았습니다")
                continue
            rgb = fc.load_rgb(src)
            if rgb is None:
                bad.append(f"{src.name}: 열 수 없음")
                continue
            face, reason = fc.selfie_face(rgb)
            if face is None:
                bad.append(f"{src.name}: {reason}")
                continue
            out = dst / f"{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}{src.suffix.lower()}"
            shutil.copy2(src, out)
            known[sha] = (name, out.name)    # 같은 묶음 안의 중복도 걸러냄
            gallery.append((fc.crop_face(rgb, face.bbox), out.name))
        if not any(dst.iterdir()):
            dst.rmdir()                      # 한 장도 등록 안 됐으면 빈 폴더 정리
        msg = f"✅ **{name}**: {len(gallery)}장 새로 등록 (총 {len(fc.person_images(name))}장)"
        if dup:
            msg += f"\n\n↩️ 중복이라 건너뜀 {len(dup)}장\n" + "\n".join(f"- {b}" for b in dup)
        if bad:
            msg += f"\n\n❌ 제외 {len(bad)}장\n" + "\n".join(f"- {b}" for b in bad)
        msg += "\n\n➡️ 모든 팀원 등록 후 **2. 모델 학습**에서 학습하세요."
        return msg, gallery, dd(name), dd(name, "확인할 사람"), status_html()

    def browse(name):
        name = fc.safe_name(name)
        if not name or not (fc.DATASET / name).exists():
            return [], None
        items = [(t, p.name) for p in fc.person_images(name) if (t := fc.thumbnail(p)) is not None]
        return items, None

    def delete_img(name, idx):
        name = fc.safe_name(name)
        imgs = fc.person_images(name) if name else []
        if idx is None or not (0 <= idx < len(imgs)):
            raise gr.Error("삭제할 사진을 먼저 클릭하세요.")
        imgs[idx].unlink()
        g, _ = browse(name)
        return g, None, status_html()

    def delete_person(name):
        name = fc.safe_name(name)
        if not name or not (fc.DATASET / name).exists():
            raise gr.Error("삭제할 사람을 선택하세요.")
        shutil.rmtree(fc.DATASET / name)
        return [], None, dd(), dd(None, "확인할 사람"), status_html()

    # ── 2. 모델 학습
    def do_train(hidden, epochs, progress=gr.Progress()):
        try:
            rep = fc.train(progress, hidden=int(hidden), epochs=int(epochs))
        except ValueError as e:
            raise gr.Error(str(e))
        m = fc.load_model()
        curve = str(fc.CURVE_PATH) if m.get("head") and fc.CURVE_PATH.exists() else None
        th = gr.Slider(value=round(m["threshold"], 3))
        return rep, curve, status_html(), th, th

    # ── 3. 단체사진 인식 / 4. 결과 바로잡기
    def do_recognize(img_path, th, unique, method, body):
        m = fc.load_model()
        if m is None:
            raise gr.Error("먼저 2. 모델 학습에서 학습하세요.")
        if not img_path:
            raise gr.Error("단체 사진을 올리세요.")
        if method == "head" and not m.get("head"):
            raise gr.Error("분류 레이어가 학습되지 않았습니다 (2명 이상 필요). 2. 모델 학습에서 다시 학습하세요.")
        rgb = fc.load_rgb(img_path)
        if rgb is None:
            raise gr.Error("사진을 열 수 없습니다.")
        res = fc.recognize(rgb, m, th, unique, method, body=body)
        vis = fc.draw_results(rgb, res, m)
        out = fc.OUTPUT / f"result_{time.strftime('%Y%m%d_%H%M%S')}_{method}.jpg"
        Image.fromarray(vis).save(out, quality=92)
        sc = "확률" if method == "head" else "유사도"
        rows = [f"| 이름 | {sc} | 팀원 유사도 | 2순위 후보 |", "|---|---|---|---|"]
        for r in res:
            rows.append(f"| **{r['name']}** | {r['fmt'](r['score'])} | {r['sim']:.3f} | {r['alt']} ({r['fmt'](r['alt_score'])}) |")
        found = sorted({r["name"] for r in res if r["name"] != fc.UNKNOWN})
        body_txt = f" · 몸 윤곽 연결 {sum(r['body'] is not None for r in res)}/{len(res)}명" if body else ""
        summary = (f"**{fc.METHODS[method]}** · 얼굴 {len(res)}개 검출{body_txt} · 인식: {', '.join(found) if found else '없음'}\n\n"
                   + "\n".join(rows) + f"\n\n저장됨: `output/{out.name}`")
        crops = [(fc.crop_face(rgb, r["face"].bbox), r["text"]) for r in res]
        state = {"rgb": rgb, "sha": fc.file_sha1(img_path), "faces": [(r["face"].bbox.copy(), r["name"]) for r in res]}
        return vis, summary, crops, state, None, dd(None, "정답 이름")

    def _save_crop(state, bbox, name):
        """단체사진 얼굴 1개를 학습 데이터로 저장. 같은 얼굴은 한 번만 (다른 이름으로 저장돼 있으면 고쳐 저장).
        → 'new' | 'dup' | 'moved:<이전 이름>'"""
        key = f"group_{state['sha'][:8]}_{int(bbox[0])}x{int(bbox[1])}_"   # 원본사진 + 얼굴 위치
        prev = [p for n in fc.people() for p in fc.person_images(n) if p.name.startswith(key)]
        if any(p.parent.name == name for p in prev):
            return "dup"
        for p in prev:                                   # 예전에 다른 이름으로 저장했던 같은 얼굴 → 정정
            p.unlink()
            if not any(p.parent.iterdir()):
                p.parent.rmdir()
        (fc.DATASET / name).mkdir(exist_ok=True)
        out = fc.DATASET / name / f"{key}{uuid.uuid4().hex[:6]}.jpg"
        Image.fromarray(fc.crop_face(state["rgb"], bbox, pad=1.0)).save(out, quality=95)
        return f"moved:{prev[0].parent.name}" if prev else "new"

    def _check_not_eval(state):
        if state["sha"] in fe.eval_hashes():
            raise gr.Error("이 사진은 평가셋에 들어 있어서 학습 데이터로 쓸 수 없습니다. (같은 사진으로 학습·평가하면 점수가 부풀려짐)")

    def save_label(state, idx, name):
        name = fc.safe_name(name)
        if not state:
            raise gr.Error("먼저 인식을 실행하세요.")
        if idx is None:
            raise gr.Error("얼굴 목록에서 얼굴을 클릭하세요.")
        if not name:
            raise gr.Error("정답 이름을 고르거나 입력하세요.")
        _check_not_eval(state)
        r = _save_crop(state, state["faces"][idx][0], name)
        if r == "dup":
            msg = f"↩️ 선택한 얼굴은 이미 **{name}**(으)로 저장돼 있습니다 (중복 저장 안 함)."
        elif r.startswith("moved:"):
            msg = f"✏️ 선택한 얼굴의 라벨을 **{r[6:]} → {name}**(으)로 고쳤습니다."
        else:
            msg = f"✅ 선택한 얼굴을 **{name}** 라벨로 저장했습니다."
        return (msg + " 다 고친 뒤 2. 모델 학습에서 다시 학습하세요.\n\n⚠️ 이 사진은 이제 학습 데이터이므로 평가셋에는 넣을 수 없습니다."), \
            status_html(), dd(name, "정답 이름")

    def save_all(state):
        if not state:
            raise gr.Error("먼저 인식을 실행하세요.")
        _check_not_eval(state)
        cnt = {"new": 0, "dup": 0, "moved": 0}
        for bbox, n in state["faces"]:
            if n != fc.UNKNOWN:
                cnt[_save_crop(state, bbox, n).split(":")[0]] += 1
        return (f"✅ 새로 저장 {cnt['new']}개 · 이미 있어서 건너뜀 {cnt['dup']}개 · 라벨 정정 {cnt['moved']}개. "
                "2. 모델 학습에서 다시 학습하세요."), status_html()

    # ── 5. 성능 평가: 정답 만들기
    def ev_view(st):
        res = [{"face": f, "name": (fc.UNKNOWN if l == EV_UNKNOWN else l), "text": l} for f, l in zip(st["faces"], st["labels"])]
        vis = fc.draw_results(st["rgb"], res, fc.load_model(), show_index=True)
        crops = [(fc.crop_face(st["rgb"], f.bbox), f"{i}. {l}") for i, (f, l) in enumerate(zip(st["faces"], st["labels"]), 1)]
        left = sum(l == "?" for l in st["labels"])
        msg = f"얼굴 {len(st['faces'])}개 · 정답 미지정 {left}개" + (" (미지정 얼굴은 평가에서 제외됩니다)" if left else " ✅")
        return vis, crops, msg

    def ev_detect(path):
        if not path:
            raise gr.Error("평가용 단체 사진을 올리세요.")
        rgb = fc.load_rgb(path)
        if rgb is None:
            raise gr.Error("사진을 열 수 없습니다.")
        faces = fc.detect(rgb)
        if not faces:
            raise gr.Error("얼굴을 찾지 못했습니다.")
        why = fe.eval_conflict(fc.file_sha1(path))   # 라벨 달기 전에 미리 확인
        if why:
            raise gr.Error(why)
        faces = sorted(faces, key=lambda f: f.bbox[0])  # 왼쪽부터 번호
        st = {"path": path, "rgb": rgb, "faces": faces, "labels": ["?"] * len(faces)}
        vis, crops, msg = ev_view(st)
        return st, vis, crops, msg, None, ev_dd()

    def ev_set(st, idx, label):
        if not st:
            raise gr.Error("먼저 얼굴 검출을 하세요.")
        if idx is None:
            raise gr.Error("얼굴 목록에서 얼굴을 클릭하세요.")
        if not label:
            raise gr.Error("정답을 고르세요.")
        st["labels"][idx] = label
        vis, crops, msg = ev_view(st)
        nxt = next((i for i, l in enumerate(st["labels"]) if l == "?"), None)
        hint = f" → 다음: {nxt + 1}번 얼굴" if nxt is not None else ""
        return st, vis, crops, msg + hint, nxt

    def ev_save(st):
        if not st:
            raise gr.Error("먼저 얼굴 검출을 하세요.")
        labels = [fc.UNKNOWN if l == EV_UNKNOWN else l for l in st["labels"]]
        try:
            item_id, n = fe.save_eval_item(st["path"], st["rgb"], [f.bbox for f in st["faces"]], labels)
        except ValueError as e:
            raise gr.Error(str(e))
        return f"✅ 평가셋에 저장했습니다 (정답 얼굴 {n}개, id `{item_id}`).", status_html(), None

    def ev_run(th, unique, progress=gr.Progress()):
        m = fc.load_model()
        if m is None:
            raise gr.Error("먼저 2. 모델 학습에서 학습하세요.")
        try:
            rep, fig, errs, main_m = fe.run_evaluation(m, th, unique, progress)
        except ValueError as e:
            raise gr.Error(str(e))
        return rep, fig, errs, gr.Gallery(label=f"틀린 얼굴 ({fc.METHODS[main_m]} 기준)")

    def ev_clear():
        fe.clear_eval()
        return "평가셋을 비웠습니다.", status_html()

    m0 = fc.load_model()
    th0 = round(m0["threshold"], 3) if m0 else 0.35
    th_label = "임계값 (팀원 유사도가 이보다 낮으면 '모름'. 높이면 엄격, 낮추면 관대)"
    method_choices = [(v, k) for k, v in fc.METHODS.items()]

    # delete_cache: 브라우저에서 올린 사진의 임시 복사본(%TEMP%\gradio)을 30분마다 정리, 종료 시에도 삭제
    with gr.Blocks(title="팀원 얼굴 인식 스튜디오", delete_cache=(1800, 1800)) as demo:
        gr.HTML(HERO, padding=False)
        status = gr.HTML(status_html(), padding=False)

        with gr.Tabs(elem_classes="main-tabs"):
            # ───── 1
            with gr.Tab("1. 데이터셋 등록", id=1):
                gr.HTML(padding=False, value=step_header(
                    1, "데이터셋 등록 (= 라벨링)",
                    "이름을 고르거나 입력하고, 그 사람의 사진을 여러 장 선택해 등록하세요. <b>이름(폴더 이름)이 그대로 정답 라벨</b>이 됩니다.",
                    ["한 사람당 <b>20~30장</b>, 각도·조명·표정·안경 등이 다양할수록 좋습니다",
                     "얼굴이 없거나 비슷한 크기 얼굴이 여러 개인 사진(단체사진)은 자동 제외 → 단체사진은 3·5단계에서 사용",
                     "이미 올린 사진을 다시 올려도 중복은 자동으로 건너뜁니다"]))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=2):
                        reg_name = dd()
                        reg_files = gr.File(file_count="multiple", file_types=["image"], type="filepath",
                                            label="팀원 사진 (여러 장 선택 가능)", height=210)
                        reg_btn = gr.Button("등록하기", variant="primary", elem_classes="big-btn")
                        reg_msg = gr.Markdown()
                    with gr.Column(scale=3):
                        reg_gal = gr.Gallery(label="방금 등록된 얼굴 (모두 그 사람이 맞는지 확인)", columns=6, height=400,
                                             object_fit="cover")

                gr.HTML(section("등록된 사진 관리", "사람을 고르고 사진을 클릭해 선택 → 잘못 라벨링된 사진 삭제"))
                with gr.Row(equal_height=True):
                    br_name = dd(None, "확인할 사람")
                    br_del = gr.Button("🗑 선택한 사진 삭제", variant="stop")
                    br_del_p = gr.Button("이 사람 전체 삭제", variant="stop")
                br_gal = gr.Gallery(label="등록된 사진", columns=8, height=360, object_fit="cover")
                br_idx = gr.State(None)

            # ───── 2
            with gr.Tab("2. 모델 학습", id=2):
                gr.HTML(padding=False, value=step_header(
                    2, "모델 학습",
                    "등록된 사진으로 두 가지 방식을 함께 학습하고 검증합니다. 사진을 추가·삭제했다면 반드시 다시 학습하세요.",
                    ["<b>최근접 이웃 (kNN)</b> — 팀원별 얼굴 특징(512차원)과 가장 비슷한 사람을 고름 · 사진을 1장씩 빼고 맞혀보는 LOO 검증",
                     "<b>분류 레이어 (MLP)</b> — 얼굴 특징 위에 신경망 레이어(512→128→N)를 직접 학습 · 사진 20%를 떼어 검증",
                     "얼굴 특징 추출기 ArcFace ResNet-50(4,360만 파라미터)은 사전학습된 그대로 쓰고(동결) 분류 레이어만 학습합니다. 자세한 원리는 README.md 참고"]))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=3):
                        with gr.Accordion("분류 레이어 설정 (기본값 권장)", open=False):
                            with gr.Row():
                                tr_hidden = gr.Dropdown(choices=[("없음 (선형 512→N)", 0), ("64", 64), ("128 (기본)", 128), ("256", 256)],
                                                        value=128, label="은닉층 크기")
                                tr_epochs = gr.Slider(50, 1000, value=300, step=50, label="에폭 수")
                    with gr.Column(scale=1, min_width=200):
                        tr_btn = gr.Button("🚀 학습 시작", variant="primary", elem_classes="big-btn")
                tr_curve = gr.Image(label="분류 레이어 학습 곡선 (마지막 학습)", type="filepath", height=360,
                                    value=str(fc.CURVE_PATH) if fc.CURVE_PATH.exists() else None)
                tr_rep = gr.Markdown()

            # ───── 3
            with gr.Tab("3. 단체사진 인식", id=3):
                gr.HTML(padding=False, value=step_header(
                    3, "단체사진 인식",
                    "새 단체사진을 올리면 얼굴마다 이름과 점수를 표시하고, 몸 전체를 사람별 색으로 칠합니다. 보기만 하면 아무것도 저장되지 않습니다.",
                    ["점수 읽는 법: kNN은 <b>유사도</b>(0~1, 0.5 이상이면 확실), MLP는 <b>확률</b>(%)",
                     "팀원 유사도가 임계값보다 낮으면 <b>모름</b>(외부인)으로 표시"]))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        rc_in = gr.Image(type="filepath", label="단체 사진", height=280)
                        rc_method = gr.Radio(method_choices, value="knn", label="인식 방식")
                        with gr.Accordion("세부 설정", open=False):
                            rc_th = gr.Slider(0.1, 0.8, value=th0, step=0.005, label=th_label)
                            rc_uni = gr.Checkbox(value=True, label="한 사진에 같은 사람은 한 번만 등장")
                            rc_body = gr.Checkbox(value=fc.body_available(), interactive=fc.body_available(),
                                                  label="몸 전체 윤곽 표시 (Mask R-CNN)" + ("" if fc.body_available() else " — PyTorch 미설치"))
                        rc_btn = gr.Button("🔍 인식하기", variant="primary", elem_classes="big-btn")
                    with gr.Column(scale=2):
                        rc_out = gr.Image(label="결과", height=560, interactive=False)
                rc_md = gr.Markdown()
                rc_state = gr.State(None)
                rc_idx = gr.State(None)

            # ───── 4
            with gr.Tab("4. (선택) 결과 바로잡기", id=4):
                gr.HTML(padding=False, value=step_header(
                    4, "결과 바로잡기 (추가 라벨링)",
                    "3단계에서 인식한 단체사진의 얼굴을 학습 데이터로 추가합니다. 셀카와 단체사진은 조명·각도·해상도가 달라서, "
                    "단체사진 얼굴을 넣으면 단체사진 정확도가 올라갑니다. 저장한 뒤 <b>2. 모델 학습</b>에서 다시 학습하세요.",
                    ["언제 쓰나요? 이름이 틀렸을 때 · 점수가 임계값 근처로 낮을 때 · 특정 팀원이 자주 '모름'으로 나올 때"],
                    warn="⚠️ 여기서 저장한 사진은 학습 데이터가 되므로 <b>5. 성능 평가</b>에는 쓸 수 없습니다. "
                         "단체사진이 적다면 이 단계는 건너뛰고 모두 평가용으로 쓰세요.",
                    optional=True))
                rc_gal = gr.Gallery(label="① 얼굴 클릭 (3단계에서 마지막으로 인식한 사진)", columns=10, height=230, object_fit="cover")
                with gr.Row(equal_height=True):
                    fix_name = dd(None, "② 정답 이름")
                    fix_btn = gr.Button("③ 선택한 얼굴을 이 이름으로 저장", variant="primary")
                    all_btn = gr.Button("결과가 전부 맞음 → 모두 저장")
                fix_msg = gr.Markdown()

            # ───── 5
            with gr.Tab("5. 성능 평가", id=5):
                gr.HTML(padding=False, value=step_header(
                    5, "성능 평가",
                    "정답을 미리 지정해 둔 단체사진으로 두 방식(kNN · MLP)을 자동 채점합니다. 발표·보고서용 정확도는 여기서 얻습니다.",
                    ["<b>학습에 쓰지 않은</b> 단체사진만 사용하세요 (학습에 쓴 사진은 자동으로 막힙니다)",
                     "팀원이 아닌 사람은 <b>모름 (외부인)</b>, 너무 작거나 흐린 얼굴은 <b>제외</b>",
                     "같은 사진을 다시 저장하면 이전 정답을 덮어씁니다"],
                    optional=True))
                gr.HTML(section("① 정답 단체사진 만들기", "올리기 → 얼굴 검출 → 얼굴 클릭 → 정답 지정 → 저장 (이 화면에서만 얼굴 번호 표시)"))
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=300):
                        ev_in = gr.Image(type="filepath", label="평가용 단체 사진", height=260)
                        ev_det = gr.Button("얼굴 검출")
                        ev_label = ev_dd()
                        ev_setbtn = gr.Button("선택한 얼굴에 정답 지정")
                        ev_savebtn = gr.Button("💾 평가셋에 저장", variant="primary")
                        ev_msg = gr.Markdown()
                    with gr.Column(scale=2):
                        ev_vis = gr.Image(label="번호 확인", height=440, interactive=False)
                        ev_gal = gr.Gallery(label="얼굴 클릭으로 선택", columns=10, height=170, object_fit="cover")
                ev_state = gr.State(None)
                ev_idx = gr.State(None)

                gr.HTML(section("② 평가 실행", "저장된 평가셋 전체로 두 방식 비교"))
                with gr.Row(equal_height=True):
                    ev_th = gr.Slider(0.1, 0.8, value=th0, step=0.005, label=th_label, scale=3)
                    ev_uni = gr.Checkbox(value=True, label="한 사진에 같은 사람은 한 번만 등장", scale=1)
                with gr.Row():
                    ev_run_btn = gr.Button("📊 평가 실행", variant="primary", elem_classes="big-btn", scale=3)
                    ev_clear_btn = gr.Button("평가셋 전체 삭제", variant="stop", scale=1)
                ev_rep = gr.Markdown()
                with gr.Row(equal_height=False):
                    ev_fig = gr.Image(label="임계값에 따른 정확도", type="filepath")
                    ev_err = gr.Gallery(label="틀린 얼굴", columns=5, height=300, object_fit="cover")

        demo.load(status_html, None, status, show_progress="hidden")   # 새로고침할 때 상태 카드 최신화
        reg_btn.click(register, [reg_name, reg_files], [reg_msg, reg_gal, reg_name, br_name, status])
        br_name.change(browse, br_name, [br_gal, br_idx])
        br_gal.select(pick, None, br_idx)
        br_del.click(delete_img, [br_name, br_idx], [br_gal, br_idx, status])
        br_del_p.click(delete_person, br_name, [br_gal, br_idx, reg_name, br_name, status])
        tr_btn.click(do_train, [tr_hidden, tr_epochs], [tr_rep, tr_curve, status, rc_th, ev_th])
        rc_btn.click(do_recognize, [rc_in, rc_th, rc_uni, rc_method, rc_body], [rc_out, rc_md, rc_gal, rc_state, rc_idx, fix_name])
        rc_gal.select(pick, None, rc_idx)
        fix_btn.click(save_label, [rc_state, rc_idx, fix_name], [fix_msg, status, fix_name])
        all_btn.click(save_all, rc_state, [fix_msg, status])
        ev_det.click(ev_detect, ev_in, [ev_state, ev_vis, ev_gal, ev_msg, ev_idx, ev_label])
        ev_gal.select(pick, None, ev_idx)
        ev_setbtn.click(ev_set, [ev_state, ev_idx, ev_label], [ev_state, ev_vis, ev_gal, ev_msg, ev_idx])
        ev_savebtn.click(ev_save, ev_state, [ev_msg, status, ev_state])
        ev_run_btn.click(ev_run, [ev_th, ev_uni], [ev_rep, ev_fig, ev_err, ev_err])
        ev_clear_btn.click(ev_clear, None, [ev_rep, status])
    return demo


def main():
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="팀원 얼굴 인식")
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="dataset 폴더로 학습 (kNN + 분류 레이어)")
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--epochs", type=int, default=300)
    r = sub.add_parser("recognize", help="사진 인식 → output 폴더에 저장")
    r.add_argument("images", nargs="+")
    r.add_argument("--threshold", type=float, default=None)
    r.add_argument("--method", choices=list(fc.METHODS), default="knn")
    r.add_argument("--no-body", action="store_true", help="몸 전체 윤곽(Mask R-CNN) 끄기")
    e = sub.add_parser("eval", help="평가셋으로 두 방식 정확도 비교")
    e.add_argument("--threshold", type=float, default=None)
    a = ap.parse_args()

    if a.cmd == "train":
        print(fc.train(hidden=a.hidden, epochs=a.epochs))
        print(f"\n[연산 장치] 얼굴 검출·특징 추출: {fc.DEVICE}")
    elif a.cmd in ("recognize", "eval"):
        m = fc.load_model()
        if m is None:
            sys.exit("모델이 없습니다. 먼저 train 하세요.")
        if a.cmd == "eval":
            rep, fig, _, _ = fe.run_evaluation(m, a.threshold)
            print(rep + f"\n\n그래프: {fig}")
            return
        for p in a.images:
            rgb = fc.load_rgb(p)
            if rgb is None:
                print(f"{p}: 열 수 없음")
                continue
            res = fc.recognize(rgb, m, a.threshold, True, a.method, body=not a.no_body)
            out = fc.OUTPUT / f"result_{Path(p).stem}_{a.method}.jpg"
            Image.fromarray(fc.draw_results(rgb, res, m)).save(out, quality=92)
            print(f"{p}: " + ", ".join(r["text"] for r in res) + f" → {out}")
    else:
        fc.engine()  # 모델 미리 로드
        if fc.body_available():
            try:
                fc.body_model()
            except Exception as e:  # 가중치 다운로드 실패 등 → 몸 윤곽만 끄고 계속
                print(f"[경고] Mask R-CNN 로드 실패: {e}")
        build_ui().launch(inbrowser=True, server_name="127.0.0.1", theme=make_theme(), css=CSS)


if __name__ == "__main__":
    main()
