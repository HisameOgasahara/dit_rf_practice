# 2026-10-12 · 생성 텍스트 마스크 정렬과 ComfyUI LoRA 변환

학습한 LoRA를 이 리포의 노트북뿐 아니라 ComfyUI에서도 생성에 쓸 수 있다는 점을 전제로 정리한 수정이다.

## 1. 생성 쪽 텍스트 조건을 학습 캐시와 같은 방식으로 맞춤

### 문제

같은 프롬프트를 넣어도 학습과 생성에서 LLMAdapter가 만드는 조건 벡터가 달랐다.

| | 학습 캐시 (`training/dataset.py`) | 기존 생성 (`runtime.py`) |
|---|---|---|
| Qwen 패딩 위치 출력 | 0으로 지움 | 그대로 둠 |
| adapter의 Qwen 마스크 | 전달함: 실제 토큰만 참조 | 전달하지 않음: 패딩 포함 512개를 모두 참조 |

빈 프롬프트에서 차이가 가장 크다. Qwen은 BOS를 붙이지 않아 `""`의 실제 토큰이 0개다. 학습의 caption dropout 조건은 전부 마스크되지만, 생성의 negative `""`는 패딩 512개를 그대로 읽었다. CFG의 기준점인 uncond가 학습과 달랐던 셈이다.

### 원인

참조한 두 프로젝트는 각자 내부에서는 일관된다. 생성은 DiffSynth `AnimaImagePipeline`(마스크 없음)에서, 학습 캐시는 anima-lora `AnimaTextEncodingStrategy`(마스크 사용)에서 옮겨 오면서 두 방식이 섞였다.

### 변경

anima-lora 방식으로 통일했다.

- `encode_prompt`: Qwen 패딩 출력을 항상 0으로 만든다. `return_attention_masks=True`면 Qwen·T5 마스크를 함께 반환한다. 이제 토큰 가중치와도 함께 쓸 수 있다.
  - 반환 순서: `(Qwen 출력, T5 ids[, T5 가중치][, Qwen mask, T5 mask])`
- `adapt_conditioning`: `qwen_mask`, `t5_mask`를 받아 adapter에 전달한다. 마스크 없이 호출하면 경고를 남긴다.
- 생성 노트북 3개(`anima_minimal_colab`, sampler 비교 노트북 2개): 마스크를 저장하고 전달하도록 수정했다. 마스크가 없는 기존 `anima_minimal_colab` conditioning 파일은 이전 방식으로 동작한다.
- 테스트: 생성 경로의 adapter 출력이 학습 캐시 경로와 같은지 확인하는 테스트를 추가했다.

### 왜 anima-lora 쪽인가

- ComfyUI는 Qwen을 패딩하지 않는다(`pad_to_max_length=False`). 그래서 adapter가 실제 토큰만 보게 되고, 마스크를 쓴 결과와 같다.
- 학습·이 리포의 생성·ComfyUI 생성이 같은 조건을 쓰게 된다.
- 학습 캐시는 바뀌지 않으므로 기존 캐시와 이미 학습한 LoRA는 다시 만들 필요가 없다.
- 반대로 README의 기존 예시 이미지는 이전 생성 방식으로 만든 것이다. 같은 시드여도 결과가 달라질 수 있다.

## 2. 학습 LoRA를 ComfyUI 형식으로 변환

### 문제

학습 결과 `lora.safetensors`의 키는 `blocks.0.self_attn.q_proj.lora_A.weight`처럼 접두어가 없다. ComfyUI는 `diffusion_model.` 또는 `lora_unet_` 접두어가 붙은 키만 모델 레이어에 연결한다. 그래서 이 파일은 ComfyUI에서 적용되지 않는다.

### 변경

- `training/lora.py`에 `convert_to_comfyui`를 추가했다. anima-lora가 내보내는 kohya 키로 이름만 바꾼다.
  - `lora_unet_blocks_0_self_attn_q_proj.lora_down.weight` / `.lora_up.weight` / `.alpha`
  - 모듈 경로가 ComfyUI Anima(`comfy/ldm/cosmos/predict2.py`)와 같아서 이름 변환만으로 충분하다.
  - 값과 dtype은 원본 그대로 둔다(학습 LoRA는 FP32).
- LoRA 학습 노트북의 Drive 보관 셀 앞에 `10. ComfyUI 형식 LoRA 변환` 셀을 추가했다.
  - 경로를 비우면 최신 step의 LoRA를 변환한다.
  - 결과 `lora_comfyui.safetensors`는 같은 step 폴더에 저장되어 Drive 보관에 함께 포함된다.
- 변환한 파일은 이 리포의 `fuse_lora`로도 읽힌다. 테스트에서 원본과 같은 결과가 나오는지 확인했다.

## 3. 생성 쪽 추가 정리

- **denoise 고정**: 입력 이미지 없이 순수 노이즈에서 시작하는데 `denoise < 1`을 허용하고 있었다. 이 경우 σ<1부터 풀어서 노이즈가 덜 지워진 이미지가 나온다. img2img가 없으므로 `sample_latents`는 이제 1.0만 받고, 노트북의 `DENOISE` 입력은 없앴다. 부분 denoise schedule 자체는 sampler 단위(`run_sampler`)에 남겨 두었다.
- **CFG 배치 호출**: positive·negative를 배치 2로 묶어 DiT를 한 스텝에 한 번만 호출한다. 결과 식은 같고 호출 오버헤드가 줄어든다. GPU에서는 배치 연산이라 기존 이미지와 비트 단위로 같지는 않을 수 있다. 두 조건의 길이가 다르면 예전처럼 따로 호출한다.

## 4. LoRA 훈련 쪽 정리

- **timestep FP32 전달**: 노이즈는 FP32 σ로 섞으면서 모델에는 bf16으로 반올림한 σ(상대오차 최대 약 0.4%)를 넣고 있었다. 이제 같은 FP32 σ를 넘긴다. DiT의 시간 임베딩은 원래 FP32로 계산한다.
- **fp16 overflow 재시도**: overflow로 업데이트를 건너뛸 때, 그 업데이트에 쓴 샘플(기본 4장)이 그 epoch에서 버려지고 있었다. 이제 스트림 위치를 되돌려 같은 샘플로 다시 계산한다. 연속 32번을 넘으면 발산으로 보고 중단한다.
  - 이 동작은 최초 학습 코드부터 있었다. 전처리 크롭 수정(`004942c`)이나 크기별 배치의 중복 샘플 문제를 고친 전역 셔플(`0e4fbbc`)과는 관계가 없다.
  - bf16 학습에서는 GradScaler를 쓰지 않으므로 해당하지 않는다.
- **설정 주석·문서 정정**
  - 예전 주석 "작은 원본은 확대하지 않습니다"와 달리, 실제 전처리는 작은 원본도 목표 면적까지 확대한다. 예: 200×300 → 416×624.
  - `batch_size`가 한 번에 GPU에 올리는 배치가 아니라 누적 단위라는 점을 TOML 주석과 `notebook_usage.md`에 적었다.
  - `notebook_usage.md`에 남아 있던 크기별 배치 설명도 현재 방식(전역 셔플·한 장씩 누적)으로 고쳤다.
  - 이름을 바꾸면 기존 TOML과 체크포인트 재개가 깨지므로 `batch_size` 이름은 유지했다.

## 확인 범위

- 로컬 CPU 단위 테스트로 계산 경로만 확인했다(전체 통과).
- 실사용 환경인 Colab에서 확인할 항목:
  - 생성 노트북 실행
  - 수정 전후 같은 시드 비교
  - LoRA 노트북 10번 셀로 변환한 파일을 ComfyUI에서 로드
  - Muon 학습: 로컬 PyTorch 2.8에는 Muon이 없어 skip됨
