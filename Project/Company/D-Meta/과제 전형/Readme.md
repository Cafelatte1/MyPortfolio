## D-Meta.ai 과제 전형

D-Meta.ai 과제 전형 김영준 지원자 Github 페이지 입니다.

과제에 대한 솔루션 설명 자료는 [솔루션자료_디메타 과제 전형.pdf](https://github.com/Cafelatte1/MyPortfolio/blob/main/Project/Company/D-Meta/%EA%B3%BC%EC%A0%9C%20%EC%A0%84%ED%98%95/%EC%86%94%EB%A3%A8%EC%85%98%EC%9E%90%EB%A3%8C_%EB%94%94%EB%A9%94%ED%83%80%20%EA%B3%BC%EC%A0%9C%20%EC%A0%84%ED%98%95.pdf) 파일이니 먼저 참고 부탁드립니다.

소스코드는 'sources' 폴더 안에 있습니다.

이미지는 꼭 실행하는 스크립트와 같은 depth 내 'data' 폴더 안에 있어야 합니다.

(시간 상 세부적인 Hyper-Parameter를 조정하는 기능은 구현하지 않았습니다.)

```
root
  ㄴcut_image.py
  ㄴdata\
       ㄴ image1.jpg
```

## Requirements

```
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install scikit-learn==1.1.3
pip install opencv-python
pip install scikit-image
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cpu
pip install optuna
```


## 소스 실행 Code Snipet

```
# 1. cut_image.py
# format
python cut_image.py ${input_file_name} ${M} ${N} ${prefix} ${seed}
# example
python .\cut_image.py yerin2.jpg 2 2 test 42

# 2. merge_image.py
# format
python merge_image.py ${input_file_name} ${M} ${N} ${output_file_path} ${seed}
# example
python .\merge_image.py yerin2.jpg 2 2 test_merge.jpg 42
```
