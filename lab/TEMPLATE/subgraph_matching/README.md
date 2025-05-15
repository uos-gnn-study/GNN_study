# 🧪 Subgraph Matching 실습 템플릿

이 저장소는 **Graph Neural Network (GNN)** 기반 실험을 위한  
`Subgraph Matching` 데이터셋 생성 및 실험 코드 템플릿입니다.

스터디 또는 프로젝트에서 PyG 실습을 빠르게 시작할 수 있도록  
필수 구성 요소만 모아두었습니다.

---

## 📦 구성

```
.
├── conf
│   └── data.yaml
├── db.sh
├── pyscripts
│   ├── __init__.py
│   ├── data.py
│   ├── graph.py
├── setup.sh
└── utils
    ├── __init__.py
    ├── configure.py
    ├── prepare_data.py
    └── validate_data.py
```


---

## ⚙️ 설치 및 초기화

실험용 폴더(ex. `lab/test`)에서 다음을 실행하면,  
해당 폴더에 필요한 코드만 복사됩니다.

```bash
bash ../TEMPLATE/subgraph_matching/setup.sh
```
---

## 🚀 빠른 시작
복사 완료 후, 명령어:
```bash
bash db.sh
```
를 실행하시면 자동으로 데이터가 생성됩니다. 데이터의 구성은 `conf/data.yaml`를 수정하여 변경할 수 있습니다.

---

## ⚠️ 주의사항
`PyG`, `PyTorch` 등 환경의 설치가 필요합니다. 제가 사용한 버전은
```txt
PyYAML                   6.0.2
networkx                 3.4.2
numpy                    2.2.5
torch                    2.6.0
torch-geometric          2.6.1
torch_scatter            2.1.2+pt26cu124
```
입니다. `requirements.txt`에 명시되어 있으니 명령어:
```
pip install -r requirements.txt
```
로 설치할 수 있습니다.
⚠️ `torch-scatter`는 `PyTorch`와 `CUDA` 버전에 맞춰 설치해야 하므로,
[공식 설치 가이드](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) 참고 권장