# Titanic 특성 공학 실험

Python 3.12 환경에서 실행하였습니다.

```bash
python -m pip install -r requirements.txt
python analysis.py --out results
```

기본 입력은 같은 폴더의 `titanic.csv`입니다. 동일한 Kaggle 학습 데이터를
별도로 사용하려면 `python analysis.py --data train.csv --out results`를 실행합니다.
교차검증 및 테스트 분할에 난수 시드 42를 사용합니다.

- `comparison.csv`: 모델·실험별 검증 및 테스트 성능
- `selection.csv`: 변환된 열의 ANOVA F 점수와 선택 여부
- `grid.csv`: 검증 기준으로 선정한 실험의 하이퍼파라미터 탐색 결과
- `eda.png`, `missing.csv`, `outliers.csv`, `by_sex.csv`: EDA 산출물
- `importance.png`: 테스트 세트에서 원본 변수 단위 순열 중요도

데이터 출처: [Kaggle Titanic](https://www.kaggle.com/competitions/titanic/data)
및 [Data Science Dojo CSV 미러](https://github.com/datasciencedojo/datasets/blob/master/titanic.csv).
번들 CSV의 SHA-256은
`4a437fde05fe5264e1701a7387ac6fb75393772ba38bb2c9c566405af5af4bd7`입니다.