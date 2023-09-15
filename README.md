# 파이썬을 활용한 베이지안 통계 2판 - 앨런 다우니 저

### 중요사항
1. 분포에 기반한 베이지안 통계를 코드로 구현할 수 있도록 다시 학습
2. 책에 있는 코드에 더하여 가지고 있는 베이지안 통계 지식 조합
3. 변수명은 Snake Case 사용

### Chapter.1 - Probability
1. 확률의 정의(빈도론적, 주관적)와 확률의 Axiom 고려

### Chapter.2 - Bayes's Theorum
1. Diachronic Bayes → Bayesian Update : 데이터 일부 D가 주어졌을 때, 가설 H의 확률을 갱신
    - P(H|D) = P(H)P(D|H) / P(D), P(D) = ∑P(Hi)P(D|Hi)
2. Divide and Conquer 전략 → 가설과 데이터 정리, 사전확률 도출, 각 가설 하에서의 가능도 도출

### Chapter.3 - Distributions
1. Pmf 클래스는 pandas의 시리즈를 상속받는 형태
2. MAP추정량(Maximum a Posteriori) : 사후확률밀도함수를 최대화하는 파라미터를 추정값으로
    - prior가 균등분포면 MAP추정량은 MLE추정량과 동일
    - max_prob() 메서드를 통해 추정 가능

### Chapter.4 - Estimating Proportions
1. prob_ge(k), prob_le(k) 메서드 → CDF(P(X <= k))를 파악하는데 활용 가능
2. np.allclose() 함수 → 부동소수점수의 차이를 무시하고 같은지 여부를 Boolean으로 반환

### Chapter.5 - Estimating Counts
1. HPD Credible Set과 Equal tail Credible Set
    - Pmf 객체의 메서드는 아마도 Equal tail을 활용하는듯?
2. quantile() 메서드와 credible_interval() 메서드 → Credible Interval 구하는데 활용

### Chapter.6 - Odds and Addends
1. Bayes Factor → Likelihood Ratio == Posterior Odds / Prior Odds
2. add_dist() 함수 혹은 메서드 → Addends, 즉 합의 분포를 표현

### Chapter.7 - Minimum, Maximum and Mixture
1. make_cdf(), make_pmf() 메서드를 통해 Pmf 객체를 PMF와 CDF로 서로 변경 가능
    - quantile() 메서드를 체인하면서 분위 수 파악 가능, credible_interval() 메서드도 가능
2. max_dist(n) 메서드 n개의 확률변수를 고른 후 그 중 최대값의 분포를 표시 ↔ min_dist(n) 메서드

### Chapter.8 - Poisson Processes
1. ss.gamma()에 대한 정확한 이해 필요 → 내가 알고있는 감마분포랑 연결이 잘 안되는데..
    - shape parameter가 1일 때는 지수분포가 되어야되는건데, 이걸 어떻게 맞추는지 확인
    - ss.expon()에도 'scale'이라는 파라미터를 넣어줘야!
    - 책에 있는 코드를 따라가기보다 내가 가지고 있는 지식과 연계하여 진행
2. 포아송 분포의 conjugate prior는 감마분포 → 이를 활용하는 방식 고찰
3. prob_gt(), prob_lt() 함수 등 활용하여 확률 비교

### Chapter.9 - Decision Analysis
1. ss.gaussian_kde(sample) → sample에 대한 KDE PDF를 생성
2. 내용을 분석하고 그에 맞춰 쪼개서 함수를 작성하는 것이 중요
    - ex. compute_prob_win, total_prob_win, compute_gain, expected_gain 등

### Chapter.10 - Testing
1. 베이지안 가설검정은 Null Hypothesis가 기준이 되는 것이 아니므로, Alternative에 대한 기준이 필요
2. 베이지안 밴딧 전략 + Thompson Sampling
    - 데이터 수집과 활용을 동시에, 각 슬롯 머신을 고를 확률이 해당 머신이 가장 나을 확률에 비례하도록

### Chapter.11 - Comparison
1. np.meshgrid() 함수 → Tensor Product를 위한 grid 연산 진행, 두 벡터를 넣으면 두 개의 행렬을 반환
2. Joint PDF와 Marginal PDF를 적절히 사용 → Bivariate Joint PDF는 DataFrame형태임을 고려
    - Pmf객체의 marginal(dimension) 메서드 활용 가능

### Chapter.12 - Classification
1. Naive Bayes의 특징 파악 → 각 Feature는 독립적이라고 '가정'
    - likelihood 구성 시 정규분포 등의 분포를 구축(ex. 해당 데이터의 분포도 파악)
    - Multivariate Normal Dist 등을 이용하여 덜 Naive하게 만들 수 있으나, 큰 의미 없음

### Chapter.13 - Inference
1. np.meshgrid()를 활용하여 여러 모수들과 데이터에 대한 축을 만들고 확률밀도 형성
    - numpy의 prod() 메서드를 통해 3차원 Joint PDF를 2차원의 Marginal PDF로 전환 가능 → likelihood로 활용
    - 데이터셋이 크거나 차원이 높아질 경우 likelihood가 매우 작아짐 → 요약값의 likelihood를 구하는 방법 활용
        - 데이터셋을 포함한 3차원에서, 데이터는 요약값을 활용하므로 2차원으로 변경

### Chapter.14 - Survival Analysis
1. Weibull Distribution : 여러 범위의 제품 수명 분포를 나타내는데 좋은 모델 → ss.weibull_min(k, scale=lam) 활용
    - Y = (X^(1/k)) / λ  cf. X ~ exp(1)의 Y에 대한 분포 → k가 1일 경우 지수분포 형태
2. ss.weibull_min()의 sf()메서드 활용하여 생존함수 표시 가능 → censored 데이터에 적용

### Chapter.15 - Mark and Recapture
1. Mark and Recapture : 표본 추출 후 특정 표식 → 동일한 모집단에서 재추출 → 표식 개체 확인을 통해 모집단 크기 추정
    - 초기하분포 활용 → N, K, n, k에서 첫 번째 추출 표본이 K, 두 번째 추출 표본이 n, 표식 표본이 k인 형태
    - 2회의 관측에서 관측된 여부를 가지고 다항분포 활용도 가능 → [k00, k10, k01, k11] 형태