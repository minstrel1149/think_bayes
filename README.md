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