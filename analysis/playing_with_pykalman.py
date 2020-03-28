'''
=============================
EM for Linear-Gaussian Models
=============================
This example shows how one may use the EM algorithm to estimate model
parameters with a Kalman Filter.
The EM algorithm is a meta-algorithm for learning parameters in probabilistic
models. The algorithm works by first fixing the parameters and finding a closed
form distribution over the unobserved variables, then finds new parameters that
maximize the expected likelihood of the observed variables (where the
expectation is taken over the unobserved ones). Due to convexity arguments, we
are guaranteed that each iteration of the algorithm will increase the
likelihood of the observed data and that it will eventually reach a local
optimum.
The EM algorithm is applied to the Linear-Gaussian system (that is, the model
assumed by the Kalman Filter) by first using the Kalman Smoother to calculate
the distribution over all unobserved variables (in this case, the hidden target
states), then closed-form update equations are used to update the model
parameters.
The first figure plotted contains 4 sets of lines. The first, labeled `true`,
represents the true, unobserved state of the system. The second, labeled
`blind`, represents the predicted state of the system if no measurements are
incorporated.  The third, labeled `filtered`, are the state estimates given
measurements up to and including the current time step.  Finally, the fourth,
labeled `smoothed`, are the state estimates using all observations for all time
steps.  The latter three estimates use parameters learned via 10 iterations of
the EM algorithm.
The second figure contains a single line representing the likelihood of the
observed data as a function of the EM Algorithm iteration.
'''
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time

measurements = np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])

initial_state_mean = [measurements[0, 0],
                      0,
                      measurements[0, 1],
                      0]

transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0],
                      [0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

kf1 = kf1.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)
'''
=============================
EM for Linear-Gaussian Models
=============================
This example shows how one may use the EM algorithm to estimate model
parameters with a Kalman Filter.
The EM algorithm is a meta-algorithm for learning parameters in probabilistic
models. The algorithm works by first fixing the parameters and finding a closed
form distribution over the unobserved variables, then finds new parameters that
maximize the expected likelihood of the observed variables (where the
expectation is taken over the unobserved ones). Due to convexity arguments, we
are guaranteed that each iteration of the algorithm will increase the
likelihood of the observed data and that it will eventually reach a local
optimum.
The EM algorithm is applied to the Linear-Gaussian system (that is, the model
assumed by the Kalman Filter) by first using the Kalman Smoother to calculate
the distribution over all unobserved variables (in this case, the hidden target
states), then closed-form update equations are used to update the model
parameters.
The first figure plotted contains 4 sets of lines. The first, labeled `true`,
represents the true, unobserved state of the system. The second, labeled
`blind`, represents the predicted state of the system if no measurements are
incorporated.  The third, labeled `filtered`, are the state estimates given
measurements up to and including the current time step.  Finally, the fourth,
labeled `smoothed`, are the state estimates using all observations for all time
steps.  The latter three estimates use parameters learned via 10 iterations of
the EM algorithm.
The second figure contains a single line representing the likelihood of the
observed data as a function of the EM Algorithm iteration.
'''
from pykalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import time

measurements = np.asarray([(399,293),(403,299),(409,308),(416,315),(418,318),(420,323),(429,326),(423,328),(429,334),(431,337),(433,342),(434,352),(434,349),(433,350),(431,350),(430,349),(428,347),(427,345),(425,341),(429,338),(431,328),(410,313),(406,306),(402,299),(397,291),(391,294),(376,270),(372,272),(351,248),(336,244),(327,236),(307,220)])

initial_state_mean = [measurements[0, 0],
                      0,
                      measurements[0, 1],
                      0]

transition_matrix = [[1, 1, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 1],
                     [0, 0, 0, 1]]

observation_matrix = [[1, 0, 0, 0],
                      [0, 0, 1, 0]]

kf1 = KalmanFilter(transition_matrices = transition_matrix,
                  observation_matrices = observation_matrix,
                  initial_state_mean = initial_state_mean)

kf1 = kf1.em(measurements, n_iter=5)
(smoothed_state_means, smoothed_state_covariances) = kf1.smooth(measurements)

plt.figure(1)
times = range(measurements.shape[0])
plt.plot(times, measurements[:, 0], 'bo',
         times, measurements[:, 1], 'ro',
         times, smoothed_state_means[:, 0], 'b--',
         times, smoothed_state_means[:, 2], 'r--',)
plt.show()