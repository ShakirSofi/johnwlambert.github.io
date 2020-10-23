


```python
>>> import numpy as np
>>> w = np.array([7,6,5])
>>> P
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'P' is not defined
>>> import numpy as np
>>> P = np.array([[0.3,0.3,0.4],[0.4,0.4,0.2],[0.5,0.3,0.2]])
>>> P
array([[0.3, 0.3, 0.4],
       [0.4, 0.4, 0.2],
       [0.5, 0.3, 0.2]])
>>> w = np.array([7,6,5])
>>> w
array([7, 6, 5])
>>> w / w.sum()
array([0.38888889, 0.33333333, 0.27777778])
>>> 7/18
0.3888888888888889
>>> 6/18
0.3333333333333333
>>> 5/18
0.2777777777777778
>>> import scipy.linalg
>>> scipy.linalg.null_space(P.T - np.eye(3))
array([[-0.66742381],
       [-0.57207755],
       [-0.47673129]])
>>> v = scipy.linalg.null_space(P.T - np.eye(3))
>>> v *= -1
>>> v
array([[0.66742381],
       [0.57207755],
       [0.47673129]])
>>> v /= v.sum()
>>> v
array([[0.38888889],
       [0.33333333],
       [0.27777778]])
>>> quit()

```
