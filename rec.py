import mistofrutta as mf
import numpy as np

A = np.zeros((100,100))

A[50,:] = 5
A[70,:] = 10

rec = mf.geometry.draw.rectangle(A,multiple=True)
Rectangle = rec.getRectangle()

print("Rettangoli")
print(Rectangle)

