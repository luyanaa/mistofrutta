import mistofrutta as mf
import numpy as np

A = np.zeros((100,100))

rec = mf.geometry.draw.rectangle(A,multiple=True)
Rectangle = rec.getRectangle()

print("Rettangoli")
print(Rectangle)

