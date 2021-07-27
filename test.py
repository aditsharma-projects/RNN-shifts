import unittest
import numpy as np

def square(n):
  return n**2

def cube(n):
  return n**3

class Test(unittest.TestCase):
  def test_square(self):
    self.assertEquals(square(2),4)
    
  def test_square(self):
    self.assertEquals(cube(2),8)

#if __name__ == "__main__":
#  print("PASSED")
