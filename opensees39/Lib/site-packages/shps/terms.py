from math import factorial

def pascal_tri(numRows):
  """Print Pascal's triangle with numRows."""
  for i in range(numRows):
    # loop to get leading spaces
      for j in range(numRows-i+1):
          print(end="  ")

    # loop to get elements of row i
      for j in range(i+1):
          # nCr = n!/((n-r)!*r!)
          print(f"{i}|{j}",end=" ")
#         print(factorial(i)//(factorial(j)*factorial(i-j)), end=" ")

     # print each row in a new line
      print("\n")

def triangle(numRows):
  """Print Pascal's triangle with numRows."""
  for i in range(numRows):
    # loop to get leading spaces
      for j in range(numRows-i+1):
          print(end="    ")

    # loop to get elements of row i
      for j in range(i):
          print(f"{i}|{j}",end=" ")

      print(f"{i}|{i}",end=" ")
      for j in reversed(range(i)):
          print(f"{j}|{i}",end=" ")

     # print each row in a new line
      print("\n")

def lagrange(numRows):
  """Print Pascal's triangle with numRows."""
  for i in range(numRows+1):
    # loop to get leading spaces
      for j in range(numRows-i+1):
          print(end="    ")

    # loop to get elements of row i
      for j in range(i):
          print(f"{i}|{j}",end=" ")

      print(f"{i}|{i}",end=" ")
      for j in reversed(range(i)):
          print(f"{j}|{i}",end=" ")

     # print each row in a new line
      print("\n")


import sys
lagrange(int(sys.argv[1]))


