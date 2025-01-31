# Practical 08: Implement the following.

# A. Membership and Identity Operators | in, not in.

# in not in
a = 10
b = 20
list = [1, 2, 3, 4, 5]

if a in list:
    print("Line 1 - a is available in the given list")
else:
    print("Line 1 - a is not available in the given list")

if b not in list:
    print("Line 2 - b is not available in the given list")
else:
    print("Line 2 - b is available in the given list")

a = 2
if a in list:
    print("Line 3 - a is available in the given list")
else:
    print("Line 3 - a is not available in the given list")



# B. Membership and Identity Operators is, is not.

a = [1, 2, 3]
b = a  # Both a and b reference the same list
c = [1, 2, 3]  # Creating a new list with the same values as a

# Identity operators
if a is b:
    print("a and b reference the same list.")
else:
    print("a and b reference different lists.")

if a is not c:
    print("a and c reference different lists.")
else:
    print("a and c reference the same list.")
