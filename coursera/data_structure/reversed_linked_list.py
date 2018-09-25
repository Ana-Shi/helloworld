from basic_structure import SingleLinkedList

p = SingleLinkedList()
a = input().split()
b = []
for i in range(int(a[1])):
    b.append(input().split())
print(b)