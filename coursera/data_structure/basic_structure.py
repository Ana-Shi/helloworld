class Array(object):
    def __init__(self,size=32):
    
        self._size = size
        self._items = [None]*size
        
    def __getitem__(self,index):
        return self._items[index]
    
    def __setitem__(self,index,value):
        self._items[index] = value
        
    def __len__(self):
        return self._size
    
    def clear(self,value=None):
        for i in range(len(self._items)):
            self._items[i] = value
            
    def __iter__(self):
        for item in self._items:
            yield item
            
class Node(object):
    def __init__(self,value=None,next=None):
        self.value,self.next=value,next


class SingleLinkedList(object):
    def __init__(self,maxsize=None):
        self.maxsize = maxsize
        self.root = Node()
        self.length = 0
        self.tailnode = None
        
    def __len__(self):
        return self.length
    
    def append(self,value):
        if self.maxsize is not None and self.length >= self.maxsize:
            raise Exception('FULL')
        node = Node(value)
        if self.tailnode is None:
            self.root.next = node
        else:
            self.tailnode.next = node
        self.tailnode = node
        self.length += 1
        
    
    def appendleft(self,value):
        headnode = self.root.next
        node = Node(value)
        self.root.next = node
        node.next = headnode
        self.length += 1
        
    def iter_node(self):
        currnode = self.root.next
        while currnode is not self.tailnode:
            yield currnode
            currnode = currnode.next
        yield currnode
        
    def __iter__(self):
        for node in self.iter_node():
            yield node.value
            
    def remove(self,value):
        prenode = self.root
        currnode = self.root.next
        for currnode in self.iter_node():
            if currnode.value == value:
                prenode.next = currnode.next
                del currnode
                self.length -=1
                return 1
            else:
                prenode = currnode
        return -1
                

    def find(self,value):
        index = 0
        for node in self.iter_node():
            if node.value == value:
                return  index
            index += 1
        return -1

    def popleft(self):
        if self.root.next is None:
            raise Exception('pop from empty LinkedList')
        headnode = self.root.next
        self.root.next = headnode.next
        value = headnode.value
        del headnode
        return value


    def clear(self):
            for node in self.iter_node():
                del node
            self.root.next = None
            self.length = 0

class DulNode(object):
    def __init__(self,value=None,pre=None,next=None):
        self.value,self.pre,self.next = value,pre,next
        
class CircualDoubleLinkedList(object):
#循环双向链表
    def __init__(self,maxsize=None):
        self.maxsize = maxsize
        node = Node()
        node.next,node.pre = node,node
        self.root = node
        self.length = 0
    
    def __len__(self):
        return self.length
    
    def tailnode(self):
        return self.root.pre
    
    def headnode(self):
        return self.root.next
    
    def append(self,value):
        if self.maxsize is not None and self.length > self.maxsize:
            raise Exception('FuLL')
        node = Node(value=value)
        tailnode = self.tailnode()
        tailnode.next = node
        node.pre=tailnode
        node.next = self.root
        self.root.pre = node
        self.length += 1
        
    def appendleft(self,value):
        if self.maxsize is not None and self.length > self.maxsize:
            raise Exception('FuLL')
        node = Node(value=value)
        if self.length == 0:
            node.next = self.root
            node.pre = self.root
            self.root.next = node
            self.root.pre = node
            
            
        else:
            node.next = self.root.next
            node.pre = self.root
            self.root.next.pre = node
            self.root.next = node
            
        self.length += 1
        
        
    def remove(self,node):
        if node is self.root:
            return
        node.pre.next = node.next
        node.next.pre = node.pre
        self.length -= 1
        return node
    
    def iter_node(self):
        if self.maxsize is not None and self.length > self.maxsize:
            raise Exception('FuLL')
        currnode = self.root.next
        while currnode.next is not self.root:
            yield currnode
            currnode = currnode.next
        yield currnode
        
    def __iter__(self):
        for node in self.iter_node():
            yield node.value
            
    def iter_node_reverse(self):
        if self.root.pre is self.root:
            raise Exception('Empty')
        currnode = self.root.pre
        while currnode.pre is not self.root:
            yield currnode
            currnode = currnode.pre
        yield currnode
        
class Queue(object):
    def __init__(self, maxsize = None):
        self.maxsize = maxsize
        self._item_linked_list = SingleLinkedList()
        
    def __len__(self):
        return len(self._item_linked_list)
    
    def push(self,value):
        if self.maxsize is not None and len(self) >= self.maxsize:
            raise Exception('FULL')
        return self._item_linked_list.append(value)
    
    def pop(self):
        if len(self) <= 0:
            raise Exception('Empty')
        return self._item_linked_list.popleft()
        
class ArrayQueue(object):
    def __init__(self,maxsize):
        self.maxsize = maxsize
        self.array = Array(maxsize)
        self.head = 0
        self.tail = 0
    
    def __len__(self):
        return self.head - self.tail
    
    def push(self,value):
        if len(self) >= self.maxsize:
            raise Exception('FULL')
            
        self.array[self.head] = value
        self.head += 1
        
    def pop(self):
        value = self.array[self.tail % self.maxsize]
        self.tail += 1
        return value

class Deque(CircualDoubleLinkedList):
    def pop(self):
        if len(self) == 0:
            raise Exception('Empty')
        tailnode = self.tailnode()
        value = tailnode.value()
        self.remove(tailnode)
        return value
    def popleft(self):
        if len(self) ==0:
            raise Exception('Empty')
        headnode = self.headnode()
        value = headnode.value()
        self.remove(headnode)
        return value
    
class Stack():
    def __init__(self):
        self.deque = Deque()
    def push(self,value):
        return self.deque.append(value)
    def pop(self):
        return self.deque.pop()
    def __len__(self):
        return len(self.deque)
    def is_empty(self):
        return len(self) == 0
        
def test_linked_list():
    q = Stack()
    q.push(0)
    q.push(1)
    q.push(2)
    assert len(q) == 3
    assert q.pop() == 0
    assert len(q) ==2
#    
    
    
    
if __name__ == '__main__':
    test_linked_list()
