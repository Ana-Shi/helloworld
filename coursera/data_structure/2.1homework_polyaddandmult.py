from basic_structure import Deque
class Node(object):
    def __init__(self,param=None,ex=None,next=None):
        self.param ,self.ex,self.next = param,ex,next
        
class LinkedList(object):
    def __init__(self,maxsize=None):
        self.maxsize = maxsize
        self.root = Node()
        self.length = 0
        self.tailnode = None
        
    def __len__(self):
        return self.length
    
    def append(self,param,ex):
        if self.maxsize is not None and self.length >= self.maxsize:
            raise Exception('FULL')
        node = Node(param=param,ex=ex)
        if self.tailnode is None:
            self.root.next = node
        else:
            self.tailnode.next = node
        self.tailnode = node
        self.length += 1
        
    
    def appendleft(self,param,ex):
        headnode = self.root.next
        node = Node(param=param,ex=ex)
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
            yield list(node.param,node.ex)
            
    def remove(self,param,ex):
        prenode = self.root
        currnode = self.root.next
        for currnode in self.iter_node():
            if currnode.param == param and currnode.ex==ex:
                prenode.next = currnode.next
                del currnode
                self.length -=1
                return 1
            else:
                prenode = currnode
        return -1
                

    def find(self,param,ex):
        index = 0
        for node in self.iter_node():
            if node.param == param and node.ex==ex:
                return  index
            index += 1
        return -1

    def popleft(self):
        if self.root.next is None:
            raise Exception('pop from empty LinkedList')
        headnode = self.root.next
        self.root.next = headnode.next
        value = list(headnode.param,headnode.ex)
        del headnode
        return value


    def clear(self):
            for node in self.iter_node():
                del node
            self.root.next = None
            self.length = 0
            
    

class Poly(object):
    def input(self):
        p = LinkedList()
        a = input('依序输入二项式的系数和指数（指数小于10）：').split(' ')[1:]
        i=0
        print(len(a))
        while i < len(a):
            p.append(int(a[i]),int(a[i+1]))
            i = i+2
        return p
    def output(self,p):
        
        b = ''
        for node in p.iter_node():
            if node.param !=0:
                b = b + ' '+ str(node.param)+' '+str(node.ex)
            
        print(b)
        #4 3 4 -5 2 6 1 -2 0
        #3 5 20 -7 4 3 1
    
    def mutl(self,p1,p2):
        result = LinkedList()
        for node1 in p1.iter_node():
            for node2 in p2.iter_node():
                renode = Node(param = node1.param*node2.param,ex = node1.ex+node2.ex)
                if result.root.next is result.tailnode:
                    result.append(param = renode.param,ex=renode.ex)
                    result.length += 1
                else:
                    for node in result.iter_node():
                        if node.ex== renode.ex:
                            node.param += renode.param
                            break
                        elif node.next is not None:
                            if result.root.next is node:
                                if node.ex < renode.ex:
                                    result.root.next = renode
                                    renode.next = node
                                    result.length += 1
                            if node.ex> renode.ex   and node.next.ex < renode.ex:
                                renode.next = node.next
                                node.next = renode
                                result.length += 1
                                break
                        elif node.next is None:
                            if node.ex> renode.ex:
                                result.append(param = renode.param,ex=renode.ex)
                                result.length += 1
                                break
        return result
    def sumlist(self,result,renode):
        if result.root.next is result.tailnode:
                    result.append(param = renode.param,ex=renode.ex)
                    result.length += 1
        else:
            for node in result.iter_node():
                if node.ex== renode.ex:
                    node.param += renode.param
                    break
                elif node.next is not None:
                    if result.root.next is node:
                        if node.ex < renode.ex:
                            result.root.next = renode
                            renode.next = node
                            result.length += 1
                        if node.ex> renode.ex   and node.next.ex < renode.ex:
                            renode.next = node.next
                            node.next = renode
                            result.length += 1
                            break
                    elif node.next is None:
                        if node.ex> renode.ex:
                                result.append(param = renode.param,ex=renode.ex)
                                result.length += 1
                                break
        return result
    
    def add(self,p1,p2):
        result = LinkedList()
        for node1 in p1.iter_node():
            flag = 0
            for node2 in p2.iter_node():
                if node1.ex == node2.ex:
                    renode = Node(param = node1.param+node2.param,ex = node1.ex)
                    flag = 1
                    self.sumlist(result,renode)
            if flag == 0:
                
                self.sumlist(result,node1)
        return result
    
    def console(self):
        p1 = self.input()
        p2 = self.input()
        result1 = self.mutl(p1,p2)
        result2 = self.add(p1,p2)
        self.output(result1)
        self.output(result2)

                         
                        
            
p = Poly()

p.console()
            
            