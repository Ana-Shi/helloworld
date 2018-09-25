import basic_structure

class Node(object):
    def __init__(self,param=None,ex=None,next=None):
        self.param,self.ex,self.next = param,ex,next

class poly(object):
    def __init__(self):
        self.root = Node()
        self.length = 0
        self.tailnode = None

    def append(self,param,ex):
        node = Node(param = param,ex = ex)
        if self.tailnode is None:
            self.root.next = node
        else:
            self.tailnode.next = node
        self.tailnode = node
        self.length += 1

    def __len__(self):
        return self.length

    def appendleft(self,param,ex):
        node = Node(param=param, ex=ex)
        headnode = self.root.next
        node.next = headnode
        self.root.next = node
        self.length += 1



    def iter_node(self):
        currnode = self.root.next
        while currnode is not self.tailnode:

            yield currnode
            currnode = currnode.next

        yield currnode

    def __iter__(self):
        for node in self.iter_node():
            yield [node.param,node.ex]


    def remove(self,param,ex):
        if self.length <= 0:
            raise  Exception('Empty')
        prenode = self.root
        currnode = self.root.next
        for currnode in self.iter_node():
            if currnode.param == param and currnode.ex == currnode.ex:
                prenode.next = currnode.next
                self.length -= 1
                return  1
            else:
                prenode = currnode

    def find(self,param,ex):
        index = 0
        for node in self.iter_node():
            if node.param == param and node.ex == node.ex:
                return  index
            index += 1
        return -1


    def popleft(self):
        if len(self) ==0:
            raise Exception('Empty')
        headnode = self.headnode()
        param, ex = headnode.param, headnode.ex
        self.remove(headnode)
        return [param, ex]

    def popleft(self):
        if self.root.next is None:
            raise Exception('pop from empty LinkedList')
        headnode = self.root.next
        self.root.next = headnode.next
        param,ex = headnode.param,headnode.ex
        del headnode
        return [param,ex]


    def clear(self):
            for node in self.iter_node():
                del node
            self.root.next = None
            self.length = 0

class PolyMulti():
    def ouput(self,p):
        b = []
        for value in p.__iter__():
            b.append(value)
        return b

    def input(self):
        p = poly()
        a = input('依序输入二项式的系数和指数：').split()[1:]
        i = 0
        while i < len(a):
            p.append(int(a[i]), int(a[i + 1]))
            i = i + 2
        return p

    def find_ex_node(self,p,ex):
        index = 0
        if p is None:
            return -1

        for node in p.iter_node():
            if ex == node.ex:
                return  index
            index += 1
        return -1

    def insertionSortList(self, p,node):
        cur = p.root.next
        pre = p.root
        if node:
            while cur.next and cur.ex > node.ex:  # 找到插入点
                pre = cur
                cur = cur.next

                print(cur.ex)
            # 插入新的结点
            pre.next = node
            node.next = cur

        return p

    def poly_plus(self,p1,p2):
        result = poly()
        if p1 is None and p2 is None:
            return None
        elif p1 is None:
            return p2
        elif p2 is None:
            return p1

        for node in p1.iter_node():

            m = self.find_ex_node(p2,node.ex)
            if  m == -1:
                p2 = self.insertionSortList(p2,node)
            else:
                i=0
                currnode = p2.root.next
                prenode = p2.root
                while i < m and currnode is not None:
                    prenode = currnode
                    currcode = currnode.next
                    i = i+1
                if currnode.param + node.param != 0:
                    currnode.param = currnode.param + node.param
                else:
                    p2.remove(currnode)
        return p2

    def poly_mult_for_one_node(self,node,p2):
        p = poly()

        print('node.param:{}  node.ex:{}'.format(node.param,node.ex))
        for node2 in p2.iter_node():

            print('node2.param:{}  node2.ex:{}'.format(node2.param,node2.ex))
            node_new_param = (node2.param)*(node.param)
            node_new_ex = node.ex+node2.ex
            print('node_new.param:{}  node_new.ex:{}'.format(node_new_param, node_new_ex))
            p.append(param=node_new_param,ex=node_new_ex)
        return p

    def poly_mult(self,p1,p2):
        p = poly()
        m = []
        for node in p1.iter_node():
            m.append(node)
        p = self.poly_mult_for_one_node(node,p2)
        self.ouput(p)
        for node in m[1:]:
            p_new = self.poly_mult_for_one_node(node,p2)
            self.ouput(p_new)
            p = self.poly_plus(p_new,p)
        return p



    def run(self):
        p1 = self.input()
        p2 = self.input()
        print(self.ouput(p1))
        print(self.ouput(p2))
        pp = self.poly_mult(p1, p2)
        # p = self.poly_plus(p1,p2)

        # print(self.ouput(p))
        print(self.ouput(pp))






if __name__ == '__main__':
    p1 = PolyMulti()

if __name__ == '__main__':
    p1 = PolyMulti()
    p1.run()






