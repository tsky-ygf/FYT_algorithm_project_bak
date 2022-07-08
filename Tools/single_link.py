from loguru import logger


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    def has_value(self, value):
        if self.data == value:
            return True
        else:
            return False


class Singlelink():

    def __init__(self):
        self.head = None
        self.tail = None
        self.length = 0
        return

    def is_empty(self):
        return self.length == 0

    def add_node(self, item):
        if not isinstance(item, Node):
            item = Node(item)
        if self.head == None:
            self.head = item
            self.tail = item
        else:
            self.tail.next = item
            self.tail = item
        self.length += 1

    def delete_node(self, index: int):
        if self.is_empty():
            logger.info(f"'this link is empty")
            return
        if index < 0 or index > self.length:
            logger.error(f"out of index")
            return
        j = 0
        node = self.head
        prev = self.head
        while node.next and j < index:
            prev = node
            node = node.next
            j += 1
        if j == index:
            prev.next = node.next
            del node
            self.length -= 1

    def insert_node(self, index, data):
        if self.is_empty():
            logger.info(f"'this link is empty")
            return
        if index < 0 or index > self.length:
            logger.error(f"out of index")
            return
        if not isinstance(data, Node):
            item = Node(data)
        else:
            item = Node
        if index == 0:
            item.next = self.head
            self.head = item
            self.length += 1
            return
        node = self.head
        prev = self.head
        j = 0
        while node.next and j < index:
            prev = node
            node = node.next
            j += 1
        if j == index:
            item.next = node
            prev.next = item
            self.length += 1

    def print_link(self):
        current_node = self.head
        while current_node is not None:
            logger.info(f"current data:{current_node.data}")
            current_node = current_node.next
        return

    def get_data_list(self):
        current_node = self.head
        data_list = []
        while current_node is not None:
            data_list.append(current_node.data)
            current_node = current_node.next
        return data_list
