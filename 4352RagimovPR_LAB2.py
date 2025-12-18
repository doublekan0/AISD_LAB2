import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
import sys
import math

sys.setrecursionlimit(1000000)


class TreeNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None
        self.balance_factor = 0
        self.color = 'RED'


class TreeBase:
    def search(self, root, key):
        if root is None or getattr(root, "key", None) == key:
            return root
        if key < root.key:
            return self.search(root.left, key)
        return self.search(root.right, key)

    def get_min_value_node(self, node):
        current = node
        while current and current.left:
            current = current.left
        return current

    def get_max_value_node(self, node):
        current = node
        while current and current.right:
            current = current.right
        return current

    def preorder(self, root):
        if root:
            print(root.key, end=' ')
            self.preorder(root.left)
            self.preorder(root.right)

    def inorder(self, root):
        if root:
            self.inorder(root.left)
            print(root.key, end=' ')
            self.inorder(root.right)

    def postorder(self, root):
        if root:
            self.postorder(root.left)
            self.postorder(root.right)
            print(root.key, end=' ')

    def level_order(self, root):
        if not root:
            return []
        result, queue = [], deque([root])
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        return result

    def calculate_height(self, node):
        if not node:
            return 0
        return 1 + max(self.calculate_height(node.left), self.calculate_height(node.right))


class BST(TreeBase):
    def insert(self, root, key):
        if not root:
            return TreeNode(key)
        parent = None
        cur = root
        while cur:
            parent = cur
            if key == cur.key:
                return root
            elif key < cur.key:
                cur = cur.left
            else:
                cur = cur.right
        new_node = TreeNode(key)
        new_node.parent = parent
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        return root

    def delete(self, root, key):
        if not root:
            return root
        if key < root.key:
            root.left = self.delete(root.left, key)
            if root.left:
                root.left.parent = root
        elif key > root.key:
            root.right = self.delete(root.right, key)
            if root.right:
                root.right.parent = root
        else:
            if not root.left:
                temp = root.right
                if temp:
                    temp.parent = root.parent
                return temp
            elif not root.right:
                temp = root.left
                if temp:
                    temp.parent = root.parent
                return temp
            temp = self.get_min_value_node(root.right)
            root.key = temp.key
            root.right = self.delete(root.right, temp.key)
            if root.right:
                root.right.parent = root
        return root

    def get_height(self, root):
        return self.calculate_height(root)


class AVL(BST):
    def _calculate_node_balance(self, node):
        if not node:
            return 0
        left_height = self.calculate_height(node.left)
        right_height = self.calculate_height(node.right)
        return left_height - right_height

    def _update_balance_factors(self, node):
        if not node:
            return
        node.balance_factor = self._calculate_node_balance(node)
        self._update_balance_factors(node.left)
        self._update_balance_factors(node.right)

    def rotate_right(self, y):
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        if T2:
            T2.parent = y
        x.parent = y.parent
        y.parent = x
        self._update_balance_factors(y)
        self._update_balance_factors(x)
        return x

    def rotate_left(self, x):
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        if T2:
            T2.parent = x
        y.parent = x.parent
        x.parent = y
        self._update_balance_factors(x)
        self._update_balance_factors(y)
        return y

    def insert(self, root, key):
        if not root:
            return TreeNode(key)
        if key == root.key:
            return root
        if key < root.key:
            root.left = self.insert(root.left, key)
            if root.left:
                root.left.parent = root
        else:
            root.right = self.insert(root.right, key)
            if root.right:
                root.right.parent = root
        root.balance_factor = self._calculate_node_balance(root)
        balance = root.balance_factor
        if balance > 1 and key < root.left.key:
            return self.rotate_right(root)
        if balance < -1 and key > root.right.key:
            return self.rotate_left(root)
        if balance > 1 and key > root.left.key:
            root.left = self.rotate_left(root.left)
            if root.left:
                root.left.parent = root
            return self.rotate_right(root)
        if balance < -1 and key < root.right.key:
            root.right = self.rotate_right(root.right)
            if root.right:
                root.right.parent = root
            return self.rotate_left(root)
        return root

    def delete(self, root, key):
        if not root:
            return root
        if key < root.key:
            root.left = self.delete(root.left, key)
            if root.left:
                root.left.parent = root
        elif key > root.key:
            root.right = self.delete(root.right, key)
            if root.right:
                root.right.parent = root
        else:
            if not root.left:
                temp = root.right
                if temp:
                    temp.parent = root.parent
                return temp
            elif not root.right:
                temp = root.left
                if temp:
                    temp.parent = root.parent
                return temp
            temp = self.get_min_value_node(root.right)
            root.key = temp.key
            root.right = self.delete(root.right, temp.key)
            if root.right:
                root.right.parent = root
        if not root:
            return root
        root.balance_factor = self._calculate_node_balance(root)
        balance = root.balance_factor
        if balance > 1 and self._calculate_node_balance(root.left) >= 0:
            return self.rotate_right(root)
        if balance > 1 and self._calculate_node_balance(root.left) < 0:
            root.left = self.rotate_left(root.left)
            if root.left:
                root.left.parent = root
            return self.rotate_right(root)
        if balance < -1 and self._calculate_node_balance(root.right) <= 0:
            return self.rotate_left(root)
        if balance < -1 and self._calculate_node_balance(root.right) > 0:
            root.right = self.rotate_right(root.right)
            if root.right:
                root.right.parent = root
            return self.rotate_left(root)
        return root

    def get_balance_info(self, root):
        if not root:
            return "Empty tree"
        max_balance = 0
        min_balance = 0

        def traverse(node):
            nonlocal max_balance, min_balance
            if node:
                max_balance = max(max_balance, abs(node.balance_factor))
                min_balance = min(min_balance, node.balance_factor)
                traverse(node.left)
                traverse(node.right)

        traverse(root)
        return f"Max |balance|: {max_balance}, Min balance: {min_balance}"


class RBT:
    def __init__(self):
        self.NIL = TreeNode(None)
        self.NIL.color = 'BLACK'
        self.NIL.left = self.NIL.right = self.NIL.parent = self.NIL
        self.root = self.NIL

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node == self.NIL or key == node.key:
            return node
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

    def get_min(self):
        if self.root == self.NIL:
            return None
        node = self.root
        while node.left != self.NIL:
            node = node.left
        return node.key

    def get_max(self):
        if self.root == self.NIL:
            return None
        node = self.root
        while node.right != self.NIL:
            node = node.right
        return node.key

    def preorder_traversal(self):
        result = []
        self._preorder_helper(self.root, result)
        return result

    def _preorder_helper(self, node, result):
        if node != self.NIL:
            result.append(node.key)
            self._preorder_helper(node.left, result)
            self._preorder_helper(node.right, result)

    def inorder_traversal(self):
        result = []
        self._inorder_helper(self.root, result)
        return result

    def _inorder_helper(self, node, result):
        if node != self.NIL:
            self._inorder_helper(node.left, result)
            result.append(node.key)
            self._inorder_helper(node.right, result)

    def postorder_traversal(self):
        result = []
        self._postorder_helper(self.root, result)
        return result

    def _postorder_helper(self, node, result):
        if node != self.NIL:
            self._postorder_helper(node.left, result)
            self._postorder_helper(node.right, result)
            result.append(node.key)

    def level_order_traversal(self):
        if self.root == self.NIL:
            return []
        result = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            result.append(node.key)
            if node.left != self.NIL:
                queue.append(node.left)
            if node.right != self.NIL:
                queue.append(node.right)
        return result

    def insert(self, key):
        node = TreeNode(key)
        node.left = node.right = self.NIL
        node.color = 'RED'
        parent, current = None, self.root
        while current != self.NIL:
            parent = current
            current = current.left if key < current.key else current.right
        node.parent = parent
        if parent is None or parent == self.NIL:
            self.root = node
            node.parent = None
        elif key < parent.key:
            parent.left = node
        else:
            parent.right = node
        self._fix_insert(node)

    def _fix_insert(self, k):
        while k.parent and getattr(k.parent, "color", 'BLACK') == 'RED':
            if k.parent == k.parent.parent.left:
                uncle = k.parent.parent.right
                if getattr(uncle, "color", 'BLACK') == 'RED':
                    k.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    k.parent.parent.color = 'RED'
                    k = k.parent.parent
                else:
                    if k == k.parent.right:
                        k = k.parent
                        self._left_rotate(k)
                    k.parent.color = 'BLACK'
                    k.parent.parent.color = 'RED'
                    self._right_rotate(k.parent.parent)
            else:
                uncle = k.parent.parent.left
                if getattr(uncle, "color", 'BLACK') == 'RED':
                    k.parent.color = 'BLACK'
                    uncle.color = 'BLACK'
                    k.parent.parent.color = 'RED'
                    k = k.parent.parent
                else:
                    if k == k.parent.left:
                        k = k.parent
                        self._right_rotate(k)
                    k.parent.color = 'BLACK'
                    k.parent.parent.color = 'RED'
                    self._left_rotate(k.parent.parent)
        if self.root:
            self.root.color = 'BLACK'

    def delete(self, key):
        node = self._search(self.root, key)
        if node == self.NIL:
            return
        self._delete_node(node)

    def _delete_node(self, node):
        y = node
        y_original_color = y.color
        if node.left == self.NIL:
            x = node.right
            self._transplant(node, node.right)
        elif node.right == self.NIL:
            x = node.left
            self._transplant(node, node.left)
        else:
            y = self._minimum(node.right)
            y_original_color = y.color
            x = y.right
            if y.parent == node:
                if x:
                    x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = node.right
                y.right.parent = y
            self._transplant(node, y)
            y.left = node.left
            y.left.parent = y
            y.color = node.color
        if y_original_color == 'BLACK':
            self._fix_delete(x)

    def _minimum(self, node):
        while node.left != self.NIL:
            node = node.left
        return node

    def _transplant(self, u, v):
        if u.parent is None:
            self.root = v
        elif u == u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        if v is not None:
            v.parent = u.parent

    def _fix_delete(self, x):
        while x != self.root and getattr(x, "color", 'BLACK') == 'BLACK':
            if x == x.parent.left:
                sibling = x.parent.right
                if getattr(sibling, "color", 'BLACK') == 'RED':
                    sibling.color = 'BLACK'
                    x.parent.color = 'RED'
                    self._left_rotate(x.parent)
                    sibling = x.parent.right
                if getattr(sibling.left, "color", 'BLACK') == 'BLACK' and getattr(sibling.right, "color",
                                                                                  'BLACK') == 'BLACK':
                    sibling.color = 'RED'
                    x = x.parent
                else:
                    if getattr(sibling.right, "color", 'BLACK') == 'BLACK':
                        sibling.left.color = 'BLACK'
                        sibling.color = 'RED'
                        self._right_rotate(sibling)
                        sibling = x.parent.right
                    sibling.color = x.parent.color
                    x.parent.color = 'BLACK'
                    sibling.right.color = 'BLACK'
                    self._left_rotate(x.parent)
                    x = self.root
            else:
                sibling = x.parent.left
                if getattr(sibling, "color", 'BLACK') == 'RED':
                    sibling.color = 'BLACK'
                    x.parent.color = 'RED'
                    self._right_rotate(x.parent)
                    sibling = x.parent.left
                if getattr(sibling.right, "color", 'BLACK') == 'BLACK' and getattr(sibling.left, "color",
                                                                                   'BLACK') == 'BLACK':
                    sibling.color = 'RED'
                    x = x.parent
                else:
                    if getattr(sibling.left, "color", 'BLACK') == 'BLACK':
                        sibling.right.color = 'BLACK'
                        sibling.color = 'RED'
                        self._left_rotate(sibling)
                        sibling = x.parent.left
                    sibling.color = x.parent.color
                    x.parent.color = 'BLACK'
                    sibling.left.color = 'BLACK'
                    self._right_rotate(x.parent)
                    x = self.root
        if x:
            x.color = 'BLACK'

    def _left_rotate(self, x):
        y = x.right
        x.right = y.left
        if y.left != self.NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
            y.parent = None
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _right_rotate(self, x):
        y = x.left
        x.left = y.right
        if y.right != self.NIL:
            y.right.parent = x
        y.parent = x.parent
        if x.parent is None:
            self.root = y
            y.parent = None
        elif x == x.parent.right:
            x.parent.right = y
        else:
            x.parent.left = y
        y.right = x
        x.parent = y

    def get_height(self):
        return self._height(self.root)

    def _height(self, node):
        if node == self.NIL or node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))


def run_experiment(tree_type, max_keys=20000, step=2000, sorted_keys=False, tree_name=""):
    heights, sizes = [], []
    for size in range(step, max_keys + 1, step):
        if tree_type == RBT:
            tree = RBT()
            if sorted_keys:
                keys = list(range(1, size + 1))
            else:
                keys = random.sample(range(max_keys * 100), size)
            for key in keys:
                tree.insert(key)
            heights.append(tree.get_height())
        else:
            tree = tree_type()
            root = None
            if sorted_keys:
                keys = list(range(1, size + 1))
            else:
                keys = random.sample(range(max_keys * 100), size)
            for key in keys:
                root = tree.insert(root, key)
            heights.append(tree.get_height(root))
        sizes.append(size)
    return sizes, heights


def show_tree_demo():
    random.seed(42)
    keys = random.sample(range(1, 100), 15)
    print(f"Случайные ключи: {sorted(keys)}")
    bst = BST()
    avl = AVL()
    rbt = RBT()
    bst_root = None
    avl_root = None
    for key in keys:
        bst_root = bst.insert(bst_root, key)
        avl_root = avl.insert(avl_root, key)
        rbt.insert(key)

    print("=" * 70)
    print("ДЕМОНСТРАЦИЯ ПОИСКА")
    print("=" * 70)

    existing_key = keys[7]
    non_existing_key = 999

    print(f"\nПоиск ключа {existing_key}:")
    bst_result = bst.search(bst_root, existing_key)
    avl_result = avl.search(avl_root, existing_key)
    rbt_result = rbt.search(existing_key)
    print(f"BST: {'НАЙДЕН' if bst_result else 'НЕ НАЙДЕН'}")
    print(f"AVL: {'НАЙДЕН' if avl_result else 'НЕ НАЙДЕН'}")
    print(f"RBT: {'НАЙДЕН' if rbt_result != rbt.NIL else 'НЕ НАЙДЕН'}")

    print(f"\nПоиск ключа {non_existing_key}:")
    bst_result = bst.search(bst_root, non_existing_key)
    avl_result = avl.search(avl_root, non_existing_key)
    rbt_result = rbt.search(non_existing_key)
    print(f"BST: {'НАЙДЕН' if bst_result else 'НЕ НАЙДЕН'}")
    print(f"AVL: {'НАЙДЕН' if avl_result else 'НЕ НАЙДЕН'}")
    print(f"RBT: {'НАЙДЕН' if rbt_result != rbt.NIL else 'НЕ НАЙДЕН'}")

    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ УДАЛЕНИЯ")
    print("=" * 70)

    delete_key = keys[5]
    non_delete_key = 888

    print(f"\nУдаление существующего ключа {delete_key}:")
    bst_root = bst.delete(bst_root, delete_key)
    avl_root = avl.delete(avl_root, delete_key)
    rbt.delete(delete_key)
    print("Ключ удален из всех деревьев")

    print(f"\nУдаление несуществующего ключа {non_delete_key}:")
    bst_root = bst.delete(bst_root, non_delete_key)
    avl_root = avl.delete(avl_root, non_delete_key)
    rbt.delete(non_delete_key)
    print("Такого ключа нет в деревьях")

    print("\n" + "=" * 70)
    print("ОБХОДЫ ДЕРЕВЬЕВ ПОСЛЕ ОПЕРАЦИЙ")
    print("=" * 70)

    print("\n=== BST ДЕРЕВО ===")
    print(f"Высота: {bst.get_height(bst_root)}")
    print("Обход в ширину:", ' '.join(map(str, bst.level_order(bst_root))))
    print("Прямой обход: ", end='')
    bst.preorder(bst_root)
    print("\nСимметричный обход: ", end='')
    bst.inorder(bst_root)
    print("\nОбратный обход: ", end='')
    bst.postorder(bst_root)
    print()

    print("\n=== AVL ДЕРЕВО ===")
    print(f"Высота: {avl.get_height(avl_root)}")
    print("Обход в ширину:", ' '.join(map(str, avl.level_order(avl_root))))
    print("Прямой обход: ", end='')
    avl.preorder(avl_root)
    print("\nСимметричный обход: ", end='')
    avl.inorder(avl_root)
    print("\nОбратный обход: ", end='')
    avl.postorder(avl_root)
    print()
    print("Баланс:", avl.get_balance_info(avl_root))

    print("\n=== RBT ДЕРЕВО ===")
    print(f"Высота: {rbt.get_height()}")
    print("Обход в ширину:", ' '.join(map(str, rbt.level_order_traversal())))
    print("Прямой обход:", ' '.join(map(str, rbt.preorder_traversal())))
    print("Симметричный обход:", ' '.join(map(str, rbt.inorder_traversal())))
    print("Обратный обход:", ' '.join(map(str, rbt.postorder_traversal())))


if __name__ == "__main__":
    print("=" * 80)
    print("ЛАБОРАТОРНАЯ РАБОТА №2: АНАЛИЗ БИНАРНЫХ ДЕРЕВЬЕВ ПОИСКА")
    print("=" * 80)

    show_tree_demo()

    sizes_bst_random, heights_bst_random = run_experiment(BST, max_keys=20000, step=2000, sorted_keys=False,
                                                          tree_name="BST")
    sizes_avl_random, heights_avl_random = run_experiment(AVL, max_keys=20000, step=2000, sorted_keys=False,
                                                          tree_name="AVL")
    sizes_rbt_random, heights_rbt_random = run_experiment(RBT, max_keys=20000, step=2000, sorted_keys=False,
                                                          tree_name="RBT")
    sizes_avl_sorted, heights_avl_sorted = run_experiment(AVL, max_keys=20000, step=2000, sorted_keys=True,
                                                          tree_name="AVL")
    sizes_rbt_sorted, heights_rbt_sorted = run_experiment(RBT, max_keys=20000, step=2000, sorted_keys=True,
                                                          tree_name="RBT")

    n = np.linspace(1, 20000, 200)
    y_theory_avl = 1.44 * np.log2(n + 1)
    y_theory_rbt = 2 * np.log2(n + 1)
    y_theory_bst_avg = 1.39 * np.log2(n)
    y_theory_lower = np.log2(n + 1)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(sizes_bst_random, heights_bst_random, 'g-', linewidth=2)
    plt.plot(n, y_theory_bst_avg, 'g--', alpha=0.7, label='Средняя: 1.39·log₂n')
    plt.plot(n, y_theory_lower, 'b--', alpha=0.5, label='Нижняя: log₂(n+1)')
    plt.title('BST: Случайные ключи')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 2)
    plt.plot(sizes_avl_random, heights_avl_random, 'b-', linewidth=2, label='AVL (эксперимент)')
    plt.plot(sizes_rbt_random, heights_rbt_random, 'r-', linewidth=2, label='RBT (эксперимент)')
    plt.plot(n, y_theory_avl, 'b--', alpha=0.5, label='AVL верхняя: 1.44·log₂(n+1)')
    plt.plot(n, y_theory_rbt, 'r--', alpha=0.5, label='RBT верхняя: 2·log₂(n+1)')
    plt.plot(n, y_theory_lower, 'g--', alpha=0.3, label='Нижняя граница')
    plt.title('AVL и RBT: Случайные ключи')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 3)
    plt.plot(sizes_avl_sorted, heights_avl_sorted, 'b-', linewidth=2, label='Эксперимент')
    plt.plot(n, y_theory_avl, 'b--', alpha=0.7, label='Верхняя: 1.44·log₂(n+1)')
    plt.plot(n, y_theory_lower, 'g--', alpha=0.7, label='Нижняя: log₂(n+1)')
    plt.fill_between(n, y_theory_lower, y_theory_avl, alpha=0.1, color='blue')
    plt.title('AVL: Монотонно возрастающие ключи')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.plot(sizes_rbt_sorted, heights_rbt_sorted, 'r-', linewidth=2, label='Эксперимент')
    plt.plot(n, y_theory_rbt, 'r--', alpha=0.7, label='Верхняя: 2·log₂(n+1)')
    plt.plot(n, y_theory_lower, 'g--', alpha=0.7, label='Нижняя: log₂(n+1)')
    plt.fill_between(n, y_theory_lower, y_theory_rbt, alpha=0.1, color='red')
    plt.title('RBT: Монотонно возрастающие ключи')
    plt.xlabel('Количество ключей')
    plt.ylabel('Высота дерева')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()