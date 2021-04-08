package game; /**
 * This is my generic union find tree implementation! It allows collection of non overlapping sets of any type E!
 * Enjoy!
 *
 * Allows union, find, create, destroy
 */
import java.util.ArrayList;
import java.util.Hashtable;
public class UnionFindForest<E> {
    private ArrayList<Integer> sizeList;                        // keeps track of all of the sizes of the upTrees, size is negative for root
    private Hashtable<E, Node<E>> nodeHashtable = new Hashtable<>();         // hashes the E value to the corresponding node
    public UnionFindForest(){
    }

    public void create(E value){
        getNode(value);     // adds value to node hash table
    }

    /**
     * Given two Es, will find their root and join the two trees with the larger tree as the new root
     * @param one
     * @param two
     */
    public void union(E one, E two){
        if (one == null) {
            if (two == null) {
                return;
            }
            getNode(two);
            return;
        } if (two==null){
            getNode(one);
            return;
        }

        Node<E> oneNode = getNode(one);
        Node<E> twoNode = getNode(two);

        union(getRoot(oneNode), getRoot(twoNode));
    }
    private void union(Node<E> bigNode, Node<E> smallNode){
        if (bigNode.size < smallNode.size){         // enforces that bigNode.size > smallNode.size
            Node<E> tmp = smallNode;
            smallNode = bigNode;
            bigNode = tmp;
        }

        if (bigNode==smallNode){
            return;
        }
        smallNode.parent = bigNode;
        bigNode.size+=smallNode.size;
    }

    /**
     *
     * @param val
     * @return the root of the up tree, the identifier for the relation group. If the val is not in the hash table, returns null
     */
    public E find(E val){
        if (!nodeHashtable.containsKey(val)){
            return null;
        }
        Node<E> node = getNode(val);
        return getRoot(node).value;
    }

    /**
     * Destroys an uptree
     */
    public void destroy(){

    }

    /**
     * Figures out if the value exists in the hashtable already, if not it adds it to the hash table as a new Node
     * @return
     */
    private Node<E> getNode(E val){
        Node<E> node = nodeHashtable.get(val);
        if (node==null){
            node = new Node<>(val,1);   // makes a new node with size 1
            nodeHashtable.put(val, node);
        }
        return node;
    }

    /**
     * Returns the root and compresses path as it goes
     * @param node
     * @return
     */
    private Node<E> getRoot(Node<E> node){
        if (node==null){
            return null;
        }
        if (node.parent==null){
            return node;
        }
        ArrayList<Node<E>> compressList = new ArrayList<>();  // stores the nodes to have their parents changed to the root
        while (node != null){
            compressList.add(node);
            node = node.parent;
        }
        node = compressList.get(compressList.size()-1);
        for (Node<E> comNode : compressList){
            comNode.parent = node;
        }
        node.parent = null;
        return node;
    }

    /**
     * A single E node of the non-binary uptree
     * allows large numbers of parents per node
     * @param <E>
     */
    private static class Node<E>{
        E value;                   // the value
        Node<E> parent;            // the parent Node (tree refers up), if root parent is null
        Integer size;              // the size of the up tree
        public Node(E value){
            this.value = value;
        }
        public Node(E value, Integer size){
            this.value=value;
            this.size = size;
        }
        public String toString(){
            return value.toString();
        }

    }
}
