import game.UnionFindForest;

/**
 * Make sure to run these tests with assertions enabled in the java jvm with -ea
 */
public class UnionFindForestTest {
    public static void main(String[] args) {
        UnionFindForest<Integer> forest = new UnionFindForest<>();
        forest.create(0);
        for (int i =0; i <= 20; i++){
            forest.union(0,i);
        }
        for (int i = 0; i <= 20; i++){
            assert forest.find(i)==0 : "Find Failure";
        }
        for (int i = 21; i < 30; i++){
            assert forest.find(i)==null : "Should be null!";
        }
    }
}
