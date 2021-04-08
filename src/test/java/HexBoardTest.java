/**
 * Make sure to run these test with assertions enabled in the java jvm with -ea
 */
import game.HexBoard;

public class HexBoardTest {
    public static void main(String[] args) {
        HexBoard testHex;

        testHex = new HexBoard(3, 6, 0);
        int[] mylist = new int[]{2,8,14};
        for (int num : mylist){
            testHex.setBoard(num, 2);
        }
        System.out.println(testHex.toString());

    }
}
