package game; /**
 * Takes a file with the move list as an input and inputs the moves until someone wins or the moves run out.
 * Players take turns to take an unoccupied position on the board. The first person to continuously unite their two
 * sides wins.
 *
 */

/**
 * Allows instantiation of the game and control of the turns.
 */
public class PlayHex {
    static final String RESOURCE_PATH = "src/main/resources/";
    static boolean isVisible = true;                                                           // this controls whether the board is printed out
    public HexBoard HexBoard;
    public int player = 2;                                                                    // 1 is blue, 2 is red. This variable will always be the player who just went.
    public PlayHex(){
        this.HexBoard = new HexBoard(11,11);
    }                   // defaults to 11x11
    public PlayHex(int rows, int columns){
        this.HexBoard = new HexBoard(rows, columns);
    }

    /**
     * Perform a move! This assumes that the player knows when it is their turn.
     * If you make a invalid move in Hex your turn is wasted!
     * @param location  the location on the board to attempt to take.
     * @return -1 if the attempted move is invalid, returns 1 if blue won, 2 if red won
     */
    public int setMove(int location) {
        // swaps the player
        player = player%2 +1;

        if (isVisible) {
            System.out.printf("Player %d takes: %d %n",player, location);
        }

        return HexBoard.setBoard(location, player);
    }

    /**
     * @return A string representation of the board
     */
    public String toString(){
        return HexBoard.toString();
    }

    /**
     * A short demonstration of the Hex Game
     */
    public static void main(String[] args) {
        PlayHex game = new PlayHex(20,20);
        System.out.println(game.toString());
        int[] moveList = {1,100,2,103,3,106,4,108,5,120,6,200,7,53,8,69,9,24,10,1,11,1,12,1,13,1,14,1,15,1,16,1,17,1,18,1,19,1,20,24};


        for (int move : moveList){
            int result = game.setMove(move);
            System.out.println(game.toString());
            if (result==1 || result==2){
                System.out.printf("player %d won Hex!%n", result);
                break;
            }
        }
    }
}
