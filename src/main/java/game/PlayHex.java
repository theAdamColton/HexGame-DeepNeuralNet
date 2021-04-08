package game; /**
 * Takes a file with the move list as an input and inputs the moves until someone wins or the moves run out.
 * Players take turns to take an unoccupied position on the board. The first person to continuously unite their two
 * sides wins.
 *
 */

import ai.djl.util.Hex;

import java.util.Arrays;

/**
 * Allows instantiation of the game and control of the turns.
 */
public class PlayHex {
    static final boolean DEBUG = false;
    static final String RESOURCE_PATH = "src/main/resources/";
    static boolean isVisible = true;                                                           // this controls whether the board is printed out
    public HexBoard HexBoard;
    public int player = 2;                                                                    // 1 is blue, 2 is red. This variable will always be the player who just went.
    public int moves =0;
    public int maxMoves;

    public PlayHex(int rows, int columns){
        maxMoves = rows*columns;
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

        int result = HexBoard.setBoard(location,player);

        if (result!=-1){
            moves++;
        }
        return result;
    }

    /**
     * returns a comprehensive list representation of the board. It would also be useful to know which player's turn it is.
     */
    public int[] getBoardList(){
        if (DEBUG) System.out.println("getting board: " +
                Arrays.toString(Arrays.copyOfRange(HexBoard.board, 1, HexBoard.board.length)));
        return Arrays.copyOfRange(HexBoard.board, 1, HexBoard.board.length);          // index 0 is never used so it is removed.
    }

    /**
     * @return True if the game is a draw and the board is saturated.
     */
    public boolean isDraw(){
        return moves<=maxMoves;
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
            System.out.println(Arrays.toString(game.getBoardList()));
            if (result==1 || result==2){
                System.out.printf("player %d won Hex!%n", result);
                break;
            }
        }
    }
}
