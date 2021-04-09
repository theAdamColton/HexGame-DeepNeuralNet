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
    static boolean isVisible = false;                                                           // this controls whether the board is printed out
    public HexBoard HexBoard;
    public int moves =0;
    public int maxMoves;
    private final int rewardRate = 1;

    public PlayHex(int rows, int columns){
        maxMoves = rows*columns;
        // the punishment rate for picking an occupied location, same punishment as loosing. Setting this to 0 will break stuff
        int punishRate = 0;
        int rewardRate = 0;
        int winGameReward = 1;
        HexBoard = new HexBoard(rows, columns, punishRate, rewardRate, winGameReward);
    }

    /**
     * Perform a move! This assumes that the player knows when it is their turn.
     * If you make a invalid move in Hex your turn is wasted!
     * @param location  the location on the board to attempt to take.
     * @return 0 if the attempted move was valid, punishRate if the move was invalid, returns 100 if blue won, -100 if red won
     */
    public int setMove(int location, int player) {
        int result = HexBoard.setBoard(location+1,player);

        if (isVisible) {
            System.out.printf("Player %d takes: %d \t Reward was %d %n",player, location, result);
            System.out.println(HexBoard.toString());
        }


        if (HexBoard.wasValid){       // if the move was valid and recorded, and non winning.
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
        return moves>=maxMoves;
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

        int player = 1;
        for (int move : moveList){
            int result = game.setMove(move, player);
            System.out.println(game.toString());
            System.out.println(Arrays.toString(game.getBoardList()));
            if (result!=0){
                System.out.printf("player %d won Hex!%n", result);
                break;
            }
            player = -player;
        }
    }
}
