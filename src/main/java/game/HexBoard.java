package game; /**
 * The fabulous game of Hex!
 *
 * The blue player always goes first!
 *
 * example hex board with columns 6 and rows 3:
 * 1  2  3  4  5  6
 *  7  8  9  10 11 12
 *   13 14 15 16 17 18
 *
 * If numbering from 1 to the total hex size (hex size = height* width),
 * neighbors are location -1 , location +1, location - width, location - width+1, location + width, location + width -1
 */
import java.util.Arrays;
import java.util.Hashtable;
public class    HexBoard {

    private static int rows;
    private static int columns;
    private static int punishRate;
    public int[] board;        // 0 is empty, 1 is blue and -1 is red
    private UnionFindForest<Integer> redForest = new UnionFindForest<>(); // player 2
    private UnionFindForest<Integer> blueForest = new UnionFindForest<>();// player 1

    public HexBoard(int rows, int columns, int punishRate){
        HexBoard.punishRate = punishRate;
        HexBoard.rows = rows;
        HexBoard.columns = columns;
        board = new int[rows*columns +1];
    }

    /**
     * Allows to set the board and refreshes the neighbor forests for the set tile
     * @return returns -1 if the attempted move is invalid, returns 1 if blue won, -1 if red won
     * otherwise, returns 0 if the move was recorded and valid
     * returns punishRate if the move was invalid.
     */
    public int setBoard(int loc, int player){
        int playerSpot = board[loc];

        if (playerSpot!=0){
            return punishRate;
        }

        int[] neighborPlayerPieces = getNeighborPlayerPieces(loc, player);
        board[loc] = player;

        if (player==1) {
            if (refreshForest(blueForest, loc, neighborPlayerPieces)){
                return 100;
            }
        } else{
            if (refreshForest(redForest, loc, neighborPlayerPieces)){
                return -100;
            }
        }

        return 0;
    }

    /**
     * Refreshes the forest, and checks if a player has won
     * returns true if the game was won
     */
    private boolean refreshForest(UnionFindForest<Integer> forest, int loc, int[] neighborPlayerPieces){
        for (int i =0; i <6; i++){
            if (neighborPlayerPieces[i]==0)        // empty list location
                continue;
            forest.union(loc, neighborPlayerPieces[i]);
        }
        if (forest.find(-2)==forest.find(-3) && forest.find(-2)!=null){  // if the two borders are connected
            return true;
        }
        return false;
    }

    /**
     * @return a list of valid neighbor locations of the same player as specified
     */
    public int[] getNeighborPlayerPieces(int loc, int player){
        int[] out = new int[6];
        int[] locs = getNeighbors(loc);


        for (int i = 0; i < locs.length; i++){
            if (locs[i] <=0) continue;
            Integer currLoc = board[locs[i]];
            if (currLoc!=null&&currLoc==player){
                out[i] = locs[i];
            }
        }
        return getBorders(out, loc, player);
    }

    private int[] getBorders(int[] out, int loc, int player){
        if (player==1){
            if (isOnLeftBorder(loc))
                out[0] = -2;                    // left border for blue pieces counts as a valid piece
            else if (isOnRightBorder(loc))
                out[1] = -3;
        } else if (player==-1){
            if (isOnTopBorder(loc))
                out[2] = -2;
            else if (isOnLowBorder(loc))
                out[4] = -3;
        }
        return out;
    }

    /**
     * Returns a list of valid neighbor locations
     * @param loc
     * @return  neighborList, in order of index: left[0], right[1], up left[2], up right[3], down left[4], down right[5]
     *          empty slots are filled with 0
     */
    public int[] getNeighbors(int loc){
        int[] neighborList= new int[6];

        if (!isOnLeftBorder(loc)){
            neighborList[0] = loc-1;
        }
        if (!isOnRightBorder(loc)){
            neighborList[1] = loc+1;
        }
        if (!isOnTopBorder(loc)){
            neighborList[2] = loc- columns;
            if (!isOnRightBorder(neighborList[2])){
                neighborList[3] = loc- columns +1;
            }
        }
        if (!isOnLowBorder(loc)){
            neighborList[5] = loc+ columns;
            if (!isOnLeftBorder(neighborList[5])){
                neighborList[4] = loc+ columns -1;
            }
        }

        return neighborList;
    }

    /**
     * Use this to print the board. Uses colors!
     * @return The string representation of the board
     */
    public String toString(){
        String outStr = "";
        for (int j = 0; j < rows; j++){
            for (int i = 0; i < columns; i++){
                outStr += hexSquare(columns * j + i +1) + " ";
            }
            outStr+="\n";
            for (int k = 0; k < j+1; k++){
                outStr+=" ";
            }
        }
        return outStr;
    }

    private String hexSquare(int location){
        if (board[location]==0){
            return "" + 0;
        }
        else if (board[location]==1){
            return "\u001B[34m" + "B" + "\u001B[0m";
        } else {
            return "\u001B[31m" + "R" + "\u001B[0m";
        }
    }
    private boolean isOnTopBorder(int loc){
        return loc- columns <1;
    }
    private boolean isOnLowBorder(int loc){
        return loc+ columns > columns * rows;
    }
    private boolean isOnRightBorder(int loc){
        return loc% columns ==0;
    }
    private boolean isOnLeftBorder(int loc){
        return loc% columns ==1;
    }
}
