package env;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import game.PlayHex;

import java.util.Arrays;

public class HexGame implements RlEnv {
	private static final boolean DEBUG_MODE = false;
	private final NDManager manager;
	private final ReplayBuffer replayBuffer;
	private State state;
	public static int stepCount;

	private final int rows;
	private final int columns;


	/**
	 * @param manager the manager for creating the game in
	 * @param batchSize the number of steps to train on per batch
	 * @param replayBufferSize the number of steps to hold in the buffer
	 */
	public HexGame(NDManager manager, int batchSize, int replayBufferSize, int rows, int columns){
		this(manager, new LruReplayBuffer(batchSize, replayBufferSize), rows, columns);
	}
	public HexGame(NDManager manager, ReplayBuffer replayBuffer, int rows , int columns) {
		this.manager = manager;
		this.state = new State(rows, columns);
		this.rows = rows;
		this.columns = columns;
		this.replayBuffer = replayBuffer;
	}

	public int getTurn(){
		return state.turn;
	}

	@Override
	public void reset() {
		this.state = new State(rows, columns);		// makes a new PlayHex object
		state.turn = -1;
		stepCount = 0;
	}

	@Override
	public NDList getObservation() {
		return state.getObservation(manager);
	}

	@Override
	public ActionSpace getActionSpace() {
		return state.getActionSpace(manager);
	}

	/**
	 * A single step in the training or implementation of the agent.
	 * @param action		An NDList representing the agents desired position to take on the HexBoard
	 * @param isTraining	If the model is being trained or tested
	 * @return
	 */
	@Override
	public Step step(NDList action, boolean isTraining) {
		int move = action.singletonOrThrow().getInt();

		//System.out.println("attempting..board["+move+"]"+"="+state.boardGame.getBoardList()[move]);
		//System.out.println(Arrays.toString(state.boardGame.getBoardList()));

		if (state.boardGame.getBoardList()[move]!=0){
			throw new IllegalArgumentException("Attempted move is on an occupied space!");
		}
		State preState = new State(state.boardGame.getBoardList());


		//if (DEBUG_MODE) System.out.printf("p%2dr%2d,", state.turn, state.winner);

		state.turn   = -state.turn;
		state.move(move);

		//System.out.println("Moved.");
		//System.out.println("board["+move+"]"+"="+ state.boardGame.getBoardList()[move]);
		//System.out.println("prestate["+move+"]"+"="+ preState.board[move]);

		HexGameStep step = new HexGameStep(manager.newSubManager(), preState, state, action);
		if (isTraining){
			replayBuffer.addStep(step);
		}
		stepCount++;
		return step;
	}

	//@Override
	//public float runEnvironment(RlAgent agent, boolean training) {
	//	return RlEnv.super.runEnvironment(agent, training);
	//}

	@Override
	public Step[] getBatch() {
		return replayBuffer.getBatch();
	}

	@Override
	public void close() {
		manager.close();
	}

	@Override
	public String toString(){
		return state.toString();
	}

	static final class HexGameStep implements RlEnv.Step {
		private final NDManager manager;
		private final State preState;
		private final State postState;
		private final NDList action;

		private HexGameStep(NDManager manager, State preState, State postState, NDList action){
			this.manager = manager;
			this.preState = preState;
			this.postState = postState;
			this.action = action;
		}

		@Override
		public NDList getPreObservation() { return preState.getObservation(manager);}

		@Override
		public NDList getAction() { return action; }

		@Override
		public NDList getPostObservation() { return postState.getObservation(manager);}

		@Override
		public ActionSpace getPostActionSpace() { return postState.getActionSpace(manager);}

		@Override
		public NDArray getReward() {
			return manager.create((float) postState.getWinner()); }

		@Override
		public boolean isDone() {
			boolean result = postState.isDraw() || postState.getWinner() !=0;	//TODO
			if (result)
				if (DEBUG_MODE)System.out.printf("%nPlayer %2d won Hex! %d steps%n", preState.turn, stepCount);
			return  result;}


		@Override
		public void close() { manager.close(); }
	}

	private static class State{

		private PlayHex boardGame;
		int turn;			// blue always starts
		int winner;			// is set to either 1 or 2 if blue or red wins
		private int rows;
		private int columns;
		private int[] board;

		private State(int[] board){
			this.board = board;
		}
		private State(int rows, int columns){
			boardGame = new PlayHex(rows, columns);
			this.rows = rows;
			this.columns = columns;
		}

		private void move(int loc){
			//System.out.println("Moving " +turn + " to " + loc);
			this.winner = boardGame.setMove(loc, turn);
		}

		private NDList getObservation(NDManager manager){
			if (board==null){
				board = boardGame.getBoardList();
			}
			return new NDList(manager.create(board), manager.create(turn));
		}

		private ActionSpace getActionSpace(NDManager manager){
			ActionSpace actionSpace = new ActionSpace();
			for (int i = 0; i < boardGame.getBoardList().length; i++) {
				if (boardGame.getBoardList()[i]==0){
					actionSpace.add(new NDList(manager.create(i)));
				}
			}
			return actionSpace;
		}

		private int getWinner(){ return winner; }

		private boolean isDraw(){
			if (DEBUG_MODE && boardGame.isDraw())System.out.println("Draw!");
			return boardGame.isDraw(); }

		@Override
		public String toString(){ return boardGame.toString(); }

	}

	/**
	 * Run this to tesst if your mxnet installation and your NDmanager starts correctly.
	 */
	public static void main(String[] args) {
		Logger logger = LoggerFactory.getLogger(HexGame.class);
		//State state = new State(11,11);
		NDManager manager = NDManager.newBaseManager();
		//System.out.println(state.getObservation(manager));
		//state.getActionSpace(manager);
	}
}
