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

import javax.sound.midi.Soundbank;
import java.util.Arrays;
import java.util.Random;

public class HexGame implements RlEnv {
	private static final boolean DEBUG_GAME_MODE = false;			// prints debug info for every game that is finished
	private static final boolean DEBUG_STEP_MODE = false;			// prints debug info for every move that is made

	private NDManager manager;
	private ReplayBuffer replayBuffer;
	private State state;

	public int stepCount;
	private static int batchSize;
	private static int bufferSize;

	private final int rows;
	private final int columns;

	/**
	 * @param manager the manager for creating the game in
	 * @param batchSize the number of steps to train on per batch
	 * @param replayBufferSize the number of steps to hold in the buffer
	 */
	public HexGame(NDManager manager, int batchSize, int replayBufferSize, int rows, int columns){
		this.manager = manager;
		manager.setName("HexGame Manager");
		this.state = new State(manager.newSubManager(), rows, columns);
		this.rows = rows;
		this.columns = columns;
		HexGame.batchSize = batchSize;
		HexGame.bufferSize = replayBufferSize;
		this.replayBuffer = new LruReplayBuffer(HexGame.batchSize, bufferSize);
	}

	@Override
	public void reset() {
		state = new State(manager.newSubManager(), rows, columns);		// makes a new PlayHex object
		Random rnd = new Random();
		state.turn = -1 + 2*rnd.nextInt(2);		// sets state to a random -1 or 1
		stepCount = 0;
	}

	// memory leak?
	@Override
	public NDList getObservation() {
		return state.getObservation();
	}

	// i think memory leak is from here?
	@Override
	public ActionSpace getActionSpace() {
		return state.getActionSpace();

	}

	/**
	 * Allows a human to make a move
	 * @param  loc position of the board to take
	 * @return true if the move was valid
	 */
	public boolean move(int loc){
		return state.move(loc);
	}

	public int getWinner(){
		return state.getWinner();
	}


	/**
	 * A single step in the training or implementation of the agent.
	 * @param action		An NDList representing the agents desired position to take on the HexBoard
	 * @param isTraining	If the model is being trained or tested
	 * @return The step to train from
	 */
	@Override
	public Step step(NDList action, boolean isTraining) {
		//action.detach();
		int move = action.singletonOrThrow().getInt();
		if (DEBUG_STEP_MODE){
			System.out.printf("attempting player %d board[%d]=%d%n", state.turn*-1, move, state.boardGame.getBoardList()[move]);
			System.out.println(Arrays.toString(state.boardGame.getBoardList()));
		}

		if (state.boardGame.getBoardList()[move]!=0){
			throw new IllegalArgumentException("Attempted move is on an occupied space!");
		}
		State preState = new State(manager.newSubManager(), state.boardGame.getBoardList());

		state.move(move);

		if (DEBUG_STEP_MODE){
			System.out.println("Moved. Now:");
			System.out.println("board["+move+"]"+"="+ state.boardGame.getBoardList()[move]);
			System.out.println("prestate["+move+"]"+"="+ preState.board[move]);
		}

		HexGameStep step = new HexGameStep(manager.newSubManager(), preState, state, action);
		if (isTraining){
			replayBuffer.addStep(step);
		}

		stepCount++;
		return step;
	}


	@Override
	public Step[] getBatch() {
		Step[] tmp = replayBuffer.getBatch();
		replayBuffer = new LruReplayBuffer(batchSize, bufferSize);
		return tmp;
	}

	@Override
	public void close() {
		state = null;
		if (DEBUG_GAME_MODE)
			System.out.println("closing and resetting "+manager.getName());
		replayBuffer = new LruReplayBuffer(batchSize, bufferSize);
		manager.close();
		manager = NDManager.newBaseManager();
		manager.setName("HexGame Manager");
	}

	@Override
	public String toString(){
		return state.toString();
	}

	/**
	 * Two states for the neural net to compare.
	 */
	public class HexGameStep implements RlEnv.Step {
		private final NDManager manager;
		private final State preState;
		private final State postState;
		private final NDList action;

		private HexGameStep(NDManager manager, State preState, State postState, NDList action){
			if (!manager.isOpen()){
				throw new IllegalArgumentException("Hey! You're meant to be an opened manager. I want to speak with your manager.");
			}
			this.manager = manager;
			this.manager.setName("HexGameStep Manager");
			this.preState = preState;
			this.postState = postState;
			this.action = action;
		}

		@Override
		public NDList getPreObservation() { return preState.getObservation();}

		@Override
		public NDList getAction() { return action; }

		@Override
		public NDList getPostObservation() { return postState.getObservation();}

		@Override
		public ActionSpace getPostActionSpace() { return postState.getActionSpace();}

		@Override
		public NDArray getReward() {
			return manager.create((float) postState.getWinner()); } // reward is always 1 or 0

		@Override
		public boolean isDone() {
			return postState.getWinner() != 0;
		}

		@Override
		public void close() {
			if (DEBUG_STEP_MODE)
				System.out.println("closing! "+manager.getName());
			preState.subMgr.close();
			postState.subMgr.close();
			manager.close(); }
	}

	private static final class State{

		private final PlayHex boardGame;
		int turn;					// blue always starts
		int winner;					// is set to either 1 or 2 if blue or red wins
		private int[] board;		// the board list. Only exists for the preState State in the step.
		private NDManager subMgr;	// the sub manager for the State. Close after use!

		private State(NDManager subMgr, int[] board){		// This constructor is only used for the preState
			this.subMgr = subMgr;
			this.board = board;
			boardGame = null;
		}
		private State(NDManager subMgr, int rows, int columns) {
			this.subMgr = subMgr;
			this.boardGame = new PlayHex(rows, columns);
		}

		/**
		 * @return true if move was valid
		 */
		private boolean move(int loc){
			turn   = -turn;
			this.winner = boardGame.setMove(loc, turn);
			return boardGame.wasValid();
		}

		private NDList getObservation(){
			if (board==null){
				board = boardGame.getBoardList();
			}
			return new NDList(subMgr.create(board), subMgr.create(turn));
		}

		// memory leak is here? something with the manager not closing when called from HexGame getActionSpace?
		private ActionSpace getActionSpace(){
			ActionSpace actionSpace = new ActionSpace();
			for (int i = 0; i < boardGame.getBoardList().length; i++) {
				if (boardGame.getBoardList()[i]==0){
					actionSpace.add(new NDList(subMgr.create(i)));
				}
			}
			return actionSpace;
		}

		private int getWinner(){ return winner; }

		@Override
		public String toString(){ return boardGame.toString(); }
	}

	/**
	 * Run this to test if your mxnet installation and your NDmanager starts correctly.
	 */
	public static void main(String[] args) {
		Logger logger = LoggerFactory.getLogger(HexGame.class);
		//State state = new State(11,11);
		NDManager manager = NDManager.newBaseManager();
		//System.out.println(state.getObservation(manager));
		//state.getActionSpace(manager);
	}
}
