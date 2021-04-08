package env;

import ai.djl.modality.rl.ActionSpace;
import ai.djl.modality.rl.LruReplayBuffer;
import ai.djl.modality.rl.ReplayBuffer;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import game.PlayHex;

public class HexGame implements RlEnv {
	private static final boolean DEBUG_MODE = true;
	private NDManager manager;
	private ReplayBuffer replayBuffer;
	private State state;

	private int rows;
	private int columns;


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

	@Override
	public void reset() {
		this.state = new State(rows, columns);		// makes a new PlayHex object
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

		State preState = state;

		state.winner = state.boardGame.setMove(move);
		HexGameStep step = new HexGameStep(manager.newSubManager(), preState, state, action);
		if (isTraining){
			replayBuffer.addStep(step);
		}
		return step;
	}

	@Override
	public float runEnvironment(RlAgent agent, boolean training) {
		return RlEnv.super.runEnvironment(agent, training);
	}

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
		private NDManager manager;
		private State preState;
		private State postState;
		private NDList action;

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
		public NDArray getReward() { return manager.create((float) postState.getWinner()); }

		@Override
		public boolean isDone() { return postState.isDraw() || postState.getWinner() != 0; }

		@Override
		public void close() { manager.close(); }
	}

	private static final class State{

		PlayHex boardGame;
		int turn;
		int winner = 0;			// is set to either 1 or 2 if blue or red wins

		private State(int rows, int columns){
			this.boardGame = new PlayHex(rows, columns);
			this.turn = boardGame.player %2 +1;
		}

		private NDList getObservation(NDManager manager){
			this.turn = boardGame.player %2 +1;
			return new NDList(manager.create(boardGame.getBoardList()), manager.create(turn));
		}

		private ActionSpace getActionSpace(NDManager manager){
			ActionSpace actionSpace = new ActionSpace();
			actionSpace.add(new NDList(manager.create(boardGame.getBoardList())));
			if (DEBUG_MODE) System.out.println(actionSpace);
			return actionSpace;
		}

		private int getWinner(){ return winner; }

		private boolean isDraw(){ return boardGame.isDraw(); }

		@Override
		public String toString(){ return boardGame.toString(); }

	}

	/**
	 * Run this to tesst if your mxnet installation and your NDanager starts correctly.
	 */
	public static void main(String[] args) {
		Logger logger = LoggerFactory.getLogger(HexGame.class);
		State state = new State(11,11);
		NDManager manager = NDManager.newBaseManager();
		System.out.println(state.getObservation(manager));
		state.getActionSpace(manager);
	}
}
