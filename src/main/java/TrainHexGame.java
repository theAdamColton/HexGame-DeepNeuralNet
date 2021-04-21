/*
  Adam Colton 2021

  Lots of code in this class came from the tic tac toe example on the DJL examples on their github:
  https://github.com/awslabs/djl/blob/6ec9256c8deb104dff71250cec91aad5e0d968d7/examples/src/main/java/ai/djl/examples/training/TrainTicTacToe.java
 */

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.modality.rl.agent.EpsilonGreedy;
import ai.djl.modality.rl.agent.QAgent;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.modality.rl.env.RlEnv.Step;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.LinearTracker;
import ai.djl.training.tracker.Tracker;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;
import java.lang.System;

import ai.djl.training.util.ProgressBar;
import env.HexGame;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import util.Arguments;

public final class TrainHexGame {
	private static final Logger logger =LoggerFactory.getLogger(TrainHexGame.class);
	private static final boolean DEBUG_MODE = false;
	private static final String MODEL_PATH = "src/main/resources/model/";
	private static final int rows =8;
	private static final int columns =8;

	public TrainHexGame() {}

	public static void main(String[] args) throws IOException {
		run(args);
	}

	public static void run(String[] args) throws IOException {
		Arguments arguments = new Arguments().parseArgs(args);
		if (arguments == null) {
			return;
		}
		int closeMgnEveryXEpochs = arguments.getCloseEveryX();            // this is a flimsy hack

		int epoch = arguments.getEpoch();
		int validationGamesPerEpoch = 1;
		int saveEveryXEpochs = arguments.getSaveEveryX();
		int batchSize = arguments.getBatchSize();
		float rewardDiscount = 0.3f;
		int replayBufferSize = rows * columns;    // It is impossible to tie the game of Hex, so the maximum buffer size is the size of the board
		int gamesPerEpoch = arguments.getGamesPerEpoch(); //
		String modelPath = arguments.getModelPath();

		boolean shouldLoad = true;

		long startTime = System.currentTimeMillis();
		for (int i =0; i < epoch; i++){
			runXEpochs(closeMgnEveryXEpochs, validationGamesPerEpoch, saveEveryXEpochs, batchSize,
					rewardDiscount, replayBufferSize, gamesPerEpoch, modelPath, shouldLoad);
			logger.info("Average total training time of one move {}", (System.currentTimeMillis() - startTime) /
					((float)(i+1)*gamesPerEpoch*batchSize));
		}

	}

	private static TrainingResult runXEpochs(int epoch, int validationGamesPerEpoch, int saveEveryXEpochs,
											 int batchSize, float rewardDiscount, int replayBufferSize,
											 int gamesPerEpoch, String modelPath, boolean shouldLoad) throws IOException
	{
		HexGame game = new HexGame(NDManager.newBaseManager(), batchSize, replayBufferSize, rows, columns);

		Block  block = getBlock();
		Model model = Model.newInstance("Hex-Game!");
		model.setBlock(block);

		// Loads the saved parameters
		if (shouldLoad){
			try {
				loadModel(model);
			} catch (MalformedModelException | IOException e) {
				System.out.println(Arrays.toString(e.getStackTrace()));
			}
		}

		DefaultTrainingConfig config = setupTrainingConfig();

		// this part looks important but I really don't know
		Trainer trainer = model.newTrainer(config);
		trainer.initialize(new Shape(batchSize, rows*columns), new Shape(batchSize), new Shape(batchSize));

		RlAgent agent = new QAgent(trainer, rewardDiscount);
		Tracker exploreRate =
				LinearTracker.builder()						// this sets a declining ramp rate for the exploring
						.setBaseValue(0.5f)
						.optSlope(-.8f / (100 * gamesPerEpoch * 7))
						.optMinValue(0.02f)
						.build();
		agent = new EpsilonGreedy(agent, exploreRate);

		float validationWinRate = 0;
		float trainWinRate = 0;

		for (int i = 0; i < epoch; i++) {
			logger.info("Starting train... "+(i+1));
			int trainingWins = 0;

			long epochStartTime = System.currentTimeMillis();
			// This is done to lighten resource load
			int showDetailsEveryJ = gamesPerEpoch / 50;

			// Initializes the progress bar
			ProgressBar progressBar = new ProgressBar("" + gamesPerEpoch, gamesPerEpoch+1);
			progressBar.start(1);

			for (int j = 0; j < gamesPerEpoch; j++) {

				// Updates the progress bar and calculates game rate
				if (j % showDetailsEveryJ == 0){
					if (j==0) continue;
					progressBar.update(j+1, ((System.currentTimeMillis() - epochStartTime) /
							(float)j) / batchSize +"ms per move");
				}

				// Runs the simulation
				float result = game.runEnvironment(agent, true);
				Step[] batchSteps = game.getBatch();
				agent.trainBatch(batchSteps);
				trainer.step();

				// Record if the game was won
				if (result > 0) {
					trainingWins++;
				}
			}
			progressBar.end();


			trainWinRate = (float) trainingWins / gamesPerEpoch;
			logger.info("{} total moves; Training wins: {}",(i+1)*gamesPerEpoch*batchSize , trainWinRate);

			// Counts win rate after playing {validationGamesPerEpoch} games
			int validationWins = 0;
			for (int j = 0; j < validationGamesPerEpoch; j++) {
				float result = game.runEnvironment(agent, false);
				if (result > 0) {
					validationWins++;
				}
			}
			validationWinRate = (float) validationWins / validationGamesPerEpoch;
			logger.info("Validation wins: {}", validationWinRate);

			saveModel(model);

		}

		TrainingResult trainingResult = trainer.getTrainingResult();
		trainingResult.getEvaluations().put("validate_winRate", validationWinRate);
		trainingResult.getEvaluations().put("train_winRate", trainWinRate);

		model.close();
		trainer.close();
		game.close();

		return trainingResult;
	}

	private static void saveModel(Model model) throws IOException {
		logger.info("Saving model...");
		model.save(Paths.get(MODEL_PATH), "Hex-Game!");
	}

	private static void loadModel(Model model) throws MalformedModelException, IOException {
		model.load(Paths.get(MODEL_PATH), "Hex-Game!");
		logger.info("Loaded model "+ model.getModelPath()+"/"+model.getName());
	}

	/**
	 * This does something. I think it creates the block hidden layers?
	 */
	public static Block getBlock() {
		SequentialBlock outBlock = new SequentialBlock()
				.add(
						arrays -> {
							NDArray board = arrays.get(0); // Shape(N, 9)
							NDArray turn = arrays.get(1).reshape(-1, 1); // Shape(N, 1)
							NDArray action = arrays.get(2).reshape(-1, 1); // Shape(N, 1)

							// Concatenate to a combined vector of Shape(N, 11)
							NDArray combined = NDArrays.concat(new NDList(board, turn, action), 1);

							NDList outList = new NDList(combined.toType(DataType.FLOAT32, true));

							if (DEBUG_MODE){
								for (Shape shape : outList.getShapes())
									System.out.println(shape.toString());
							}

							return outList;
						})
				.add(new Mlp(rows*columns +2, 1, new int[] {20, 20, 20, 20, 20, 20, 20, 20}));
		if (DEBUG_MODE) System.out.println(outBlock.toString());
		return outBlock;
	}

	/**
	 * I also do not know what this does. Taken from the tic tac toe training example
	 */
	public static DefaultTrainingConfig setupTrainingConfig() {
		return new DefaultTrainingConfig(Loss.l2Loss())
				.addTrainingListeners(TrainingListener.Defaults.basic())
				.optOptimizer(
						Adam.builder().optLearningRateTracker(Tracker.fixed(0.0001F)).build());
	}
}
