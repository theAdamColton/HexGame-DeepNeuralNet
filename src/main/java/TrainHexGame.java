/**
 * Adam Colton 2021
 *
 * Lots of code in this clas came from the tic tac toe example on the DJL examples on their github:
 * https://github.com/awslabs/djl/blob/6ec9256c8deb104dff71250cec91aad5e0d968d7/examples/src/main/java/ai/djl/examples/training/TrainTicTacToe.java
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

import env.HexGame;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.sound.midi.Soundbank;

public class TrainHexGame {
	private static final Logger logger =LoggerFactory.getLogger(TrainHexGame.class);
	private static final boolean DEBUG_MODE = false;
	private static final String MODEL_PATH = "src/main/resources/model/";
	private static final int SAVE_EVERY_X_STEPS = 5;
	private static final int rows =8;
	private static final int columns =8;

	private TrainHexGame() {}

	public static void main(String[] args) throws IOException {
		TrainHexGame.run(500, 1);
	}

	public static TrainingResult run(int epoch, int batchSize) throws IOException {
		int validationGamesPerEpoch = 0;
		float rewardDiscount = 0.9f;
		int replayBufferSize = 1024;
		int gamesPerEpoch = 500;

		HexGame game = new HexGame(NDManager.newBaseManager(), batchSize, replayBufferSize, rows, columns);

		Block  block = getBlock();

		Model model = Model.newInstance("Hex-Game!");
		model.setBlock(block);

		if (true){			// TODO for loading
			try {
				loadModel(model);
			} catch (MalformedModelException e) {
				System.out.println(Arrays.toString(e.getStackTrace()));
			}
		}



		DefaultTrainingConfig config = setupTrainingConfig();

		// this part looks important but I really don't know
		Trainer trainer = model.newTrainer(config);
		trainer.initialize(new Shape(batchSize, rows*columns), new Shape(batchSize), new Shape(batchSize));

		trainer.notifyListeners(listener -> listener.onTrainingBegin(trainer));

		RlAgent agent = new QAgent(trainer, rewardDiscount);
		Tracker exploreRate =
				LinearTracker.builder()
						.setBaseValue(0.9f)
						.optSlope(-.9f / (epoch * gamesPerEpoch * 7))
						.optMinValue(0.01f)
						.build();
		agent = new EpsilonGreedy(agent, exploreRate);

		float validationWinRate = 0;
		float trainWinRate = 0;
		for (int i = 0; i < epoch; i++) {
			logger.info("Starting Epoch "+ i);
			int trainingWins = 0;
			for (int j = 0; j < gamesPerEpoch; j++) {
				float result = game.runEnvironment(agent, true);
				System.out.printf("Game %4d result: score %15f steps %d%n", j, result, game.stepCount);
				Step[] batchSteps = game.getBatch();
				agent.trainBatch(batchSteps);
				trainer.step();

				// Record if the game was won
				if (result > 0) {
					trainingWins++;
				}
			}
			if (i % SAVE_EVERY_X_STEPS == 0){
				saveModel(model);
			}

			trainWinRate = (float) trainingWins / gamesPerEpoch;
			logger.info("Training wins: {}", trainWinRate);

			trainer.notifyListeners(listener -> listener.onEpoch(trainer));

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
		}

		trainer.notifyListeners(listener -> listener.onTrainingEnd(trainer));

		TrainingResult trainingResult = trainer.getTrainingResult();
		trainingResult.getEvaluations().put("validate_winRate", validationWinRate);
		trainingResult.getEvaluations().put("train_winRate", trainWinRate);

		saveModel(model);
		return trainingResult;

	}

	private static void saveModel(Model model) throws IOException {
		logger.info("Saving model...");
		model.save(Paths.get(MODEL_PATH), "Hex-Game!");
	}

	private static Model loadModel(Model model) throws MalformedModelException, IOException {
		logger.info("Loading Model....");
		model.load(Paths.get(MODEL_PATH), "Hex-Game!");
		return model;
	}

	/**
	 * This does something. I think it creates the hidden layers?
	 * @return
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
				.add(new Mlp(rows*columns +2, 1, new int[] {20, 10}));
		if (DEBUG_MODE) System.out.println(outBlock.toString());
		return outBlock;
	}

	/**
	 * I also do not know what this does. Taken from the tic tac toe training example
	 * @return
	 */
	public static DefaultTrainingConfig setupTrainingConfig() {
		return new DefaultTrainingConfig(Loss.l2Loss())
				.addTrainingListeners(TrainingListener.Defaults.basic())
				.optOptimizer(
						Adam.builder().optLearningRateTracker(Tracker.fixed(0.0001F)).build());
	}
}
