import ai.djl.Model;
import ai.djl.modality.rl.agent.EpsilonGreedy;
import ai.djl.modality.rl.agent.QAgent;
import ai.djl.modality.rl.agent.RlAgent;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.tracker.LinearTracker;
import ai.djl.training.tracker.Tracker;
import env.HexGame;

import java.nio.file.Paths;
import java.sql.SQLOutput;
import java.util.Scanner;

/*
Test the agent with the might of human meat-brain!
 */
public class PlayAgent {
    public static void main(String[] args) {
        play();
    }
    public static void play() {
        HexGame game = new HexGame(NDManager.newBaseManager(), 0, 0, 8, 8);
        Block block = TrainHexGame.getBlock();
        Model model = Model.newInstance("Hex-Game!");
        model.setBlock(block);
        try {
            model.load(Paths.get("src/main/resources/model/"), "Hex-Game!");        } catch (Exception e){
            e.printStackTrace();
        }
        DefaultTrainingConfig config = TrainHexGame.setupTrainingConfig();

        // this part looks important but I really don't know
        Trainer trainer = model.newTrainer(config);
        trainer.initialize(new Shape(1, 8*8), new Shape(1), new Shape(1));

        RlAgent agent = new QAgent(trainer, .5f);
        Tracker exploreRate =
                LinearTracker.builder()						// this sets a declining ramp rate for the exploring
                        .setBaseValue(0.5f)
                        .optSlope(-.9f / (1000 * 1000 * 7))
                        .optMinValue(0.02f)
                        .build();
        agent = new EpsilonGreedy(agent, exploreRate);


        Scanner inputScnr = new Scanner(System.in);  // Create a Scanner object
        game.reset();
        for (int i =0; i < 200; i++){
            // the agents turn
            //float result = game.runEnvironment(agent, false);
            NDList action = agent.chooseAction(game, false);
            game.step(action, false);
            System.out.println("agent moved");
            System.out.println(game.toString());
            isOver(game);

            // the player's turn
            System.out.print("Make a move:");
            int move = Integer.parseInt(inputScnr.nextLine());
            System.out.println("moving: "+move);

            boolean valid = game.move(move-1);
            while (!valid){
                System.out.println("invalid move!");
                move = Integer.parseInt(inputScnr.nextLine());
                valid = game.move(move);
            }
            System.out.println("move successful");
            System.out.println(game);
            isOver(game);
        }

    }

    private static void isOver(HexGame game){
        if (game.getWinner()!=0){
            System.out.println("game over. player "+ game.getWinner() + " won.");
        }
    }
}
