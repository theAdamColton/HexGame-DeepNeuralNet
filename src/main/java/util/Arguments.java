package util;

import org.apache.commons.cli.Option;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

public class Arguments {

    protected int epoch;
    protected int batchSize;
    protected int saveEveryX;
    protected int closeEveryX;
    protected int gamesPerEpoch;
    protected boolean debugTrain;
    protected String modelPath;

    protected void initialize() {
        epoch = 1000;
        modelPath = "src/main/resources/model/";
        debugTrain = false;
        gamesPerEpoch = 1000;
        batchSize = 20;
        saveEveryX = 10;
        closeEveryX = 3;
    }

    public Arguments parseArgs(String[] args){
        initialize();
        Options options = getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            if (cmd.hasOption("help")){
                printHelp("Help me if you can I'm feeling down...", options);
                return null;
            }
            setArgs(cmd);
            return this;
        } catch (ParseException e) {
            printHelp("Oopsie Poopsie!", options);
        }
        return null;
    }

    protected void setArgs(CommandLine cmd){
        if (cmd.hasOption("epoch")){
            epoch = Integer.parseInt(cmd.getOptionValue("epoch"));
        }
        if (cmd.hasOption("batch")){
            batchSize = Integer.parseInt(cmd.getOptionValue("batch"));
        }
        if (cmd.hasOption("save")){
            saveEveryX = Integer.parseInt(cmd.getOptionValue("save"));
        }
        if (cmd.hasOption("games")){
            gamesPerEpoch = Integer.parseInt(cmd.getOptionValue("games"));
        }
        if (cmd.hasOption("debugTrain")){
            debugTrain = true;
        }
        if (cmd.hasOption("close")){
            closeEveryX = Integer.parseInt(cmd.getOptionValue("close"));
        }
        if (cmd.hasOption("path")){
            modelPath = cmd.getOptionValue("epoch");
        }
    }

    public Options getOptions() {
        Options options = new Options();
        options.addOption(
                Option.builder("h").longOpt("help").hasArg(false).desc("Print Help").build()
        );
        options.addOption(
                Option.builder("e")
                .longOpt("epoch")
                .hasArg()
                .argName("EPOCH")
                .desc("Number of epochs to train")
                .build());
        options.addOption(
                Option.builder("b")
                        .longOpt("batch")
                        .hasArg()
                        .argName("BATCH_SIZE")
                        .desc("Number of moves to add to the batch to train together," +
                                " changing this might help speed up or slow down the training.")
                        .build());
        options.addOption(
                Option.builder("s")
                        .longOpt("save")
                        .hasArg()
                        .argName("SAVE_EVERY_X")
                        .desc("Save every x epochs")
                        .build());
        options.addOption(
                Option.builder("g")
                        .longOpt("games")
                        .hasArg()
                        .argName("GAMES_PER_EPOCH")
                        .desc("Number of batches run desired per epoch")
                        .build());
        options.addOption(
                Option.builder("d")
                        .longOpt("debugTrain")
                        .hasArg()
                        .argName("DEBUG_TRAINING")
                        .desc("Causes the training class to print out some more information")
                        .build());
        options.addOption(
                Option.builder("c")
                        .longOpt("close")
                        .hasArg()
                        .argName("CLOSE_EVERY_X")
                        .desc("Number of epochs before the program refreshes the memory heap")
                        .build());
        options.addOption(
                Option.builder("p")
                        .longOpt("path")
                        .hasArg()
                        .argName("MODEL_PATH")
                        .desc("Path to load the model from")
                        .build());
        return options;
    }

    private void printHelp(String msg, Options options){
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(msg, options);
    }

    public int getEpoch(){ return epoch; }
    public int getBatchSize(){ return batchSize; }
    public int getSaveEveryX(){ return saveEveryX; }
    public int getCloseEveryX(){ return closeEveryX; }
    public int getGamesPerEpoch(){ return gamesPerEpoch; }
    public boolean isDebugTrain(){ return debugTrain; }
    public String getModelPath(){ return modelPath; }

}
