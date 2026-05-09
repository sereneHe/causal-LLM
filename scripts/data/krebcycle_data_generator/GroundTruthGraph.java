
import betterChemicalReactions.BoltzmannGenerator;
import betterChemicalReactions.ChemistryDriver;
import betterChemicalReactions.ParticleContainer;
import betterChemicalReactions.RandomGenerator;
import betterChemicalReactions.ReactionsDictionary;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.Random;
import java.util.Scanner;

/**
 * Stores the ground truth adjacecny graph.
 *
 * @author Petr Ryšavý <petr.rysavy@fel.cvut.cz>
 */
public class GroundTruthGraph {

    private static final double size = 50;
    private static final double reactionRadius = 4;
    private static final double meanFreeTime = 2;
    private static final double temperature = 293.15;
    private static final double deltaTime = .01;
    private static ReactionsDictionary reactionDictionary = new ReactionsDictionary();

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        ParticleContainer container = new ParticleContainer(size, size, size, reactionRadius, reactionDictionary);
        RandomGenerator generator = new BoltzmannGenerator(container, temperature);

        HashMap<String, Double> catalysts = new HashMap<>();
        Scanner scan = new Scanner(Paths.get("challenge.txt"));
        ChemistryDriver.parseReaction(scan, catalysts, container);

        final int numPart = 1000;

        final long seed = new Date().getTime();
        Random r = new Random(seed);

        container.addRandomParticles(generator, 10, "FUMARATE");
        container.addRandomParticles(generator, 10, "GTP");
        container.addRandomParticles(generator, 10, "H2O");
        container.addRandomParticles(generator, 10, "CIS-ACONITATE");
        container.addRandomParticles(generator, 10, "MALATE");
        container.addRandomParticles(generator, 10, "OXALOACETATE");
        container.addRandomParticles(generator, 10, "FAD");
        container.addRandomParticles(generator, 10, "SUCCINYL-COA");
        container.addRandomParticles(generator, 10, "NAD");
        container.addRandomParticles(generator, 10, "A-K-GLUTARATE");
        container.addRandomParticles(generator, 10, "GDP");
        container.addRandomParticles(generator, 10, "NADH");
        //container.addRandomParticles(generator105), "OXALOACETATE");
        container.addRandomParticles(generator, 10, "CITRATE");
        container.addRandomParticles(generator, 10, "SUCCINATE");
        container.addRandomParticles(generator, 10, "ISOCITRATE");
        container.addRandomParticles(generator, 10, "ACETY-COA");

        container.updateParticleList();

        final int nParticles = container.getParticleMap().keySet().size();
        String[] particles = container.getParticleMap().keySet().toArray(new String[nParticles]);

        ArrayList<String> reactions = new ArrayList<>();
        for (String particle : particles)
            reactions.add(particle + "_lag1 " + particle + "_lag0");

        for (ReactionsDictionary.ReactionInfo reaction : reactionDictionary) {
            for (String src : reaction.getReactants().keySet())
                for (String target : reaction.getProducts().keySet())
                    reactions.add(src + "_lag1 " + target + "_lag0");
        }

        Files.write(Paths.get("groundtruth.txt"), reactions);

    }

}
