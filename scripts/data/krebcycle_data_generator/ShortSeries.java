
import betterChemicalReactions.BoltzmannGenerator;
import betterChemicalReactions.ChemistryDriver;
import betterChemicalReactions.ParticleContainer;
import betterChemicalReactions.RandomGenerator;
import betterChemicalReactions.ReactionsDictionary;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

/**
 * Use this file to generate KrebsS dataset.
 *
 * @author Petr Ryšavý <petr.rysavy@fel.cvut.cz>
 */
public class ShortSeries {

    private static final double size = 50;
    private static final double reactionRadius = 4;
    private static final double meanFreeTime = 2;
    private static final double temperature = 293.15;
    private static final double deltaTime = .01;
    private static ReactionsDictionary reactionDictionary = new ReactionsDictionary();
    private static final int N_SERIES = 10000;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        for (int i = 0; i < N_SERIES; i++) {
            ParticleContainer container = new ParticleContainer(size, size, size, reactionRadius, reactionDictionary);
            RandomGenerator generator = new BoltzmannGenerator(container, temperature);
            HashMap<String, Double> catalysts = new HashMap<>();
            try {
                Scanner scan = new Scanner(Paths.get("challenge.txt"));
                ChemistryDriver.parseReaction(scan, catalysts, container);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            } catch (IOException e) {
                e.printStackTrace();
            }

            final int numPart = 1000;

            final long seed = new Date().getTime();
            Random r = new Random(seed);

            container.addRandomParticles(generator, gaussian(r, 30), "FUMARATE");
            container.addRandomParticles(generator, gaussian(r, 100), "GTP");
            container.addRandomParticles(generator, gaussian(r, 600), "H2O");
            container.addRandomParticles(generator, gaussian(r, 20), "CIS-ACONITATE");
            container.addRandomParticles(generator, gaussian(r, 100), "MALATE");
            container.addRandomParticles(generator, gaussian(r, 30), "OXALOACETATE");
            container.addRandomParticles(generator, gaussian(r, 220), "FAD");
            container.addRandomParticles(generator, gaussian(r, 20), "SUCCINYL-COA");
            container.addRandomParticles(generator, gaussian(r, 900), "NAD");
            container.addRandomParticles(generator, gaussian(r, 40), "A-K-GLUTARATE");
            container.addRandomParticles(generator, gaussian(r, 200), "GDP");
            container.addRandomParticles(generator, gaussian(r, 20), "NADH");
            //container.addRandomParticles(generator, gaussian(r, 15), "OXALOACETATE");
            container.addRandomParticles(generator, gaussian(r, 10), "CITRATE");
            container.addRandomParticles(generator, gaussian(r, 5), "SUCCINATE");
            container.addRandomParticles(generator, gaussian(r, 40), "ISOCITRATE");
            container.addRandomParticles(generator, gaussian(r, 150), "ACETY-COA");

            container.updateParticleList();

            final int timeSteps = 5;
            //System.err.println(Arrays.toString(particles));
            String[] particles = new String[]{"FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD", "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE", "ISOCITRATE", "ACETY-COA"};
            final int[][] series = new int[particles.length][timeSteps];

            for (int t = 0; t < timeSteps; t++) {
                container.advanceParticles(deltaTime);

                for (int p = 0; p < particles.length; p++) {
                    Integer count = container.getParticleMap().get(particles[p]);
                    series[p][t] = count == null ? 0 : count;
                }
            }


            List<String> lines = new ArrayList<>();
            for (int p = 0; p < particles.length; p++) {
                StringBuilder sb = new StringBuilder();
                sb.append(particles[p]);
                sb.append('\t');
                for (int t = 0; t < timeSteps; t++) {
                    sb.append(Integer.toString(series[p][t]));
                    sb.append('\t');
                }
                sb.deleteCharAt(sb.length() - 1);
                lines.add(sb.toString());
            }

            System.err.println("short" + seed + ".tsv");
            Files.write(Paths.get("short" + seed + ".tsv"), lines);
        }
    }

    private static int gaussian(Random r, int mean) {
        return (int) (r.nextGaussian() * (mean / 3) + mean);
    }

}
