
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
 * Use this file to generate Krebs3 dataset.
 *
 * @author Petr Ryšavý <petr.rysavy@fel.cvut.cz>
 */
public class NonUniformPrior {

    private static final double size = 50;
    private static final double reactionRadius = 4;
    private static final double meanFreeTime = 2;
    private static final double temperature = 293.15;
    private static final double deltaTime = .01;
    private static ReactionsDictionary reactionDictionary = new ReactionsDictionary();
    private static final int N_SERIES = 100;

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws IOException {
        Random r = new Random(42);
        final int[] index = new int[120];
        for (int i = 1; i < index.length; i++)
            index[i] = i;
        shuffle(index, r);

        int combiIndex = 0;
        for (int i = 0; i < 1024; i++) {
            if (nnz(i) != 3) // only three indices sets
                continue;

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

            container.addRandomParticles(generator, gaussian(r, 100), "GTP");
            container.addRandomParticles(generator, gaussian(r, 600), "H2O");
            container.addRandomParticles(generator, gaussian(r, 900), "NAD");
            container.addRandomParticles(generator, gaussian(r, 200), "GDP");
            container.addRandomParticles(generator, gaussian(r, 20), "NADH");
            container.addRandomParticles(generator, gaussian(r, 220), "FAD");

            if ((i & 1) != 0)
                container.addRandomParticles(generator, gaussian(r, 30), "FUMARATE");
            if ((i & 2) != 0)
                container.addRandomParticles(generator, gaussian(r, 20), "CIS-ACONITATE");
            if ((i & 4) != 0)
                container.addRandomParticles(generator, gaussian(r, 100), "MALATE");
            if ((i & 8) != 0)
                container.addRandomParticles(generator, gaussian(r, 30), "OXALOACETATE");
            if ((i & 16) != 0)
                container.addRandomParticles(generator, gaussian(r, 20), "SUCCINYL-COA");
            if ((i & 32) != 0)
                container.addRandomParticles(generator, gaussian(r, 40), "A-K-GLUTARATE");
            if ((i & 64) != 0)
                container.addRandomParticles(generator, gaussian(r, 10), "CITRATE");
            if ((i & 128) != 0)
                container.addRandomParticles(generator, gaussian(r, 5), "SUCCINATE");
            if ((i & 256) != 0)
                container.addRandomParticles(generator, gaussian(r, 40), "ISOCITRATE");
            if ((i & 512) != 0)
                container.addRandomParticles(generator, gaussian(r, 150), "ACETY-COA");

            container.updateParticleList();

            final int timeSteps = 500;
            String[] particles = new String[]{"FUMARATE", "GTP", "H2O", "CIS-ACONITATE", "MALATE", "OXALOACETATE", "FAD", "SUCCINYL-COA", "NAD", "A-K-GLUTARATE", "GDP", "NADH", "CITRATE", "SUCCINATE", "ISOCITRATE", "ACETY-COA"};
            final int[][] series = new int[particles.length][timeSteps];

            for (int t = 0; t < timeSteps; t++) {
                container.advanceParticles(deltaTime);

                for (int p = 0; p < particles.length; p++) {
                    Integer count = container.getParticleMap().get(particles[p]);
                    series[p][t] = count == null ? 0 : count;
                }
            }
            
            // Convert to relative abundances
            final double[][] seriesRelative = new double[particles.length][timeSteps];
            for(int t = 0; t < timeSteps; t++) {
                int sum = 0;
                for(int p = 0; p < particles.length; p++)
                    sum += series[p][t];
                for(int p = 0; p < particles.length; p++)
                    seriesRelative[p][t] = ((double) series[p][t]) / sum;
            }

            List<String> lines = new ArrayList<>();
            for (int p = 0; p < particles.length; p++) {
                StringBuilder sb = new StringBuilder();
                sb.append(particles[p]);
                sb.append('\t');
                for (int t = 0; t < timeSteps; t++) {
                    sb.append(Double.toString(seriesRelative[p][t]));
                    sb.append('\t');
                }
                sb.deleteCharAt(sb.length() - 1);
                lines.add(sb.toString());
            }

            System.err.println("threes" + index[combiIndex] + ".tsv");
            Files.write(Paths.get("threes" + index[combiIndex++] + ".tsv"), lines);
        }
    }

    private static int gaussian(Random r, int mean) {
        return (int) (r.nextGaussian() * (mean / 3) + mean);
    }

    /**
     * Fisher-Yates shuffle, topdown.
     * @param array Array to shuffle.
     */
    private static void shuffle(int[] array, Random random) {
        for (int i = array.length - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
    }

    private static int nnz(int num) {
        int count = 0;
        while (num > 0) {
            count += num & 1;
            num >>= 1;
        }
        return count;
    }

}
