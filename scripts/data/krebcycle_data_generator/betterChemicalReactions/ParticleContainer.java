package betterChemicalReactions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


public class ParticleContainer {

    private Vector3 size;

    private List<Particle> particles = new ArrayList<Particle>();
    private ParticleDictionary dict = new ParticleDictionary();
    private ReactionsDictionary reactionsDictionary;
    private Random random = new Random();

    private List<Particle> toBeAdded = new ArrayList<>();
    private List<Particle> toBeRemoved = new ArrayList<>();

    private double reactionRadius;

    /**
     * @param xSize The size of the box in the x direction
     * @param ySize The size of the box in the y direction
     * @param zSize The size of the box in the z direction
     */
    public ParticleContainer(double xSize, double ySize, double zSize, double reactionRadius, ReactionsDictionary reactionsDictionary) {
        size = new Vector3(xSize, ySize, zSize);
        this.reactionRadius = reactionRadius;
        this.reactionsDictionary = reactionsDictionary;
    }

    public double getXSize() {
        return size.getX();
    }

    public double getYSize() {
        return size.getY();
    }

    public double getZSize() {
        return size.getZ();
    }

    public ParticleDictionary getDictionary() {
        return dict;
    }

    public double getTemperature() {
        double sumKE = 0;
        for (Particle part : particles) {
            double speed = part.getVelocity().magnitude();
            sumKE += .5 * part.getMass() * speed * speed;
        }

        return sumKE / particles.size() * 2 / 3 / 1.38E-23;
    }

    /**
     * @param part The Particle to add to the container
     */
    public void addParticle(Particle part) {
        toBeAdded.add(part);
    }

    public void removeParticle(Particle part) {
        toBeRemoved.add(part);
    }

    /**
     * @param generator The RandomGenerator used to create the random
     * distribution
     * @param number The number of particles to create
     */
    public void addRandomParticles(RandomGenerator generator, int number, String name) {
        for (int i = 0; i < number; ++i) {
            addParticle(generator.getNextParticle(name));
        }
    }

    private void checkCollisions(Particle particle) {
        List<Particle> closeParticles = new ArrayList<Particle>();
        closeParticles.add(particle);

        for (Particle part : particles) {
            if (Math.abs(particle.getPosition().getX() - part.getPosition().getX()) < reactionRadius
                    || Math.abs(particle.getPosition().getY() - part.getPosition().getY()) < reactionRadius
                    || Math.abs(particle.getPosition().getZ() - part.getPosition().getZ()) < reactionRadius) {
                Vector3 diff = Vector3.subtract(particle.getPosition(), part.getPosition());
                if (!toBeRemoved.contains(part) && part != particle && diff.magnitude() < reactionRadius) {
                    closeParticles.add(part);
                }
            }

        }

        if (closeParticles.size() > 1) {
            doReaction(closeParticles);
        }
    }

    public Map<String, Integer> getParticleMap() {
        Map<String, Integer> response = new HashMap<String, Integer>();
        for (Particle p : particles) {
            response.put(p.getName(), (response.containsKey(p.getName()) ? response.get(p.getName()) + 1 : 1));
        }
        return response;
    }

    private boolean doReaction(List<Particle> list) {
        Particle reactingParticle = list.get(0);
        Map<String, Integer> newParticles = reactionsDictionary.reactionResults(list);
        ArrayList<Particle> particlesToRemove = reactionsDictionary.getParticlesToRemove();
        HashMap<String, Double> catalysts = reactionsDictionary.getCatalysts(reactingParticle);
        double AE = reactionsDictionary.getAE(reactingParticle);
        for (Double deltaAE : catalysts.values()) {
            AE -= deltaAE;
        }

        if (newParticles.isEmpty() == false) {
            Vector3 totalP = calculateP(particlesToRemove);
            double totalKE = calculateKE(particlesToRemove);
            if (totalKE >= AE) {
                for (String key : newParticles.keySet()) {
                    int numParticles = newParticles.get(key);
                    while (numParticles > 0) {
                        Vector3 newVelocity = getRandomDirection(totalP.scale(1.0 / particlesToRemove.size()).scale(1.0 / dict.getMass(key)).magnitude());
                        addParticle(dict.makeParticle(reactingParticle.getPosition(), newVelocity, key));
                        numParticles--;
                    }
                }
                for (Particle p : particlesToRemove) {
                    removeParticle(p);
                }
                return true;
            }
        }
        return false;

    }

    public double calculateKE(ArrayList<Particle> array) {
        double ke = 0;
        for (Particle p : array) {
            ke += Math.pow(p.getVelocity().magnitude(), 2) * dict.getMass(p.getName());
        }
        return ke;
    }

    public Vector3 calculateP(ArrayList<Particle> array) {
        Vector3 momentum = new Vector3();
        for (Particle p : array) {
            momentum = Vector3.add(momentum, p.getVelocity().scale(dict.getMass(p.getName())));
        }
        return momentum;
    }

    public Vector3 getRandomDirection(double speed) {
        double phi = random.nextDouble() * 2 * Math.PI;
        double theta = random.nextDouble() * Math.PI;
        Vector3 randDirection = Vector3.sphericalVector(speed, phi, theta);
        return randDirection;
    }

    public List<Particle> getParticles() {
        return particles;
    }

    /**
     * This advances all the particles by one time step
     * @param deltaTime The amount of time to advance the particle collection by
     */
    public void advanceParticles(double deltaTime) {
        for (Particle proj : particles) {
            proj.update(deltaTime);
            checkParticle(proj);
            checkCollisions(proj);
        }
        updateParticleList();
    }

    public void updateParticleList() {
        for (Particle part : toBeRemoved) {
            particles.remove(part);
        }
        for (Particle part : toBeAdded) {
            particles.add(part);
        }

        toBeRemoved.clear();
        toBeAdded.clear();
    }

    private void checkParticle(Particle particle) {
        if (particle.getPosition().getX() < 0 || particle.getPosition().getX() > size.getX()) {
            particle.getVelocity().setX(-particle.getVelocity().getX());
        }
        if (particle.getPosition().getY() < 0 || particle.getPosition().getY() > size.getY()) {
            particle.getVelocity().setY(-particle.getVelocity().getY());
        }
        if (particle.getPosition().getZ() < 0 || particle.getPosition().getZ() > size.getZ()) {
            particle.getVelocity().setZ(-particle.getVelocity().getZ());
        }
    }

    /**
     * This runs the whole simulation
     * @param deltaTime The amount of time to advance the simulation each step
     */
    public void run(double deltaTime) {
        updateParticleList();

        /*ParticleVisualizer viz = new ParticleVisualizer(this, deltaTime);
		
		JFrame frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.add(viz);
		frame.setSize(viz.getWidth(), viz.getHeight());

		frame.setVisible(true);		
         */
    }
}
