package betterChemicalReactions;

import java.util.Random;

/**
 * A class to generate a random distribution of Particles according to a programmed rule
 */
abstract public class RandomGenerator {

	private double mass;
	
	private ParticleContainer grid;
	
	/**
	 * @param mass The mass of each particle
	 * @param color The color of the particle in the display
	 * @param meanFreeTime The mean free time for each particle
	 * @param grid The ParticleContainer to which the particle will be added
	 */
	public RandomGenerator(ParticleContainer grid) {
		this.grid = grid;
	}
	
	static private Random random = new Random();

	/**
	 * @return A random number generator
	 */
	protected Random getRandom() {
		return random;
	}
	
	protected double getMass() {
		return mass;
	}
	
	/**
	 * @param grid The container of Particles
	 * @return A randomly generated position vector within a ParticleContainer
	 */
	protected Vector3 randomPosition(ParticleContainer grid) {
		double x = getRandom().nextDouble() * grid.getXSize();
		double y = getRandom().nextDouble() * grid.getYSize();
		double z = getRandom().nextDouble() * grid.getZSize();
		
		return new Vector3(x, y, z);
	}
	
	/**
	 * @return A unit vector that points in a random direction
	 */
	protected Vector3 randomDirection() {
		double phi = getRandom().nextDouble() * 2 * Math.PI;
		double theta = getRandom().nextDouble() * Math.PI;
		
		return Vector3.sphericalVector(1, phi, theta);
	}
	
	/**
	 * @return A randomly generated speed based on an algorithm in a derived class
	 */
	abstract protected double getSpeed();
	
	/**
	 * @return A randomly generated Projectile using the getSpeed() function
	 */
	public Particle getNextParticle(String name) {
		mass = grid.getDictionary().getMass(name);
		Vector3 velocity = randomDirection().scale(getSpeed());
		
		return grid.getDictionary().makeParticle(randomPosition(grid), velocity, name);
	}
}
