package betterChemicalReactions;

import java.awt.Color;
import java.util.Random;


/**
 * A Particle that moves randomly every mean free time
 */
public class RandomWalkParticle extends Particle {

	static private Random random = new Random();
	
	private double meanFreeTime;
	private double timeCounter = 0;
	
	/**
	 * @param position The original position of the Particle
	 * @param velocity The initial velocity of the Particle
	 * @param mass The mass of the particle
	 * @param color The color of the particle in the display
	 * @param meanFreeTime The average time between collisions
	 */
	public RandomWalkParticle(Vector3 position, Vector3 velocity, double mass, Color color, String name, double meanFreeTime) {
		super(position, velocity, mass, color, name);
		this.meanFreeTime = meanFreeTime;
	}

	private void collision() {
		double speed = getVelocity().magnitude();
		
		double ranPhi = random.nextDouble() * 2 * Math.PI;
		double ranTheta = random.nextDouble() * Math.PI;
		
		setVelocity(Vector3.sphericalVector(speed, ranPhi, ranTheta));
	}
	
	@Override
	protected void advancePosition(double timeIncrement) {
		timeCounter += timeIncrement;
		if (timeCounter > meanFreeTime) {
			collision();
			timeCounter = 0;
		}
		super.advancePosition(timeIncrement);
	}

}
