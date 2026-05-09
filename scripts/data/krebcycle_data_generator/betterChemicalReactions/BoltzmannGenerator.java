package betterChemicalReactions;

public class BoltzmannGenerator extends RandomGenerator {

	private double temp;
	static private double BOLTZMANN = 1.38e-23;
	static private double increment = 1;
	
	public BoltzmannGenerator(ParticleContainer grid, double temp) {
		super(grid);
		this.temp = temp;
	}

	@Override
	protected double getSpeed() {
		double ran = getRandom().nextDouble();
		double velocity = 0;
		double cumulative = 0;
		while (cumulative < ran) {
			velocity += increment;
			cumulative += increment * getProb(velocity);
		}
		
		return velocity;
	}
	
	private double getProb(double velocity) {
		return 4 * Math.PI * Math.pow(getMass() / (2 * Math.PI * BOLTZMANN * temp), 1.5) * velocity * velocity
				* Math.exp(-getMass() * velocity * velocity / (2 * BOLTZMANN * temp));
	}

}
