package betterChemicalReactions;

public class Vector3 {
	
	private double x;
	private double y;
	private double z;

	/**
	 * Default constructor - initializes to null vector
	 */
	public Vector3() {
		x = 0;
		y = 0;
		z = 0;
	}

	public Vector3(double x, double y, double z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
	/**
	 * Vector copy constructor
	 * @param rhs - Vector3 object to copy
	 */
	public Vector3(Vector3 rhs) {
		x = rhs.x;
		y = rhs.y;
		z = rhs.z;
	}
	
	public String toString() {
		return new String(x + "\t" + y + '\t' + z + '\t' + magnitude());
	}
	
	/**
	 * @return x coordinate
	 */
	public double getX() {
		return x;
	}

	/**
	 * @param x - New x coordinate
	 */
	public void setX(double x) {
		this.x = x;
	}

	/**
	 * @return y coordinate
	 */
	public double getY() {
		return y;
	}

	/**
	 * @param y - New y coordinate
	 */
	public void setY(double y) {
		this.y = y;
	}

	/**
	 * @return z coordinate
	 */
	public double getZ() {
		return z;
	}

	public void setZ(double z) {
		this.z = z;
	}
	
	public Vector3 add(Vector3 v2) {
		return add(this, v2);
	}

	public Vector3 subtract(Vector3 v2) {
		return subtract(this, v2);
	}
	
	public Vector3 scale(double scaler) {
		return scale(this, scaler);
	}
	
	/**
	 * @return the magnitude of the vector
	 */
	public double magnitude() {
		return Math.sqrt(x * x + y * y + z * z);
	}
	
	/**
	 * @return the azimuthal angle of the vector - the angle from the x axis in the x-y plane (phi in physics, theta in math)
	 */
	public double azimuthal() {
		return Math.atan2(y, x);
	}
	
	/**
	 * @return the rpolar angle of the vector - the angle of declination from the z axis (theta in physics, phi in math)
	 */
	public double theta() {
		return Math.atan(Math.sqrt(x * x + y * y) / z);
	}

	public static Vector3 add(Vector3 v1, Vector3 v2) {
		return new Vector3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}
	
	public static Vector3 inverse(Vector3 v1) {
		return new Vector3(-v1.x, -v1.y, -v1.z);
	}
	
	public static Vector3 subtract(Vector3 v1, Vector3 v2) {
		return new Vector3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}
	
	public static Vector3 scale(Vector3 v1, double scaler) {
		return new Vector3(v1.x * scaler, v1.y * scaler, v1.z * scaler);
	}
	
	public static double dot(Vector3 v1, Vector3 v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}
	
	public static Vector3 cross(Vector3 v1, Vector3 v2) {
		return new Vector3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}
	
	/**
	 * @return a null vector
	 */
	public static Vector3 nullVector() {
		return new Vector3(0, 0, 0);
	}
	
	/**
	 * @param v1 - the input vector
	 * @return a unit vector that points in the same direction as the input vector
	 */
	public static Vector3 unitVector(Vector3 v1) {
		if (v1.magnitude() == 0) {
			return nullVector();
		} else {
			return scale(v1, 1 / v1.magnitude());
		}
	}
	
	public static Vector3 sphericalVector(double r, double phi, double theta) {
		double x = r * Math.cos(phi) * Math.sin(theta);
		double y = r * Math.sin(phi) * Math.sin(theta);
		double z = r * Math.cos(theta);

		return new Vector3(x, y, z);
	}
	
	/**
	 * @param origin The origin of the axis about which the vector is to be reflected
	 * @param axis The axis of reflection that goes through the origin
	 * @return A vector that has been reflected about the given axis
	 */
	public Vector3 reflectParticle(Vector3 origin, Vector3 axis) {
		Vector3 unitN = Vector3.unitVector(axis);
		
		// From MathWorld:
		// reflection = -this + 2 * origin + 2 * unitN * ((this - origin) dot unitN)
		
		Vector3 response = new Vector3(this).scale(-1);
		response = response.add(origin.scale(2));
		response = response.add(unitN.scale(2 * Vector3.dot(Vector3.subtract(this, origin), unitN)));
		
		return response;
	}
	
}
